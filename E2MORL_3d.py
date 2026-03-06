import argparse
import os
import time
import gym
import numpy as np
import torch
import half_cheetah_v3
import hopper_v3
import humanoid_v3
import ant_v3
import walker2d_v3
import ant3d_v3
import MOTD7
from torch.utils.tensorboard import SummaryWriter
import copy
import random

pop_size = 20
f_update_t = np.zeros(pop_size, dtype=int)
UT = np.zeros([20, 20], dtype=float)
init_UT = -9999
UT += init_UT
temp_buffer = []
evo_exp_enabled_t = 300000
max_used_num = 5


def calculate_crowding_distance(fitness_array, new_fitness):
	fitness_array = np.vstack([fitness_array, new_fitness])
	m, n = fitness_array.shape

	crowding_distances = np.zeros(m)

	for i in range(n):
		sorted_indices = np.argsort(fitness_array[:, i])
		sorted_fitness = fitness_array[sorted_indices, i]

		crowding_distances[sorted_indices[0]] = np.inf
		crowding_distances[sorted_indices[-1]] = np.inf

		for j in range(1, m - 1):
			if np.isinf(crowding_distances[sorted_indices[j]]):
				continue
			crowding_distances[sorted_indices[j]] += (
					(sorted_fitness[j + 1] - sorted_fitness[j - 1]) /
					(sorted_fitness[-1] - sorted_fitness[0] + 1e-9)
			)

	return crowding_distances


def replace_individual(idx, new_fit, t, buffer, w):
	global pop_exp
	fitness[idx] = new_fit
	f_update_t[idx] = t
	pop_exp[idx] = copy.deepcopy(buffer)
	pop_exp_w[idx] = w
	used_num[idx] = 0


def update_pop(new_fit, t, buffer, w):
	for i in range(pop_size):
		if np.all(fitness[i] <= new_fit):
			replace_individual(i, new_fit, t, buffer, w)
			return
		elif np.all(fitness[i] >= new_fit):
			return
	crowding_distances = calculate_crowding_distance(fitness, new_fit)
	min_idx = np.argmin(crowding_distances)
	if min_idx != pop_size:
		replace_individual(min_idx, new_fit, t, buffer, w)


def exp_crossover(t):
	p1, p2 = random.sample(range(pop_size), 2)
	c1_w = np.zeros(reward_dim)
	c2_w = np.zeros(reward_dim)
	# crossover
	for i in range(0, int(reward_dim/2)):
		c1_w[i] = pop_exp_w[p1][i]
		c2_w[i] = pop_exp_w[p2][i]
	for i in range(int(reward_dim/2), reward_dim):
		c1_w[i] = pop_exp_w[p2][i]
		c2_w[i] = pop_exp_w[p1][i]
	# normalization
	c1_w /= (c1_w.sum() + 1e-9)
	c2_w /= (c2_w.sum() + 1e-9)

	# evaluation and update
	evaluation_and_update(p1, c1_w, t)
	evaluation_and_update(p2, c2_w, t)


def exp_mutation(t):
	p = random.sample(range(pop_size), 1)[0]
	# mutation
	c_w = pop_exp_w[p] + np.abs(np.random.randn(reward_dim)*0.1)
	# normalization
	c_w /= (c_w.sum() + 1e-9)
	# evaluation and update
	evaluation_and_update(p, c_w, t)


def evaluation_and_update(p, c_w, t):
	idx0, idx1 = min(int(c_w[0] * 20), UT.shape[0] - 1), min(int(c_w[1] * 20), UT.shape[1] - 1)
	if np.sum(fitness[p] * c_w) > UT[idx0][idx1] and UT[idx0][idx1] != init_UT and used_num[p] < max_used_num:
		# pop_exp -> RL
		if t >= evo_exp_enabled_t:
			for i in range(len(pop_exp[p])):
				exp = pop_exp[p][i]
				RL_agent.replay_buffer.add(exp[0], exp[1], exp[2], exp[3], c_w, exp[5])
			used_num[p] += 1
		# try update parent's weight
		p_w = pop_exp_w[p]
		idx2, idx3 = min(int(p_w[0] * 20), UT.shape[0] - 1), min(int(p_w[1] * 20), UT.shape[1] - 1)
		if np.sum(fitness[p] * c_w) - UT[idx0][idx1] > np.sum(fitness[p] * p_w) - UT[idx2][
			idx3]:  # if child's weight can bring more improvement
			pop_exp_w[p] = c_w  # update parent's weight

		i, j = fitness[p].argmax(), fitness[p].argmin()
		if abs(fitness[p][i] / (fitness[p][j] + 1e-8)) > 2 * c_w[i] / (c_w[j] + 1e-8):  # normalize Q value if rewards/weights in generated experiences are unbalanced
			RL_agent.base_dim = [j]
			RL_agent.other_dims = []
			for k in range(reward_dim):
				if k != i and k != j:
					RL_agent.other_dims.append(k)


def train_online(RL_agent, env, eval_env, args):
	global temp_buffer, UT
	evals = []
	start_time = time.time()
	allow_train = False
	state, ep_finished = env.reset(), False
	train_steps = 0

	weights = np.random.rand(env.reward_num)
	weights /= weights.sum()
	state = np.append(state, weights)

	ep_total_reward, ep_timesteps, ep_num = np.zeros(env.reward_num), 0, 1

	for t in range(int(args.max_timesteps+1)):
		maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args)
		
		if allow_train:
			action = RL_agent.select_action(np.array(state))
		else:
			action = env.action_space.sample()

		next_state, vector_reward, ep_finished, _ = env.step(action)
		next_state = np.append(next_state, weights)

		ep_total_reward += vector_reward
		ep_timesteps += 1

		done = float(ep_finished) if ep_timesteps < env.max_episode_steps else 0

		RL_agent.replay_buffer.add(state, action, next_state, vector_reward, weights, done)
		if allow_train and t < evo_exp_enabled_t:
			for _ in range(3):
				her_weights = np.random.rand(env.reward_num)
				her_weights /= weights.sum()
				RL_agent.replay_buffer.add(state, action, next_state, vector_reward, her_weights, done)
		temp_buffer.append([state, action, next_state, vector_reward, weights, done])

		state = next_state

		if allow_train:
			train_steps += 1

		if ep_finished:
			f.write(f"\nTotal T: %s Episode Num: %s Episode T: %s Weights: %s Reward: %s    " % (t+1, ep_num, ep_timesteps, weights, ep_total_reward))
			f.flush()
			update_pop(ep_total_reward, t, temp_buffer, weights)

			if t >= args.timesteps_before_training:
				allow_train = True
			idx0, idx1 = min(int(weights[0] * 20), UT.shape[0] - 1), min(int(weights[1] * 20), UT.shape[1] - 1)
			if np.sum(ep_total_reward * weights) > UT[idx0][idx1]:
				UT[idx0][idx1] = np.sum(ep_total_reward * weights)
			exp_crossover(t)
			exp_mutation(t)
			for _ in range(train_steps):
				RL_agent.train(f)

			train_steps = 0
			state, done = env.reset(), False
			weights = np.random.rand(env.reward_num)
			weights /= weights.sum()
			state = np.append(state, weights)
			ep_total_reward, ep_timesteps = np.zeros(env.reward_num), 0
			ep_num += 1
			temp_buffer = []

		if t % 100000 == 0:
			save_filename = f"{model_path}/{run_name}__{t}"
			if t >= 1500000:
				test_models.append(save_filename)
			torch.save(RL_agent.actor.state_dict(), save_filename + "_actor")
			torch.save(RL_agent.fixed_encoder.state_dict(), save_filename + "_fixed_encoder")


def maybe_evaluate_and_print(RL_agent, eval_env, evals, t, start_time, args):
	if t % args.eval_freq == 0:
		f.write("---------------------------------------\n")
		f.write(f"Evaluation at %s time steps\n" % t)
		f.write(f"Total time passed: %.2f min(s)\n" % (round((time.time()-start_time)/60.,2)))
		if env.reward_num == 1:
			w = [1]*10
		elif env.reward_num == 2:
			w = [[0.05, 0.95], [0.1, 0.9], [0.15, 0.85], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6], [0.5, 0.5], [0.6, 0.4], [0.7, 0.3], [0.8, 0.2], [0.9, 0.1], [1.0, 0.0]]   # 2 objectives
		elif env.reward_num == 3:
			w = [[0.9, 0.05, 0.05], [0.7, 0.15, 0.15], [0.5, 0.25, 0.25], [0.33, 0.33, 0.33], [0.25, 0.5, 0.25], [0.25, 0.25, 0.5], [0.15, 0.7, 0.15], [0.15, 0.15, 0.7], [0.05, 0.9, 0.05], [0.05, 0.05, 0.9]]
		elif env.reward_num == 4:
			w = [[0.9, 0.033, 0.033, 0.033], [0.7, 0.1, 0.1, 0.1], [0.5, 0.166, 0.166, 0.166], [0.033, 0.9,  0.033, 0.033], [0.033, 0.033, 0.9, 0.033], [0.033, 0.033, 0.033, 0.9], [0.25, 0.25, 0.25, 0.25], [0.166, 0.5, 0.166, 0.166], [0.166, 0.166, 0.5, 0.166], [0.166, 0.166, 0.166, 0.5]]
		total_reward = np.zeros((len(w), env.reward_num))
		for ep in range(len(w)):
			state, done = eval_env.reset(), False
			weights = w[ep]
			state = np.append(state, weights)
			while not done:
				action = RL_agent.select_action(np.array(state), use_exploration=False)
				state, vector_reward, done, _ = eval_env.step(action)
				state = np.append(state, weights)
				total_reward[ep] += vector_reward
			f.write(f"weights: %s rewards %s\n" % (weights, total_reward[ep]))
			tag = "charts/" + str(weights)
			writer.add_scalar(tag, np.dot(total_reward[ep], weights), t)
		
		f.write("---------------------------------------\n")


def get_test_weights(obj_num):
	weights_list = []
	if obj_num == 2:
		eval_step_size = 0.001
		for w1 in range(0, int(1 / eval_step_size)):
			weights_list.append(np.array([w1 * eval_step_size,  1 - w1 * eval_step_size]))
	elif obj_num == 3:
		eval_step_size = 0.01
		for w1 in range(0, int(1/eval_step_size)+1):
			for w2 in range(0, int(1/eval_step_size)+1-w1):
				weights_list.append(np.array([w1 * eval_step_size, w2 * eval_step_size, 1 - (w1 + w2) * eval_step_size]))
	return weights_list


test_models = []
def test_model():
	global cwd, env_tag
	objv_path = f'{cwd}/objective_values/E2MORL/{env_tag}/{run_name}'
	if not os.path.exists(objv_path):
		os.makedirs(objv_path)
	for model_name in test_models:
		objv_f = open(f'{objv_path}/{model_name.split("/")[-1]}.txt', 'w')
		RL_agent.actor.load_state_dict(torch.load(model_name + "_actor", map_location=torch.device('cpu')))
		RL_agent.fixed_encoder.load_state_dict(torch.load(model_name + "_fixed_encoder", map_location=torch.device('cpu')))
		objective_values = []
		weights_list = get_test_weights(reward_dim)
		for agent in [RL_agent]:
			for weights in weights_list:
				total_reward = np.zeros(eval_env.reward_num)
				state, ep_finished = eval_env.reset(), False
				state = np.append(state, weights)
				step = 0
				while not ep_finished:
					action = agent.select_action(np.array(state), use_exploration=False)
					state, vector_reward, ep_finished, _ = eval_env.step(action)
					state = np.append(state, weights)
					total_reward += vector_reward
					step += 1
				objective_values.append(total_reward)
		for i in range(len(objective_values)):
			if reward_dim == 2:
				objv_f.write("%.2f %.2f\n" % (objective_values[i][0], objective_values[i][1]))
			elif reward_dim == 3:
				objv_f.write("%.2f %.2f %.2f\n" % (objective_values[i][0], objective_values[i][1], objective_values[i][2]))


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	# RL
	parser.add_argument("--env", default="MO_half_cheetah-v0", type=str)
	parser.add_argument("--seed", default=0, type=int)
	# Evaluation
	parser.add_argument("--timesteps_before_training", default=25e3, type=int)
	parser.add_argument("--eval_freq", default=5e3, type=int)
	parser.add_argument("--eval_eps", default=10, type=int)
	parser.add_argument("--max_timesteps", default=1.5e6, type=int)
	# File
	args = parser.parse_args()
	seed = args.seed
	env_tag = args.env
	time_str = time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time()))
	run_name = f"{env_tag}__E2MORL_3d__{seed}__{time_str}"
	writer = SummaryWriter(f"runs/{env_tag}/{run_name}")
	cwd = os.getcwd()
	path = f'{cwd}/log/{env_tag}/{run_name}'
	model_path = f'{cwd}/models/{env_tag}/{run_name}'
	os.makedirs(model_path, mode=0o777)
	os.makedirs(path, mode=0o777)
	log_name = run_name + '_log.txt'
	f = open(f'{path}/{log_name}', 'w')

	env = gym.make(args.env)
	eval_env = gym.make(args.env)

	f.write("---------------------------------------\n")
	f.write(f"Algorithm: E2MORL, Env: %s, Seed: %s\n" %(args.env, args.seed))
	f.write("---------------------------------------\n")

	env.seed(args.seed)
	env.action_space.seed(args.seed)
	eval_env.seed(args.seed+100)
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0] + env.reward_num
	action_dim = env.action_space.shape[0]
	reward_dim = env.reward_num
	max_action = float(env.action_space.high[0])

	RL_agent = MOTD7.Agent(state_dim, action_dim, reward_dim, max_action)
	pop_exp = []
	pop_exp_w = np.zeros([pop_size, reward_dim])
	used_num = np.zeros(pop_size, dtype=int)
	for _ in range(pop_size):
		pop_exp.append([])
	fitness = np.zeros([pop_size, reward_dim])
	fitness -= 99999

	train_online(RL_agent, env, eval_env, args)

	test_model()

