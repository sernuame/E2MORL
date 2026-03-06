# E²MORL
Code for "Experience Evolution-Guided Multi-Objective Reinforcement Learning".

# Dependency
python 3.8.13, pytorch 1.8.2, gym 0.23.1, mujoco-py 2.1.2.14, pymoo 0.5.0, numpy 1.21.5, matplotlib 3.7.5

# Train
python E2MORL.py --env 'MO_half_cheetah-v0' --seed 0  

# Visualize PF approximation from the evaluation results 
python visualize.py --env MO_half_cheetah-v0 --path MO_half_cheetah-v0__0.txt
