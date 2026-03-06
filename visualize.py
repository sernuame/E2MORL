import argparse
import os
import time
import numpy as np
import half_cheetah_v3
import hopper_v3
import humanoid_v3
import ant_v3
import walker2d_v3
import copy
from pymoo.factory import get_performance_indicator
import matplotlib.pyplot as plt
from pymoo.config import Config

Config.show_compile_hint = False


def get_HV_and_SP():
    if args.env == 'MO_half_cheetah-v0':
        ref_point = np.array([0, -4000])
    elif args.env == 'MO_walker-v0':
        ref_point = np.array([0, -2500])
    elif args.env == 'MO_hopper-v0':
        ref_point = np.array([0, -1000])
    elif args.env == 'MO_humanoid-v0':
        ref_point = np.array([0, -1500])
    elif args.env == 'MO_ant-v0':
        ref_point = np.array([0, -3000])
    else:
        ref_point = np.array([0, -1000])
    hv = get_performance_indicator("hv", ref_point=ref_point)

    path = f'{cwd}/objective_values/E2MORL/{args.env}/{args.path}'
    o2 = (get_objectives_from_file(path, separator=" "))
    HV = (hv.do(-np.array(o2) + 2 * ref_point))
    SP = get_SP(o2)
    print("HV:%s" % HV)
    print("SP:%s" % SP)

    draw_scatter(o2, no_dominated=True)


def get_SP(objective_values):
    o = get_no_dominated_solutions(objective_values)
    sp = 0
    for i in range(2):
        sp_i = 0
        sorted_o = sorted(o, key=lambda x: x[i], reverse=True)
        for j in range(len(o) - 1):
            sp_i += pow(sorted_o[j][i] - sorted_o[j + 1][i], 2)
        sp += sp_i / (len(o) - 1)
    return sp


def get_no_dominated_solutions(objective_values):
    nds = []
    sorted_o = sorted(objective_values, key=lambda x: x[0], reverse=True)
    max2 = -99999
    for i in range(0, len(sorted_o)):
        if sorted_o[i][1] > max2:
            nds.append([sorted_o[i][0], sorted_o[i][1]])
            max2 = sorted_o[i][1]
    return np.array(nds)


def get_objectives_from_file(path, separator=" ", rounds=False):
    f = open(path, 'r')
    o2 = []
    line = f.readline()
    while line:
        lines = line.split(separator)
        if not rounds:
            o2.append([float(lines[0]), float(lines[1])])
        else:
            o2.append([int(float(lines[0])), int(float(lines[1]))])
        line = f.readline()
    return np.array(o2)


def draw_scatter(o1,  no_dominated=True):
    plt.rcParams.update({'font.size': 18})
    plt.rcParams.update({'font.family': 'Times New Roman'})
    if no_dominated:
        o1 = get_no_dominated_solutions(o1)
    plt.scatter(o1[:, 1], o1[:, 0], color='red', marker='o', label='E2MORL', alpha=0.4)

    plt.xlabel('Control Cost')
    plt.ylabel('Forward Reward')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # RL
    parser.add_argument("--env", default="MO_half_cheetah-v0", type=str)
    parser.add_argument("--path", default="MO_half_cheetah-v0__0.txt", type=str)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    seed = args.seed
    env_tag = args.env
    cwd = os.getcwd()

    np.random.seed(args.seed)
    get_HV_and_SP()
