#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  8 22:04:17 2023

@author: qiangwang
"""
import os
import math
import torch
import argparse
import json
import random
import numpy as np
from typing import Any
from datetime import datetime


def get_normalized_score(env_name, score):
    ref_min_score = REF_MIN_SCORE[env_name]
    ref_max_score = REF_MAX_SCORE[env_name]
    return (score - ref_min_score) / (ref_max_score - ref_min_score)


def evaluate_on_environment_DT(env, n_trials, epsilon: float = 0.0,
                            render: bool = False, BC_regs = None, use_bc_reg=False, action_pe = None):
    observation_shape = env.observation_space.shape
    is_image = len(observation_shape) == 3
    def scorer(algo, *args: Any) -> float:
        episode_rewards = []
        for _ in range(n_trials):
            observation, reward = env.reset(), 0.0
            observation = observation[0]
            episode_reward = 0.0
            while True:
                if np.random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    action = algo.predict(observation, reward)
                observation, reward, done, truncated, _ = env.step(action)
                episode_reward += reward
                if render:
                    env.render()
                if done or truncated:
                    break
            episode_rewards.append(episode_reward)
        return float(np.mean(episode_rewards))
    return scorer


def get_centroid(mode, action_data, task, dataset, ep_num = 0):
    if mode == 'set_center':
        center = np.mean(action_data, axis=0)
    elif mode == 'ep_center':
        ep_count = 0
        temp_action = []
        for idx in range(action_data.shape[0]):
            temp_action.append(action_data[idx])
            if dataset['terminals'][idx]:
                if ep_count >= ep_num:
                    break
                else:
                    temp_action = []
                ep_count += 1
        temp_action = np.array(temp_action)
        center = np.mean(temp_action, axis=0)
    elif mode == 'space_center':
        if task == 'rrc_lift' or 'rrc_push':
            center = np.array([0]*9)
        if task == 'walker2d':
            center = np.array([0]*6)
        if task == 'hopper':
            center = np.array([0]*3)
        if task == 'pen':
            center = np.array([0]*24)
        if task == 'hammer':
            center = np.array([0]*26)
    return center


def geometric_mean(count_1, num):
    if not all(x > 0 for x in count_1):
        raise ValueError("All values must the larger than 0")
    log_sum = sum(math.log(x) for x in count_1)
    mean_of_logs = log_sum / num
    return math.exp(mean_of_logs)


def calculate_column_entropy(arr, bins=10):
    flat_arr = arr.flatten()
    counts, _ = np.histogram(flat_arr, bins=bins)
    probabilities = counts / len(flat_arr)
    probabilities = probabilities[probabilities > 0]
    entropy = -np.sum(probabilities * np.log2(probabilities))
    return entropy


def save_params(args):
    args_dict = vars(args)
    with open(f'{args.save_path}/params.json', 'w') as f:
        json.dump(args_dict, f, indent=4)
        

def load_params(path):
    with open(path, 'r') as f:
        loaded_args_dict = json.load(f)
        loaded_args = argparse.Namespace(**loaded_args_dict)
    return loaded_args
        
        
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def directory_handler(args):
    if not args.save_path:
        proj_root_path = os.path.split(os.path.realpath(__file__))[0]
        args.save_path = f'{proj_root_path}/save'
    if os.path.split(args.save_path)[-1] != args.exp_name:
        args.save_path = f'{args.save_path}/{args.exp_name}'
    current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    args.save_path = f'{args.save_path}/{args.exp_name}-{current_time}'
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    print(args.save_path)
    return args


REF_MIN_SCORE = {
    'pen-human-v0' : 96.262799 ,
    'pen-cloned-v0' : 96.262799 ,
    'pen-expert-v0' : 96.262799 ,
    'hammer-human-v0' : -274.856578 ,
    'hammer-cloned-v0' : -274.856578 ,
    'hammer-expert-v0' : -274.856578 ,
    'relocate-human-v0' : -6.425911 ,
    'relocate-cloned-v0' : -6.425911 ,
    'relocate-expert-v0' : -6.425911 ,
    'door-human-v0' : -56.512833 ,
    'door-cloned-v0' : -56.512833 ,
    'door-expert-v0' : -56.512833 ,
    'HalfCheetah' : -280.178953 ,
    'halfcheetah-medium-v0' : -280.178953 ,
    'halfcheetah-expert-v0' : -280.178953 ,
    'halfcheetah-medium-replay-v0' : -280.178953 ,
    'halfcheetah-medium-expert-v0' : -280.178953 ,
    'walker2d-random-v0' : 1.629008 ,
    'walker2d-medium-v0' : 1.629008 ,
    'walker2d-expert-v0' : 1.629008 ,
    'walker2d-medium-replay-v0' : 1.629008 ,
    'walker2d-medium-expert-v0' : 1.629008 ,
    'hopper-random-v0' : -20.272305 ,
    'hopper-medium-v0' : -20.272305 ,
    'hopper-expert-v0' : -20.272305 ,
    'hopper-medium-replay-v0' : -20.272305 ,
    'hopper-medium-expert-v0' : -20.272305 ,
    'ant-random-v0' : -325.6,
    'ant-medium-v0' : -325.6,
    'ant-expert-v0' : -325.6,
    'ant-medium-replay-v0' : -325.6,
    'ant-medium-expert-v0' : -325.6,
}

REF_MAX_SCORE = {
    'pen-human-v0' : 3076.8331017826877 ,
    'pen-cloned-v0' : 3076.8331017826877 ,
    'pen-expert-v0' : 3076.8331017826877 ,
    'hammer-human-v0' : 12794.134825156867 ,
    'hammer-cloned-v0' : 12794.134825156867 ,
    'hammer-expert-v0' : 12794.134825156867 ,
    'relocate-human-v0' : 4233.877797728884 ,
    'relocate-cloned-v0' : 4233.877797728884 ,
    'relocate-expert-v0' : 4233.877797728884 ,
    'door-human-v0' : 2880.5693087298737 ,
    'door-cloned-v0' : 2880.5693087298737 ,
    'door-expert-v0' : 2880.5693087298737 ,
    'HalfCheetah' : 12135.0 ,
    'halfcheetah-medium-v0' : 12135.0 ,
    'halfcheetah-expert-v0' : 12135.0 ,
    'halfcheetah-medium-replay-v0' : 12135.0 ,
    'halfcheetah-medium-expert-v0' : 12135.0 ,
    'walker2d-random-v0' : 4592.3 ,
    'walker2d-medium-v0' : 4592.3 ,
    'walker2d-expert-v0' : 4592.3 ,
    'walker2d-medium-replay-v0' : 4592.3 ,
    'walker2d-medium-expert-v0' : 4592.3 ,
    'hopper-random-v0' : 3234.3 ,
    'hopper-medium-v0' : 3234.3 ,
    'hopper-expert-v0' : 3234.3 ,
    'hopper-medium-replay-v0' : 3234.3 ,
    'hopper-medium-expert-v0' : 3234.3 ,
    'ant-random-v0' : 3879.7,
    'ant-medium-v0' : 3879.7,
    'ant-expert-v0' : 3879.7,
    'ant-medium-replay-v0' : 3879.7,
    'ant-medium-expert-v0' : 3879.7,
}