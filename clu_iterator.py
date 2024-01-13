#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiangwang
"""
import utils
import pickle
import random
import itertools
import numpy as np
from pu_filter import runner
from sklearn.preprocessing import normalize


def total_distance(X):
    return np.sum(np.sqrt(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=2)))


def find_min_distance_subcluster(data, cluster_size, num_iterations):
    min_distance = float('inf')
    best_cluster = None
    for i in range(int(num_iterations)):
        indices = np.random.choice(len(data), size=cluster_size, replace=False)
        cluster = data[indices]
        dist = total_distance(cluster)
        if dist < min_distance:
            min_distance = dist
            best_cluster = indices
    return best_cluster, min_distance


def find_min_distance_cluster_exhaustive(data, cluster_size):
    min_distance = float('inf')
    best_cluster = None
    for subset in itertools.combinations(range(len(data)), cluster_size):
        cluster = data[np.array(subset)]
        dist = total_distance(cluster)
        if dist < min_distance:
            min_distance = dist
            best_cluster = subset
    return best_cluster, min_distance


class RunClustering():
    def __init__(self, 
                 raw_dataset_path,
                 opti_ratio_init=0.04,
                 opti_ratio_increment=0.02,
                 opti_iter_num = 2,
                 action_norm = 'l1',
                 init_sample_num = 100,
                 init_search_num = 5e5,
                 corse_init_center_size = 6,
                 fine_init_center_size = 4,
                 args = None):
        self.raw_dataset_path = raw_dataset_path
        self.opti_iter_num = opti_iter_num
        self.opti_ratio_init = opti_ratio_init
        self.opti_ratio_increment = opti_ratio_increment
        self.init_sample_num = init_sample_num
        self.init_search_num = init_search_num
        self.corse_init_center_size = corse_init_center_size
        self.fine_init_center_size = fine_init_center_size
        self.action_norm = action_norm
        self.raw_dataset_load = False
        self.final_cluster_labels = []
        self.clustered_centers = []
        self.args = args
        
    def _load_raw_dataset(self):
        self.raw_dataset = np.load(self.raw_dataset_path, allow_pickle=True).item()    
        self.raw_dataset_load = True
        self.args.total_sample_num = self.raw_dataset['actions'].shape[0]
        print(f"{self.args.total_sample_num} samples in the entire dataset")

    def _terminal_check(self):
        if not self.raw_dataset_load:
            raise RuntimeError('Raw dataset is not loaded')
        else:
            if not "terminals" in self.raw_dataset:
                if not "timeouts" in self.raw_dataset:
                    raise RuntimeError('Invalid dataset')
                else:
                    self.raw_dataset['terminals'] = self.raw_dataset['timeouts']
                    
    def _init_labels(self):
        self.traj_labels = []
        self.total_traj_num = 0
        for t in self.raw_dataset['terminals']:
            if t:
                self.traj_labels.append(-1)
                self.total_traj_num += 1
        self.traj_labels = np.array(self.traj_labels)
        self.current_traj_num = self.total_traj_num
        print(f'{self.total_traj_num} trajectaries in the dataset')
        
    def _remake_raw_dataset(self):
        self._dim_check()
        traj_idx = 0
        mark = []
        inv_mark = []
        traj_mark = []
        temp_true = []
        temp_false = []
        for idx in range(self.norm_actions.shape[0]):
            temp_true.append(True)
            temp_false.append(False)
            if self.raw_dataset['terminals'][idx]:
                if self.traj_labels[traj_idx] == -1:
                    mark += temp_true
                    inv_mark += temp_false
                    traj_mark.append(True)
                else:
                    mark += temp_false
                    inv_mark += temp_false
                    traj_mark.append(False)
                temp_true = []
                temp_false = []
                traj_idx += 1
        self.raw_dataset['observations'] = self.raw_dataset['observations'][mark]
        self.raw_dataset['actions'] = self.raw_dataset['actions'][mark]
        self.raw_dataset['terminals'] = self.raw_dataset['terminals'][mark]
        self.raw_dataset['rewards'] = self.raw_dataset['rewards'][mark]
        self.raw_dataset['labels'] = self.raw_dataset['labels'][mark]
        self.traj_labels = self.traj_labels[traj_mark]
        
    def _remake_traj_labels(self, pu_labels):
        assert self.traj_labels.shape[0] == pu_labels.shape[0], 'Labels dim error'
        for i, value in enumerate(pu_labels):
            if value == 1:
                self.traj_labels[i] = 1
    
    def _dim_check(self):
        self.raw_dataset_load = True
        assert self.norm_actions.shape[0] == self.raw_dataset['terminals'].shape[0] == \
            self.raw_dataset['observations'].shape[0] == self.raw_dataset['actions'].shape[0] == \
                self.raw_dataset['rewards'].shape[0], 'Dimension Error'
        
    def _normlize_dataset(self):
        if self.action_norm == 'l1':
            self.norm_actions = normalize(self.raw_dataset['actions'], norm='l1')
        elif self.action_norm == 'l2':
            self.norm_actions = normalize(self.raw_dataset['actions'], norm='l2')
        else:
            raise RuntimeError('Invalid norm type')

    def _aggregate_trajs(self):
        self._dim_check()
        agg_trajs = []
        temp_action = []
        temp_label = []
        traj_count = 0
        self.count_labels = []
        for idx in range(self.norm_actions.shape[0]):
            temp_label.append(self.raw_dataset['labels'][idx])
            temp_action.append(self.norm_actions[idx])
            if self.raw_dataset['terminals'][idx]:
                temp_action = np.array(temp_action)
                temp_label = np.array(temp_label).mean()
                self.count_labels.append(temp_label)
                agg_traj = np.std(temp_action, axis=0)
                agg_trajs.append(agg_traj)
                temp_action = []
                temp_label = []
                traj_count += 1
        self.current_traj_num = traj_count
        print(f'{traj_count} number trajactaries in the entire dataset')
        return np.array(agg_trajs)
        
    def _get_init_centroid(self, init_search_num =None):
        if init_search_num  == None:
            init_search_num  = self.init_search_num 
        negative_one_indices = [index for index, value in enumerate(self.traj_labels) if value == -1]
        if not negative_one_indices:
            return "No negative-one elements found!"
        sampled_index = random.choice(negative_one_indices)
        return sampled_index
    
    def _get_traj_distance(self, centroid):
        distances = np.sqrt(np.sum((self.norm_actions - centroid)**2, axis=1))
        dis_all = []
        dis_single = []
        for idx in range(distances.shape[0]):
            dis_single.append(distances[idx])
            if self.raw_dataset['terminals'][idx]:
                if self.dis_mean_type == 'mean':
                    mean_dis = np.array(dis_single).mean()
                elif self.dis_mean_type == 'geo_mean':
                    mean_dis = utils.geometric_mean(dis_single, len(dis_single))
                else:
                    raise RuntimeError('Invalid mean type')
                dis_all.append(mean_dis)
                dis_single = []
        return np.array(dis_all)
        
    def _get_min_distance_cluster(self, sample_traj_num=100, min_cluster_size = 6, min_cluster_iters = 5e5, second_min_cluster_size = 4):
        samples =  np.array(random.sample(range(self.current_traj_num), sample_traj_num))
        samples = np.sort(samples)
        ep_count = 0
        temp_action = []
        temp_label = [] 
        labels = []
        centers = []
        for idx in range(self.norm_actions.shape[0]):
            temp_action.append(self.norm_actions[idx])
            temp_label.append(self.raw_dataset['labels'][idx]) 
            if self.raw_dataset['terminals'][idx]:
                if (ep_count == samples).any():
                    temp_action = np.array(temp_action)
                    center = np.std(temp_action, axis=0)
                    centers.append(center)
                    temp_label = np.array(temp_label) 
                    label = np.mean(temp_label, axis=0) 
                    labels.append(label) 
                    temp_label = [] 
                    temp_action = []
                else:
                    temp_label = [] 
                    temp_action = []
                ep_count += 1
        centers = np.array(centers)
        coarse_cluster, _ = find_min_distance_subcluster(centers, min_cluster_size, min_cluster_iters)
        fine_cluster, _ = find_min_distance_cluster_exhaustive(centers[coarse_cluster], second_min_cluster_size)
        fine_cluster_label = [coarse_cluster[i] for i in fine_cluster]
        print(f"The cluster with minimum total distance is {fine_cluster_label}.")
        selected_traj_centers = centers[fine_cluster_label]
        return selected_traj_centers, fine_cluster_label, np.array(labels)[fine_cluster_label]
    
    def _get_neg_set(self, pos_centroid, agg_trajs):
        differences = agg_trajs - pos_centroid
        distances = np.linalg.norm(differences, axis=1)
        max_distance_index = np.argmax(distances)
        neg_centroid = agg_trajs[max_distance_index]
        return neg_centroid
        
    def _get_pu_seed_set(self, overall_labels):
        subset_obs = []
        subset_action = []
        subset_reward = []
        subset_terminal = []
        subset_norm_action = []
        temp_obs = []
        temp_action = []
        temp_reward = []
        temp_terminal = []
        temp_norm_action = []
        data_amount = 0
        episode_num = 0
        for idx in range(self.norm_actions.shape[0]):
            temp_obs.append(self.raw_dataset['observations'][idx])
            temp_action.append(self.raw_dataset['actions'][idx])
            temp_reward.append(self.raw_dataset['rewards'][idx])
            temp_terminal.append(self.raw_dataset['terminals'][idx])
            temp_norm_action.append(self.norm_actions[idx])
            if self.raw_dataset['terminals'][idx]:
                if overall_labels[episode_num] == 1:
                    subset_obs += temp_obs
                    subset_action += temp_action
                    subset_reward += temp_reward
                    subset_terminal += temp_terminal
                    subset_norm_action += temp_norm_action
                    data_amount += 1
                episode_num += 1
                temp_obs = []
                temp_action = []
                temp_reward = []
                temp_terminal = []
                temp_norm_action = []
        print(f'{data_amount} trajectries in the clusted subset')
        subset_obs = np.array(subset_obs)
        subset_action = np.array(subset_action)
        subset_reward = np.array(subset_reward)
        subset_terminal = np.array(subset_terminal)
        subset_norm_action = np.array(subset_norm_action)
        seedset = {}
        seedset['observations'] = subset_obs
        seedset['actions'] = subset_action
        seedset['rewards'] = subset_reward
        seedset['terminals'] = subset_terminal
        seedset['norm_actions'] = subset_norm_action
        return seedset
    
    def run(self):
        cluster_num = 1
        self._load_raw_dataset()
        self._terminal_check()
        self._init_labels()
        self._normlize_dataset()
        while True:
            pu_seed_set_neg = None
            if self.args.use_neg_set:
                if cluster_num > self.args.use_neg_set_iters:
                    self.args.use_neg_set = False
            self._remake_raw_dataset()
            self._normlize_dataset()
            agg_trajs = self._aggregate_trajs()
            if self.args.init_method == 'max-density':
                selected_traj_centers, fine_cluster_label, labels = self._get_min_distance_cluster(sample_traj_num=self.init_sample_num, 
                                                                       min_cluster_size = self.corse_init_center_size, 
                                                                       min_cluster_iters = self.init_search_num, 
                                                                       second_min_cluster_size = self.fine_init_center_size)
                centroid = selected_traj_centers.mean(axis=0)
            elif self.args.init_method == 'random':
                k = random.randint(0, agg_trajs.shape[0]-1)
                centroid = agg_trajs[k]
                labels = fine_cluster_label = self.count_labels[k]
            _opti_ratio = self.opti_ratio_init
            _iter_num = 0
            while True:
                distances = np.sqrt(np.sum((agg_trajs - centroid)**2, axis=1))
                split_point = self.total_traj_num * _opti_ratio
                _opti_ratio += self.opti_ratio_increment
                value_at_percentile = np.sort(distances)[int(split_point-1)]
                temp_trajs_labels = []
                for idx, d in enumerate(distances):
                    if d < value_at_percentile:
                        temp_trajs_labels.append(1)
                    else:
                        temp_trajs_labels.append(0)
                temp_trajs_labels = np.array(temp_trajs_labels)
                count_ones_col = np.sum(temp_trajs_labels == 1)
                if _iter_num >= self.opti_iter_num:
                    break
                assert temp_trajs_labels.shape[0] == agg_trajs.shape[0], 'Error'
                temp_centers = []
                traj_amount = 0
                for idx, lb in enumerate(temp_trajs_labels):
                    if lb == 1:
                        temp_centers.append(agg_trajs[idx])
                        traj_amount += 1
                temp_centers = np.array(temp_centers)
                centroid = np.mean(temp_centers, axis=0)
                _iter_num += 1
            pu_seed_set = self._get_pu_seed_set(temp_trajs_labels)
            if self.args.use_neg_set:
                print('using neg data')
                pos_centroid = pu_seed_set['actions'].mean(axis=0)
                neg_centroid = self._get_neg_set(pos_centroid, agg_trajs)
                distances_neg = np.sqrt(np.sum((agg_trajs - neg_centroid)**2, axis=1))
                _opti_ratio -= self.opti_ratio_increment
                split_point_neg = self.total_traj_num * _opti_ratio
                value_at_percentile_neg = np.sort(distances_neg)[int(split_point_neg-1)]
                temp_trajs_labels_neg = []
                for idx, d_neg in enumerate(distances_neg):
                    if d_neg < value_at_percentile_neg:
                        temp_trajs_labels_neg.append(1)
                    else:
                        temp_trajs_labels_neg.append(0)
                temp_trajs_labels_neg = np.array(temp_trajs_labels_neg)
                assert temp_trajs_labels_neg.shape[0] == agg_trajs.shape[0], 'Error'    
                pu_seed_set_neg = self._get_pu_seed_set(temp_trajs_labels_neg)
            last_cluster, pu_labels = runner.run(pu_seed_set, self.raw_dataset, self.args, pu_seed_set_neg, cluster_num)
            self._remake_traj_labels(np.array(pu_labels))
            self.final_cluster_labels.append(pu_labels)
            if last_cluster or cluster_num > 6:
                print('This is the last cluster, ending')
                break
            else:
                print(f'This is not the last cluster, this is cluster {cluster_num}')
                cluster_num += 1
        self.save_results()
        return self.traj_labels, cluster_num
    
    def save_results(self):
        with open(f'{self.args.save_path}/estimated_traj_labels.pkl', 'wb') as file:
            pickle.dump(self.final_cluster_labels, file)