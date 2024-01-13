#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiangwang
"""
from pu_filter import trainer
from pu_filter import buffer


def check_last_cluster(relabel_num, raw_len, args):
    remain = raw_len - relabel_num
    threshold = args.total_sample_num * args.last_cluster_remain_ratio
    if remain < threshold:
        return True
    else: 
        return False
    
    
def should_end(relabel_num, raw_len, pu_early_stop_sample_amount):
    if abs(raw_len - relabel_num) < pu_early_stop_sample_amount:
        return True
    else:
        return False
    
    
def run(pos_seed_dataset, raw_dataset, args, pu_seed_set_neg, cluster_num):
    buff = buffer.PUBuffer(pos_seed_dataset, raw_dataset, args, pu_seed_set_neg) 
    classifier = trainer.PUTrainer(buff.args, cluster_num) 
    buff.set_seed_positive()
    for t in range(args.iterations):
        probs, pu_traj_labels, log_save_path, relabel_num, raw_len = classifier.train(buff.torch_loader, args.epochs_per_iteration, 
                                                                                      t, buff.classifier_validate, args)
        buff.update_pos() 
        if should_end(relabel_num, raw_len, args.pu_early_stop_sample_amount):
            break
    if_last_cluster = check_last_cluster(relabel_num, raw_len, args)
    return if_last_cluster, pu_traj_labels