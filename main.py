#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiangwang
"""
import utils
import argparse
import warnings
from clu_iterator import RunClustering
warnings.filterwarnings("ignore")


def main_PU_cluster(args):
    args = utils.directory_handler(args)
    utils.set_seed(args.seeds)
    clustering_runner = RunClustering(raw_dataset_path = args.raw_dataset_path,
                           opti_ratio_init=args.opti_ratio_init,
                           opti_ratio_increment=args.opti_ratio_increment,
                           opti_iter_num = args.opti_iters_num,
                           action_norm = args.action_norm,
                           init_sample_num = args.init_sample_num,
                           init_search_num = args.init_search_num,
                           corse_init_center_size = args.corse_init_center_size,
                           fine_init_center_size = args.fine_init_center_size,
                           args = args)
    utils.save_params(args)
    traj_labels, cluster_num = clustering_runner.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, default="trifinger-lift", help='Experiment name')
    parser.add_argument('--raw-dataset-path', type=str, default="<your local path>", help='Path to the raw multi-behaviour dataset')
    parser.add_argument('--seeds', type=int, default=4, help='random seed')
    parser.add_argument('--use-gpu', type=bool, default=True, help='Use GPU for accelerating or not')
    parser.add_argument('--save-path', type=str, default=None, help="Path to save the output results. If it is not specified, one folder named 'save' will be created under the root project path for saving the results")
    parser.add_argument('--action-norm', type=str, default='l1', help="l1 or l2")
    parser.add_argument('--dis-mean-type', type=str, default='mean', help="mean or geo_mean")
    parser.add_argument('--opti-ratio-init', type=int, default=0.02, help="The ratio of the initial seed dataset's size to the size of the entire dataset")
    parser.add_argument('--opti-ratio-increment', type=int, default=0.02, help="The increment ratio per epoch used to optimize the seed dataset relative to the initial seed dataset.")
    parser.add_argument('--opti-iters-num', type=int, default=1, help="The number of epochs for optimizing the seed dataset in relation to the initial seed dataset")
    parser.add_argument('--init-method', type=str, default='max-density',  help="random or max-density")
    parser.add_argument('--init-sample-num', type=int, default=100, help='Pamameter for Monte-Carlo search')
    parser.add_argument('--init-search-num', type=int, default=200000, help='Pamameter for Monte-Carlo search')
    parser.add_argument('--corse-init-center-size', type=int, default=6, help='Pamameter for Monte-Carlo search')
    parser.add_argument('--fine-init-center-size', type=int, default=4, help='Pamameter for Monte-Carlo search')
    parser.add_argument('--last-cluster-remain-ratio', type=int, default=5e-2, help="The threshold used to determine if the current cluster is the last one")
    parser.add_argument('--models-in-ensemble', type=int, default=3, help='Number of unit models in the ensemble model')
    parser.add_argument('--ensemble-method', type=str, default='vote',  help="'avg' or 'vote'")
    parser.add_argument('--iterations', type=int, default=5, help='Iteration number for each classifier training trail')
    parser.add_argument('--pu-early-stop-sample-amount', type=int, default=5000, help="If the remaining number of samples is lower than this value, the iteration will be terminated")
    parser.add_argument('--negative-sampler', type=str, default='part', help="part or full")
    parser.add_argument('--epochs-per-iteration', type=int, default=10, help='Number of epoch lasts per classifier training iteration')
    parser.add_argument('--th-bins', type=int, default=100, help='Probability bins for adaptive threshold')
    parser.add_argument('--th-high-bound', type=float, default=0.98, help='Highest possible adaptive threshold')
    parser.add_argument('--th-fit-pow', type=int, default=8, help='Polynomial order for adaptive threshold')
    parser.add_argument('--batch-size', type=int, default=1024, help='Classifier training batch size')
    parser.add_argument('--learning-rate', type=float, default=0.001, help='Classifier training learning rate')
    parser.add_argument('--fix-init-th-conf', type=bool, default=False, help="Whether to use a fixed threshold during the initial iterations for pu-filtering")
    parser.add_argument('--fix-init-th-iters', type=int, default=2, help="The number of iterations using a fixed threshold during the initial iterations for pu-filtering")
    parser.add_argument('--fix-init-th-conf-value', type=float, default=0.9, help="Initial fixed threshold value")
    parser.add_argument('--use-neg-set',  type=bool, default=False, help="Whether to use the negative dataset")
    parser.add_argument('--use-neg-set-iters',  type=int, default=1, help="Iteration number for using the negative dataset")
    args = parser.parse_args()
    main_PU_cluster(args)