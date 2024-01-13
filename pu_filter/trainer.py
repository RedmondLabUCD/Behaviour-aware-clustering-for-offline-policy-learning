#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: qiang
"""
import os
import gc
import torch
import random
import seaborn
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from pu_filter import models
from pu_filter import pu_utils as utils
from pu_filter.pu_utils import matrix, conf_matrix


def set_seed(seed):
    utils.set_seed(seed)
    models.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class PUTrainer():
    def __init__(self, args, cluster_num):
        set_seed(args.seeds)
        self.args = args
        self.device = utils.device_handler(args)
        self.criterion = torch.nn.BCELoss().to(self.device)
        self.train_logger = utils.TrainLogger(
            experiment_name=f'{args.exp_name}-cluster{cluster_num}',
            save_metrics=True,
            root_dir=args.save_path,
            verbose=False,
            tensorboard_dir=1,
            with_timestamp=True,
        )
        self.mat = matrix()
        self.conf_mat = conf_matrix()
        self.turn = 0
        self.turn_step = 0
        self.total_step = 0
        self.model = None
    
    # Reset the neural network and its corresponding optimizer after each iteration.
    def reset(self):
        if self.model != None:
            del self.model
            gc.collect()
        self.model = []
        for i in range(self.args.models_in_ensemble):
            self.model.append(models.FilterNet(self.args,
                                              bias=True
                                              ).to(self.device))
        self.optimizer = []
        for j in self.model:
            self.optimizer.append(torch.optim.Adam(j.parameters(),
                                              lr=self.args.learning_rate,
                                              weight_decay=3e-3,
                                              betas=(0.9, 0.999),
                                              eps=1e-8,
                                              amsgrad=False
                                              ))

    def train(self, loaders, epochs, turn, validate_func, args):
        self.reset()
        for idx,loader in enumerate(loaders):
            self.model[idx].train()
            for epoch in range(1, epochs + 1):
                epoch_loss = defaultdict(list)
                self.turn_step += 1
                
                for [observations, actions, labels] in loader:
                    observations = torch.tensor(observations).to(
                        torch.float32).to(self.device)
                    actions = torch.tensor(actions).to(
                        torch.float32).to(self.device)
                    labels = torch.tensor(labels).to(
                        torch.float32).to(self.device)
                    
                    pred = self.model[idx](observations, actions)
                    loss = self.criterion(pred, labels)
    
                    self.optimizer[idx].zero_grad()
                    loss.backward()
                    self.optimizer[idx].step()
                    
                    self.train_logger.add_metric(
                        f'loss-model={idx+1}', loss.cpu().detach().numpy())
                    epoch_loss['loss'].append(loss.cpu().detach().numpy())
                    self.total_step += 1
                    self.train_logger.commit(self.turn_step, self.total_step)
                
                print(f"Iteration: {turn+1}   Model: {idx}   Epoch: {epoch}   Loss: {np.array(epoch_loss['loss']).mean()}")
        
        # This step is important. It utilizes the classifier to filter the dataset to get the labels.
        amount, count, probs, pu_traj_labels, relabel_num, raw_len = validate_func(self.model,
                                                        f'{self.args.save_path}/{self.train_logger._experiment_name}/adap_probs-iteration={turn+1}.jpg')
        
        # Save models and logs
        seaborn.displot( 
          data=np.array(probs), 
          kind="hist", 
          aspect=1.4,
          bins=args.th_bins,
        )
        np.save(f'{self.args.save_path}/{self.train_logger._experiment_name}/probs-iteration={turn+1}.npy', np.array(probs))
        plt.plot()
        plt.tight_layout()
        plt.xlabel("Summed probs")
        plt.ylabel("Amount")
        plt.savefig(f'{self.args.save_path}/{self.train_logger._experiment_name}/hist-iteration={turn+1}.jpg',
                    dpi=400,bbox_inches='tight')
        
        txt_path = f'{self.args.save_path}/{self.train_logger._experiment_name}/relabel_num-iteration={turn+1}.txt'
        with open(txt_path, 'w') as file:
            file.write(f"{relabel_num} / {raw_len}")
        
        self.save(
            self.turn+1, path=f'{self.args.save_path}/{self.train_logger._experiment_name}')
        self.turn += 1
        
        return probs, pu_traj_labels, f'{self.args.save_path}/{self.train_logger._experiment_name}', relabel_num, raw_len

    def save(self, epoch, path=None):
        for idx, model in enumerate(self.model):
            torch.save(model.state_dict(),
                       f'{path}/ckpt-model={idx+1}-iteration={epoch}.pth')

    # Load the model from the path provided in the 'args' and utilize the classifier to filter the dataset in the 'validate_func'.
    def load(self, validate_func=None):
        self.reset()
        for idx,model in enumerate(self.model):
            model.load_state_dict(torch.load(f"{self.args.trained_filter_path}/ckpt-model={idx+1}-iteration={self.args.ckpt_iterations}.pth", map_location=self.device))
        print('Model loaded!')
        print('Using the trained model to filter the mixed dataset.')
        if validate_func:
            amount, acc, count, probs = validate_func(self.model,
                                                            f'{self.args.save_path}/{self.train_logger._experiment_name}/adap_probs_{self.args.ckpt_iterations}.jpg')
            seaborn.displot( 
              data=np.array(probs), 
              kind="hist", 
              aspect=1.4,
              bins=100,
            )
            np.save(f'{self.args.save_path}/{self.train_logger._experiment_name}/prob_{self.args.ckpt_iterations}.npy', np.array(probs))
            plt.plot()
            plt.tight_layout()
            plt.xlabel("Summed probs")
            plt.ylabel("Amount")
            plt.savefig(f'{self.args.save_path}/{self.train_logger._experiment_name}/dist_{self.args.ckpt_iterations}.jpg',
                        dpi=400,bbox_inches='tight')
