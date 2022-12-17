#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

# eval
from scipy import stats

from .processor import Processor

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv1d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
        
def compute_score(model_type, probs, diff):
    if model_type == 'single':
        pred = probs.argmax(dim=-1) * (114.8 / (101 - 1))
    else:
        # calculate expectation & denormalize & sort
        judge_scores_pred = torch.stack([prob.argmax(dim=-1) * 10. / (21 - 1)
                                         for prob in probs], dim=1).sort()[0]  # N, 7
        # keep the median 3 scores to get final score according to the rule of diving
        pred = torch.sum(judge_scores_pred, dim=1) * diff.cuda()
    return pred

def compute_loss(model_type, criterion, probs, data):
    
    if model_type == 'single':
        loss = criterion(torch.log(probs), data)
    else:
        loss = sum([criterion(torch.log(probs[i]), data[:, i].cuda()) for i in range(3)])
    return loss

KLD = nn.KLDivLoss(reduction='batchmean')
class REC_Processor(Processor):
    """
        Processor for Skeleton-based Action Recgnition
    """

    def load_model(self):
        self.model = self.io.load_model(self.arg.model,
                                        **(self.arg.model_args))
        self.model.apply(weights_init)
        self.loss = nn.KLDivLoss(reduction='batchmean')
        self.model_type = 'single'
        # self.loss = nn.MSELoss()
        # self.loss = nn.HuberLoss(delta=0.5)
        # self.loss = nn.SmoothL1Loss()
        # self.loss = nn.L1Loss()
        
    def load_optimizer(self):
        if self.arg.optimizer == 'SGD':
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.85,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay)
        elif self.arg.optimizer == 'Adam':
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay)
        else:
            raise ValueError()

    def adjust_lr(self):
        if self.arg.optimizer == 'SGD' and self.arg.step:
            lr = self.arg.base_lr * (
                0.1**np.sum(self.meta_info['epoch']>= np.array(self.arg.step)))
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k):
        rank = self.result.argsort()
        hit_top_k = [l in rank[i, -k:] for i, l in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.print_log('\tTop{}: {:.2f}%'.format(k, 100 * accuracy))

    def train(self):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        spearmanr_value = []
        l2_value = []
        relative_l2_value = []

        for group in loader:
            if len(group) == 4 :
                data, label, soft_label, diff = group['skeleton'],group['final_score'],group['soft_label'],group['difficulty']
                
            else:
                data, label, soft_label, diff, rgb_data = group['skeleton'],group['final_score'],group['soft_label'],group['difficulty'],group['video_feat']
                rgb_data = rgb_data.float().to(self.dev)
            # get data
            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            soft_label = soft_label.float().to(self.dev)
            diff = diff.float().to(self.dev)

            # forward
            if len(group) == 4:
                probs = self.model(data)
            else:
                probs = self.model(data, rgb_data)
            
            preds = compute_score(self.model_type,probs,diff)
            loss = compute_loss(self.model_type, self.loss, probs, soft_label)
            
            # backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10, norm_type=2)
            self.optimizer.step()

            # statistics
            self.iter_info['loss'] = loss.data.item()
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            loss_value.append(self.iter_info['loss'])

            preds = preds.squeeze().cpu().detach().numpy()
            label = label.cpu().detach().numpy()
          
            rho, p = stats.spearmanr(preds , label )
            spearmanr_value.append([rho,p])
                
            L2 = np.power(preds - label, 2).sum() / label.shape[0]
            RL2 = np.power((preds - label) / (label.max() - label.min()), 2).sum() / \
                    label.shape[0]

            l2_value.append(L2)
            relative_l2_value.append(RL2)

            self.show_iter_info()
            self.meta_info['iter'] += 1
            
            torch.cuda.empty_cache()

        self.epoch_info['mean_loss']= np.mean(loss_value)
        self.epoch_info['mean_spearmanr_rho_loss'],self.epoch_info['mean_spearmanr_p_loss'] = np.mean(spearmanr_value,axis = 0)
        self.epoch_info['mean_L2_loss']= np.mean(l2_value)
        self.epoch_info['mean_RL2_loss']= np.mean(relative_l2_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation=True):

        self.model.eval()
        loader = self.data_loader['test']
        loss_value = []
        spearmanr_value = []
        relative_l2_value = []
        l2_value = []
        result_frag = []
        label_frag = []

        for group in loader:
            if len(group) == 4:
                data, label, soft_label, diff = group['skeleton'],group['final_score'],group['soft_label'],group['difficulty']

            else:
                data, label, soft_label, diff, rgb_data = group['skeleton'],group['final_score'],group['soft_label'],group['difficulty'],group['video_feat']
                rgb_data = rgb_data.float().to(self.dev)
            # get data
            data = data.float().to(self.dev)
            label = label.float().to(self.dev)
            soft_label = soft_label.float().to(self.dev)
            diff = label.float().to(self.dev)

            # inference
            with torch.no_grad():
                if len(group) == 4:
                    probs = self.model(data)
                else:
                    probs = self.model(data, rgb_data)
            
            preds = compute_score(self.model_type,probs,diff)
            loss = compute_loss(self.model_type, self.loss, probs, soft_label)
            result_frag.append(preds.data.cpu().numpy())

            # get loss
            if evaluation:
                loss = compute_loss(self.model_type, self.loss, probs, soft_label)
                loss_value.append(loss.item())

                preds = preds.squeeze().cpu().detach().numpy()
                label = label.cpu().detach().numpy() 

                rho, p = stats.spearmanr(preds , label )
                spearmanr_value.append([rho,p])
                
                L2 = np.power(preds - label, 2).sum() / label.shape[0]
                RL2 = np.power((preds - label) / (label.max() - label.min()), 2).sum() / \
                    label.shape[0]

                l2_value.append(L2)
                relative_l2_value.append(RL2)
                label_frag.append(label)

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info['mean_loss']= np.mean(loss_value)
            self.epoch_info['mean_spearmanr_rho_loss'],self.epoch_info['mean_spearmanr_p_loss'] = np.mean(spearmanr_value,axis = 0)
            self.epoch_info['mean_L2_loss']= np.mean(l2_value)
            self.epoch_info['mean_RL2_loss']= np.mean(relative_l2_value)
            self.show_epoch_info()

            # show top-k accuracy
            # for k in self.arg.show_topk:
            #     self.show_topk(k)

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        # evaluation
        parser.add_argument('--show_topk', type=int, default=[1, 5], nargs='+', help='which Top K accuracy will be shown')
        # optim
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        # endregion yapf: enable

        return parser
