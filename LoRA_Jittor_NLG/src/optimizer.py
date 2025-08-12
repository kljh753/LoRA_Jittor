# ------------------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
# ------------------------------------------------------------------------------------------
import logging
import math
import os
from collections import OrderedDict
import argparse

import jittor as jt
from jittor import nn
from jittor.optim import Optimizer
# 修改第14行，移除不存在的LambdaLR导入
# from jittor.lr_scheduler import LambdaLR, _LRScheduler
# 移除所有 jittor.lr_scheduler 导入
# from jittor.lr_scheduler import _LRScheduler

# 自定义基础调度器类
class _LRScheduler:
    def __init__(self, optimizer, last_epoch=-1):
        self.optimizer = optimizer
        if last_epoch == -1:
            for group in optimizer.param_groups:
                group.setdefault('initial_lr', group['lr'])
        else:
            for i, group in enumerate(optimizer.param_groups):
                if 'initial_lr' not in group:
                    raise KeyError("param 'initial_lr' is not specified "
                                   "in param_groups[{}] when resuming an optimizer".format(i))
        self.base_lrs = list(map(lambda group: group['initial_lr'], optimizer.param_groups))
        self.last_epoch = last_epoch

    def get_lr(self):
        raise NotImplementedError

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

# 自定义 LambdaLR 类
class LambdaLR(_LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer
        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        super(LambdaLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [base_lr * lmbda(self.last_epoch)
                for lmbda, base_lr in zip(self.lr_lambdas, self.base_lrs)]


def add_optimizer_params(parser: argparse.ArgumentParser):
    parser.add_argument('--lr', default=0.00001, type=float, help='learning rate')
    parser.add_argument('--weight_decay', default=0.01, type=float, help='weight decay rate')
    parser.add_argument('--correct_bias', action='store_true', help='correct adam bias term')
    parser.add_argument('--adam_epislon', default=1e-6, type=float, help='adam epsilon')
    parser.add_argument('--no_decay_bias', action='store_true', help='no weight decay on bias weigh')
    parser.add_argument('--adam_beta1', default=0.9, type=float, help='adam beta1 term')
    parser.add_argument('--adam_beta2', default=0.98, type=float, help='adam beta2 term')
    
    parser.add_argument('--scheduler', default='linear', type=str,
                        choices=['cosine', 'inv_sqrt', 'dev_perf', 'constant', 'linear', 'cycle'],
                        help='lr scheduler to use.')

    parser.add_argument('--max_step', type=int, default=None, help='upper epoch limit')

    parser.add_argument('--max_epoch', type=int, default=None, help='max epoch of training')

    parser.add_argument('--warmup_step', type=int, default=0, help='upper epoch limit')

    parser.add_argument('--i_steps', type=str, default='0', help='interval_steps')
    parser.add_argument('--i_lrs', type=str, default='0.00025', help='interval_lrs')


class CosineAnnealingWarmupRestarts(_LRScheduler):
    def __init__(
        self,
        optimizer : jt.optim.Optimizer,
        max_lr : float = 0.1,
        min_lr : float = 0.0,
        warmup_steps : int = 0,
        max_steps : int = 1,
        alpha : float = 0.,
        last_epoch : int = -1
    ):
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        
        self.alpha = alpha
        self.max_steps = max_steps
        super(CosineAnnealingWarmupRestarts, self).__init__(optimizer, last_epoch)
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.min_lr
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            curr_lr = self.max_lr * self.last_epoch / self.warmup_steps
            return curr_lr
        else:
            _step = min(self.last_epoch, self.max_steps)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * _step / self.max_steps))
            decayed = (1 - self.alpha) * cosine_decay + self.alpha
            return self.max_lr * decayed

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups: 
            param_group['lr'] = _lr


class CyclicScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        interval_steps = [],
        interval_lrs = [],
        last_epoch = -1,
    ):
        self.optimizer = optimizer

        self.interval_steps = interval_steps
        self.interval_lrs = interval_lrs

        self.last_epoch = last_epoch

        super(CyclicScheduler, self).__init__(optimizer, last_epoch)
        
        self.init_lr()
    
    def init_lr(self):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.interval_lrs[0]
    
    def get_lr(self):
        for _i in range(0, len(self.interval_steps)-1):
            if self.last_epoch >= self.interval_steps[_i] and self.last_epoch < self.interval_steps[_i + 1]:
                _alpha = (self.last_epoch - self.interval_steps[_i]) / (self.interval_steps[_i + 1] - self.interval_steps[_i] + 1e-6)
                if _alpha < 0:
                    _alpha = 0
                if _alpha >= 1:
                    _alpha = 1
                curr_lr = _alpha * self.interval_lrs[_i + 1] + (1.0 - _alpha) * self.interval_lrs[_i]
                return curr_lr
        return self.interval_lrs[-1]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = math.floor(epoch)
        _lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = _lr



def get_linear_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def get_constant_schedule_with_warmup(
    optimizer, 
    num_warmup_steps, 
    num_training_steps, 
    last_epoch=-1
):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return 1.0
    return LambdaLR(optimizer, lr_lambda, last_epoch)


def create_grouped_parameters(model, no_decay_bias):
    if not no_decay_bias:
        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters()],
        }]
    else:
        no_decay = ["bias", "layer_norm.weight"]

        optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 
            "weight_decay": 0.0,
        }]
    return optimizer_grouped_parameters


def create_adam_optimizer(
    model, 
    lr, 
    weight_decay, 
    optimizer_grouped_parameters=None, 
    beta1=0.9, 
    beta2=0.98, 
    correct_bias=True, 
    adam_epislon=1e-6, 
    no_decay_bias=False
):
    if optimizer_grouped_parameters is None:
        optimizer_grouped_parameters = create_grouped_parameters(model, no_decay_bias)


    optimizer = jt.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=lr, 
        betas=(beta1, beta2), 
        eps=adam_epislon, 
        weight_decay=weight_decay
    )
    return optimizer


def create_sgd_optimizer(model, lr):
    optimizer = jt.optim.SGD(model.parameters(), lr=lr, momentum=0.0)
    return optimizer
    

def create_adam_optimizer_from_args(model, args, grouped_parameters=None):
    if grouped_parameters is None:
        grouped_parameters = create_grouped_parameters(model, args.no_decay_bias)

    for group in grouped_parameters:
        if 'lr' not in group:
            group['lr'] = args.lr
   
    optimizer = jt.optim.AdamW(
        grouped_parameters,
        lr=args.lr, 
        betas=(args.adam_beta1, args.adam_beta2),
        eps=args.adam_epislon,
        weight_decay=args.weight_decay,
    )
    return optimizer

def create_optimizer_scheduler(optimizer, args):
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer, 
            max_lr=args.lr, 
            min_lr=0.0, 
            warmup_steps=args.warmup_step, 
            max_steps=args.max_step, alpha=0
        )
    elif args.scheduler == 'linear':
        scheduler = get_linear_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    elif args.scheduler == 'cycle':
        if args.i_steps is not None:
            args.i_steps = [int(_i) for _i in args.i_steps.split(',')]
            args.i_lrs = [float(_i) for _i in args.i_lrs.split(',')]
        args.max_step = args.i_steps[-1]
        print('max_step is rest to', args.max_step)
        scheduler = CyclicScheduler(
            optimizer, interval_steps=args.i_steps, interval_lrs=args.i_lrs
        )
    elif args.scheduler == 'constant':
        scheduler = get_constant_schedule_with_warmup(
            optimizer, args.warmup_step, args.max_step, last_epoch=-1
        )
    else:
        scheduler = None
    return scheduler