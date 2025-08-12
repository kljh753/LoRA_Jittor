#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import functools
import os, shutil
import numpy as np
import random

import jittor as jt


def logging(s, log_path, print_=True, log_=True):
    if print_:
        print(s)
    if log_:
        with open(log_path, 'a+') as f_log:
            f_log.write(s + '\n')


def get_logger(log_path, **kwargs):
    return functools.partial(logging, log_path=log_path, **kwargs)


def create_exp_dir(dir_path, scripts_to_save=None, debug=False):
    if debug:
        print('Debug Mode : no experiment dir created')
        return functools.partial(logging, log_path=None, log_=False)

    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    print('Experiment dir : {}'.format(dir_path))
    if scripts_to_save is not None:
        script_path = os.path.join(dir_path, 'scripts')
        if not os.path.exists(script_path):
            os.makedirs(script_path)
        for script in scripts_to_save:
            dst_file = os.path.join(dir_path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)

    return get_logger(log_path=os.path.join(dir_path, 'log.txt'))


def save_checkpoint(model, optimizer, path, epoch):
    jt.save(model.state_dict(), os.path.join(path, 'model_{}.pkl'.format(epoch)))
    jt.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pkl'.format(epoch)))
    jt.save(model, os.path.join(path, 'model_{}.pt'.format(epoch)))
    jt.save(optimizer.state_dict(), os.path.join(path, 'optimizer_{}.pt'.format(epoch)))


def set_random_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    jt.set_global_seed(seed)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def print_args(args):
    """Print arguments"""
    if hasattr(args, 'rank') and args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)
    elif not hasattr(args, 'rank'):
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


# Add missing functions for compatibility
def distributed_gather(*args, **kwargs):
    """Placeholder for distributed gather function"""
    pass

def distributed_sync(*args, **kwargs):
    """Placeholder for distributed sync function"""
    pass

def cleanup(*args, **kwargs):
    """Placeholder for cleanup function"""
    pass

def parse_gpu(args):
    """Placeholder for parse_gpu function - should be imported from gpu.py"""
    pass
