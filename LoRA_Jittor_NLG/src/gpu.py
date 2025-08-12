#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import itertools

import numpy as np

import jittor as jt
from jittor import nn


def add_gpu_params(parser: argparse.ArgumentParser):
    parser.add_argument("--platform", default='k8s', type=str, help='platform cloud')
    parser.add_argument("--local_rank", default=0, type=int, help='local rank')
    parser.add_argument("--rank", default=0, type=int, help='rank')
    parser.add_argument("--device", default=0, type=int, help='device')
    parser.add_argument("--world_size", default=0, type=int, help='world size')
    parser.add_argument("--random_seed", default=10, type=int, help='random seed')


def distributed_opt(args, model, opt, grad_acc=1):
    # Jittor handles distributed training differently
    # For now, we'll return the model and optimizer as-is
    # In Jittor, distributed training is handled through mpi
    return model, opt


def distributed_gather(args, tensor):
    # Jittor's distributed operations
    # For now, return the tensor as-is since Jittor handles this internally
    return tensor


def distributed_sync(args):
    # Jittor handles synchronization internally
    # For now, this is a no-op
    pass


def parse_gpu(args):
    jt.set_global_seed(args.random_seed)
    
    # Jittor automatically handles GPU selection and distributed training
    # Set the device if CUDA is available
    if jt.has_cuda:
        jt.flags.use_cuda = 1
        print(f'Using CUDA device')
    else:
        print('CUDA not available, using CPU')
    
    # For distributed training in Jittor, use mpi
    if args.world_size > 1:
        # Initialize MPI for distributed training
        try:
            import jittor.mpi as mpi
            args.rank = mpi.world_rank()
            args.world_size = mpi.world_size()
            args.local_rank = args.rank % jt.get_device_count() if jt.has_cuda else 0
            print(f'MPI initialized: rank={args.rank}, world_size={args.world_size}')
        except:
            print('MPI not available, using single process')
            args.rank = 0
            args.world_size = 1
            args.local_rank = 0
    else:
        args.rank = 0
        args.world_size = 1
        args.local_rank = 0
    
    print(
        'myrank:', args.rank, 
        'local_rank:', args.local_rank, 
        'device_count:', jt.get_device_count() if jt.has_cuda else 1, 
        'world_size:', args.world_size
    )
    
    
def cleanup(args):
    # Jittor handles cleanup automatically
    pass
