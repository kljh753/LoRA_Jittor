#!/usr/bin/env python3
# -*- coding: utf-8 -*-
#  ------------------------------------------------------------------------------------------
#  Copyright (c) Microsoft Corporation. All rights reserved.
#  Licensed under the MIT License (MIT). See LICENSE in the repo root for license information.
#  ------------------------------------------------------------------------------------------
import argparse
import time
import math
import os, sys
import numpy as np
import itertools

import jittor as jt
import random
from jittor.dataset import DataLoader

from gpu import (
    add_gpu_params, 
    parse_gpu, 
    distributed_opt, 
    distributed_gather, 
    distributed_sync, 
    cleanup
)
from optimizer import (
    create_adam_optimizer, 
    create_optimizer_scheduler, 
    add_optimizer_params, 
    create_adam_optimizer_from_args
)

from data_utils import FT_Dataset
from model import GPT2Config, GPT2LMModel
from exp_utils import create_exp_dir

import loralib as lora

parser = argparse.ArgumentParser(description='Jittor GPT2 ft script')

add_gpu_params(parser)
add_optimizer_params(parser)

parser.add_argument('--train_data', required=True, help='location of training data corpus')

parser.add_argument('--valid_data', required=True, help='location of validation data corpus')

parser.add_argument('--train_batch_size', type=int, default=8, help='training batch size')

parser.add_argument('--valid_batch_size', type=int, default=4, help='validation batch size')

parser.add_argument('--grad_acc', type=int, default=1, help='gradient accumulation steps')

parser.add_argument('--clip', type=float, default=0.0, help='gradient clip')

parser.add_argument('--seq_len', type=int, default=512, help='number of tokens to predict.')

parser.add_argument('--model_card', default='gpt2.md', choices=['gpt2.sm', 'gpt2.md', 'gpt2.lg'], 
                    help='model names')

parser.add_argument('--init_checkpoint', default=None, help='pretrained checkpoint path')

parser.add_argument('--fp16', action='store_true', help='train model with fp16')

parser.add_argument('--log_interval', type=int, default=100, help='log interval')

parser.add_argument('--eval_interval', type=int, default=2000, help='eval interval')

parser.add_argument('--save_interval', type=int, default=500, help='save interval')

parser.add_argument('--work_dir', type=str, default=os.getenv('PT_OUTPUT_DIR', 'gpt2_model'), 
                    help='working folder.')

parser.add_argument('--lora_dim', type=int, default=0, help='lora attn dimension')

parser.add_argument('--lora_alpha', type=int, default=128, help='lora attn alpha')

parser.add_argument('--obj', default='clm', choices=['jlm', 'clm'], 
                    help='language model training objective')

parser.add_argument('--lora_dropout', default=0.0, type=float, 
                    help='dropout probability for lora layers')

parser.add_argument('--label_smooth', default=0.0, type=float, help='label smoothing')

parser.add_argument('--roll_interval', type=int, default=-1, help='rolling interval')

parser.add_argument('--roll_lr', type=float, default=0.00001, help='rolling learning rate')

parser.add_argument('--roll_step', type=int, default=100, help='rolling step')

parser.add_argument('--eval_epoch', type=int, default=1, help='eval per number of epochs')
# Remove duplicate --max_epoch parameter (already defined in optimizer.py)
# Remove duplicate --max_step parameter (already defined in optimizer.py)
# Remove duplicate --random_seed parameter (already defined in gpu.py)
# parser.add_argument('--random_seed', type=int, default=110, help='random seed')  # ×¢ÊÍµôÖØ¸´¶¨Òå


def print_args(args):
    if args.rank == 0:
        print('=' * 100)
        for k, v in args.__dict__.items():
            print(f'        - {k} : {v}')
        print('=' * 100)


class AverageMeter(object):
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


def evaluate(model, valid_loader, args):
    model.eval()
    avg_lm_loss = AverageMeter()
    
    with jt.no_grad():
        start_time = time.time()
        for idx, data in enumerate(valid_loader):
       
            def eval_step():
                data_dict = {key: value for key, value in data.items()}
                
                _input = data_dict['input']
                _target = data_dict['target']
                _msk = data_dict['mask']

                _lm_logits, _loss = model(_input, lm_labels=_target, lm_mask=_msk) 
                loss = _loss.mean()
                
                loss_value = loss.item()
                
      
                del _input, _target, _msk, _lm_logits, _loss
                
                return loss_value
            

            loss_value = eval_step()
            del data
            
            avg_lm_loss.update(loss_value)
            
   
            jt.sync_all()
            jt.gc()

            if idx % 100 == 0:
                print('eval samples:', idx, 'loss:', loss_value)

        total_time = time.time() - start_time
        print('average loss', avg_lm_loss.avg)
    
    return avg_lm_loss.avg, math.exp(avg_lm_loss.avg)


def train_validate(
    model, 
    optimizer, 
    scheduler, 
    train_loader, 
    valid_loader, 
    args, 
    train_step=0, 
    epoch=0
):
    model.train()
    avg_lm_loss = AverageMeter()
    print('start to train the model................', epoch)
    log_start_time = time.time()
    best_val_ppl = None

    for idx, data in enumerate(train_loader):
      
        def forward_step():
            data_dict = {key: value for key, value in data.items()}
            
            _input = data_dict['input']
            _target = data_dict['target'] 
            _msk = data_dict['mask']

            _lm_logits, _lm_loss = model(
                _input, lm_labels=_target, lm_mask=_msk, label_smooth=args.label_smooth
            )
            
            _lm_loss = _lm_loss.mean() 
            _lm_loss = _lm_loss / args.grad_acc
            
           
            loss_value = _lm_loss.item() * args.grad_acc
            
            
            del _input, _target, _msk, _lm_logits
            
            return _lm_loss, loss_value
        

        _lm_loss, loss_value = forward_step()
        
        del data
        
        avg_lm_loss.update(loss_value)
        optimizer.step(_lm_loss)
        

        del _lm_loss
        jt.sync_all()
        jt.gc()
        
        if (idx + 1) % args.grad_acc == 0:
            train_step += 1

            jt.sync_all() 
            jt.gc()   
            if scheduler is not None:
                scheduler.step()

        if train_step % args.log_interval == 0 and (idx + 1) % args.grad_acc == 0:
            elapsed = time.time() - log_start_time
            lr = optimizer.param_groups[0]['lr']
            log_str = f'| epoch {epoch:3d} step {train_step:>8d} | {idx + 1:>6d} batches | lr {lr:.3g} | ms/batch {elapsed * 1000 / args.log_interval:5.2f} | loss {avg_lm_loss.avg:.2f} | ppl {math.exp(avg_lm_loss.avg):>4.2f}'
            print(log_str)
            log_start_time = time.time()
            avg_lm_loss.reset()

            jt.sync_all()
            jt.gc()

        if train_step % args.eval_interval == 0 and (idx + 1) % args.grad_acc == 0:

            jt.sync_all()
            jt.gc()
            
            eval_start_time = time.time()
            valid_loss, valid_ppl = evaluate(model, valid_loader, args)
            print('-' * 100)
            print(f'| Eval {train_step // args.eval_interval:3d} at step {train_step:>8d} | time: {time.time() - eval_start_time:5.2f}s | valid loss {valid_loss:.2f} | valid ppl {valid_ppl:>4.2f}')
            print('-' * 100)

            if best_val_ppl is None or valid_ppl < best_val_ppl:
                best_val_ppl = valid_ppl

            model.train()

            jt.sync_all()
            jt.gc()

        if train_step % args.save_interval == 0 and (idx + 1) % args.grad_acc == 0:

            jt.sync_all()
            jt.gc()
            
            model_path = os.path.join(args.work_dir, f'model.{train_step}.pkl')
            print('saving checkpoint', model_path)
            jt.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_step': train_step,
                'args': args
            }, model_path)
            

            jt.sync_all()
            jt.gc()

        if args.max_step is not None and train_step >= args.max_step:
            break

    return train_step


def main():
    args = parser.parse_args()
    parse_gpu(args)
    print_args(args)

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    jt.set_global_seed(args.random_seed)
    
    jt.flags.use_cuda = 1 if jt.has_cuda else 0
    if args.rank == 0:
        args.logging = create_exp_dir(args.work_dir)

    # Set the random seed manually for reproducibility.
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    jt.set_global_seed(args.random_seed)

    # Load Data
    train_data = FT_Dataset(
        args.train_data, args.train_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    )
    
    valid_data = FT_Dataset(
        args.valid_data, args.valid_batch_size, args.seq_len, 
        joint_lm=args.obj=='jlm'
    ) 

    train_loader = DataLoader(
        train_data, batch_size=args.train_batch_size
    )
    
    valid_loader = DataLoader(
        valid_data, batch_size=args.valid_batch_size
    )

    if args.max_step is None:
        args.max_step = (args.max_epoch * train_data.num_batches + args.world_size - 1) // args.world_size
        print('set max_step:', args.max_step)

    # Load Model
    if args.model_card == 'gpt2.sm':
        config = GPT2Config(
            vocab_size_or_config_json_file=50257, 
            n_ctx=1024,          # Keep consistent with pretrained model
            n_positions=1024,    # Keep consistent with pretrained model
            n_embd=768, n_layer=12, n_head=12, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.md':
        config = GPT2Config(
            vocab_size_or_config_json_file=50257, 
            n_ctx=1024,          # Keep consistent with pretrained model
            n_positions=1024,    # Keep consistent with pretrained model
            n_embd=1024, n_layer=24, n_head=16, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )
    elif args.model_card == 'gpt2.lg':
        config = GPT2Config(
            vocab_size_or_config_json_file=50257, 
            n_ctx=1024,          # Keep consistent with pretrained model
            n_positions=1024,    # Keep consistent with pretrained model
            n_embd=1280, n_layer=36, n_head=20, 
            lora_attn_dim=args.lora_dim, lora_attn_alpha=args.lora_alpha, 
            lora_dropout=args.lora_dropout,
        )

    lm_net = GPT2LMModel(config)
    if args.init_checkpoint is not None:
        print('loading model pretrained weight.')
        checkpoint = jt.load(args.init_checkpoint)
        lm_net.load_weight(checkpoint)    

    lm_net.train()

    optimizer = create_adam_optimizer_from_args(lm_net, args)

    scheduler = create_optimizer_scheduler(optimizer, args)
    
    try:
        train_step = 0
        for epoch in itertools.count(start=1):
            train_step = train_validate(
                lm_net, optimizer, scheduler, 
                train_loader, valid_loader, args, 
                train_step, epoch
            )
            if args.max_step is not None and train_step >= args.max_step:
                if args.rank == 0:
                    print('-' * 100)
                    print('End of training')
                break
    except KeyboardInterrupt:
        if args.rank == 0:
            print('-' * 100)
            print('Exiting from training early')

    cleanup(args)

if __name__ == '__main__':
    main()