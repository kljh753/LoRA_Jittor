#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class LossComparisonPlotter:
    def __init__(self, base_dir="."):
        self.base_dir = base_dir
        self.frameworks = ["jittor", "pytorch"]
        self.datasets = ["e2e", "webnlg", "dart"]
        self.results = {}

        self.output_dir = os.path.join(base_dir, "figure", "loss")
        
    def parse_training_log(self, log_path):
        if not os.path.exists(log_path):
            print(f"Warning: Log file does not exist {log_path}")
            return None
            
        losses = []
        try:
            with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                

            loss_patterns = [
                r'avg loss\s+([0-9]+\.?[0-9]*)',  # 匹配 "avg loss  5.85"
                r'loss\s+([0-9]+\.?[0-9]*)\s+\|\s+avg loss',  # 匹配 "loss  5.40 | avg loss"
                r'Loss[:\s=]+([0-9]+\.?[0-9]*)',
                r'train_loss[:\s=]+([0-9]+\.?[0-9]*)',
                r'training_loss[:\s=]+([0-9]+\.?[0-9]*)',
            ]
            
            for pattern in loss_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    losses = [float(x) for x in matches]
                    break
                    
        except Exception as e:
            print(f"Error parsing log file {log_path}: {e}")
            return None
            
        if not losses:
            return None
            
        return {
            'losses': losses,
            'total_steps': len(losses)
        }
    
    def collect_loss_data(self):
        print("=== Collecting Loss Data ===")
        for framework in self.frameworks:
            self.results[framework] = {}
            for dataset in self.datasets:
                # 构建路径，适配实际的项目结构
                if framework == "jittor":
                    log_path = os.path.join(self.base_dir, "LoRA_Jittor_NLG", "logs", dataset, "train.log")
                else:  # pytorch
                    log_path = os.path.join(self.base_dir, "LoRA_pytorch_NLG", "logs", dataset, "train.log")
                
                train_data = None
                if os.path.exists(log_path):
                    train_data = self.parse_training_log(log_path)
                    if train_data:
                        print(f"Found training data at: {log_path}")
                
                self.results[framework][dataset] = train_data
                status = "Success" if train_data else "Failed"
                count = len(train_data['losses']) if train_data and train_data['losses'] else 0
                print(f"{framework}-{dataset}: {status} ({count} data points)")
    
    def plot_individual_charts(self):
        print("=== Generating Individual Loss Charts ===")
        
        for framework in self.frameworks:
            framework_dir = os.path.join(self.output_dir, framework)
            os.makedirs(framework_dir, exist_ok=True)
            
            for dataset in self.datasets:
                dataset_dir = os.path.join(framework_dir, dataset)
                os.makedirs(dataset_dir, exist_ok=True)
                
                train_data = self.results[framework][dataset]
                if train_data and train_data['losses']:
                    plt.figure(figsize=(10, 6))
                    losses = train_data['losses']
                    steps = list(range(len(losses)))
                    
                    plt.plot(steps, losses, 'b-', linewidth=2, alpha=0.8)
                    plt.title(f'{framework.capitalize()} - {dataset.upper()} Dataset Loss Curve', 
                              fontsize=14, fontweight='bold')
                    plt.xlabel('Step', fontsize=12)
                    plt.ylabel('Loss', fontsize=12)
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    
                    filename = f'{dataset}_loss_curve.png'
                    filepath = os.path.join(dataset_dir, filename)
                    plt.savefig(filepath, dpi=300, bbox_inches='tight')
                    plt.close()
                    print(f"Saved individual chart: {filepath}")
    
    def plot_comparison_charts(self):
        print("=== Generating Loss Comparison Charts ===")
        comparison_dir = os.path.join(self.output_dir, "comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        
        colors = {
            'jittor': '#FF6B35',
            'pytorch': '#004E89'
        }
        line_styles = {
            'jittor': '-',
            'pytorch': '--'
        }
        
        for dataset in self.datasets:
            plt.figure(figsize=(12, 8))
            has_data = False
            max_steps = 0
            
            for framework in self.frameworks:
                train_data = self.results[framework][dataset]
                if train_data and train_data['losses']:
                    losses = train_data['losses']
                    steps = list(range(len(losses)))
                    max_steps = max(max_steps, len(losses))
                    
                    plt.plot(steps, losses, 
                             label=f'{framework.capitalize()} Train Loss', 
                             linewidth=2.5,
                             color=colors[framework],
                             linestyle=line_styles[framework],
                             alpha=0.8)
                    has_data = True
                    print(f"Plotted comparison {framework} {dataset}: {len(losses)} data points")
            
            if has_data:
                plt.title(f'Jittor vs PyTorch Loss - {dataset.upper()} Dataset', 
                          fontsize=16, fontweight='bold', pad=20)
                plt.xlabel('Step', fontsize=14)
                plt.ylabel('Loss', fontsize=14)
                plt.legend(fontsize=12, loc='upper right')
                plt.grid(True, alpha=0.3, linestyle=':', linewidth=0.5)
                
                if max_steps > 0:
                    plt.xlim(0, max_steps)
                
                plt.tight_layout()
                
                filename = f'{dataset}_loss_comparison.png'
                filepath = os.path.join(comparison_dir, filename)
                plt.savefig(filepath, dpi=300, bbox_inches='tight', 
                            facecolor='white', edgecolor='none')
                plt.close()
                print(f"Saved comparison: {filepath}")
            else:
                plt.close()
                print(f"No training data available for {dataset} dataset")
    
    def run(self):
        print("Starting Loss Curves Generation...")
        print(f"Output directory: {self.output_dir}")
        self.collect_loss_data()
        self.plot_individual_charts()
        self.plot_comparison_charts()
        print("\n=== Loss Curves Generation Complete ===")

if __name__ == "__main__":
    plotter = LossComparisonPlotter()
    plotter.run()