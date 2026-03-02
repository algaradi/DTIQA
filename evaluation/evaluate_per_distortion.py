import os
import sys

# Add the project root to the python path so it can find config, core, datasets, etc.
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import random
import numpy as np
import torch
from config.config import cfg
from collections import defaultdict
from scipy import stats

from core.solver import IQASolver
from datasets import data_loader

def main(config):
    if config.dataset not in cfg.img_num:
        raise ValueError(f"Dataset {config.dataset} not configured.")

    # Define distortion key mappings exactly as they appear in the file paths
    # This guarantees 100% accurate filtering without relying on fragile index bounds
    LIVE_DISTORTION_PATTERNS = {
        'jp2k': '/jp2k/',
        'jpeg': '/jpeg/',
        'wn': '/wn/',
        'gblur': '/gblur/',
        'ff': '/fastfading/'
    }
    
    CSIQ_DISTORTION_PATTERNS = {
        'awgn': '.awgn.',
        'jpeg': '.jpeg.',
        'jpeg2000': '.jpeg2000.',
        'fnoise': '.fnoise.',
        'blur': '.blur.',
        'contrast': '.contrast.'
    }
    
    if config.dataset == 'live':
        distortion_patterns = LIVE_DISTORTION_PATTERNS
    elif config.dataset == 'csiq':
        distortion_patterns = CSIQ_DISTORTION_PATTERNS
    else:
        raise ValueError(f"Per-distortion evaluation only supports 'live' and 'csiq', got {config.dataset}")

    source_num = cfg.img_num[config.dataset]

    dataset_model_type_save_dir = os.path.join(config.model_save_base_dir, f"{config.dataset}_perDistortion_{config.model_type}")
    if not os.path.exists(dataset_model_type_save_dir):
        os.makedirs(dataset_model_type_save_dir)

    print(f'Train and Per-Distortion Test on {config.dataset}...')

    # Track overall performance
    srcc_all_overall = np.zeros(config.train_test_num, dtype=float)
    plcc_all_overall = np.zeros(config.train_test_num, dtype=float)
    
    # Track per-distortion performance
    srcc_all_dist = {dist: np.zeros(config.train_test_num, dtype=float) for dist in distortion_patterns.keys()}
    plcc_all_dist = {dist: np.zeros(config.train_test_num, dtype=float) for dist in distortion_patterns.keys()}
    
    for i in range(config.train_test_num):
        print(f'\n================== Round {i + 1}/{config.train_test_num} ==================')
        random.shuffle(source_num)
        
        split_idx = int(round(0.8 * len(source_num)))
        train_index = source_num[0:split_idx]
        test_index = source_num[split_idx:len(source_num)]

        current_round_save_dir = os.path.join(dataset_model_type_save_dir, f"round_{i + 1}")
        if not os.path.exists(current_round_save_dir):
            os.makedirs(current_round_save_dir)

        np.save(os.path.join(current_round_save_dir, "train_index.npy"), np.array(train_index))
        np.save(os.path.join(current_round_save_dir, "test_index.npy"), np.array(test_index))

        # Train and evaluate overall performance
        solver = IQASolver(config, cfg.folder_path[config.dataset], train_index, test_index,
                           save_dir_for_round=current_round_save_dir)
        
        # The best weights are automatically restored to the model at the end of solver.train()
        best_srcc, best_plcc = solver.train()
        srcc_all_overall[i] = best_srcc
        plcc_all_overall[i] = best_plcc
        
        print(f"Round {i+1} Overall SRCC: {best_srcc:.4f}")
        
        # Evaluate isolated distortion subsets directly on the restored test model
        for dist_name, dist_pattern in distortion_patterns.items():
            
            # Map test bounds to explicit file index bounds to correctly filter distorted image selections
            full_test_loader_obj = data_loader.DataLoader(config.dataset, cfg.folder_path[config.dataset], test_index, 
                                            config.patch_size, config.test_patch_num, 
                                            batch_size=config.batch_size, istrain=False, 
                                            num_workers=config.num_workers)
            
            original_samples = full_test_loader_obj.data.samples
            
            # Filter samples by checking if the image path contains the explicit distortion signature
            filtered_samples = []
            for sample in original_samples:
                path = sample[0].lower().replace('\\', '/')
                if dist_pattern in path:
                    filtered_samples.append(sample)
                
            if len(filtered_samples) > 0:
                full_test_loader_obj.data.samples = filtered_samples
                dist_data = full_test_loader_obj.get_data()
                dist_srcc, dist_plcc = solver.test(dist_data)
            else:
                dist_srcc, dist_plcc = 0.0, 0.0
                
                
            srcc_all_dist[dist_name][i] = dist_srcc
            plcc_all_dist[dist_name][i] = dist_plcc
            print(f"  -> {dist_name} SRCC: {dist_srcc:.4f}, PLCC: {dist_plcc:.4f}")

    # Aggregated Summary Medians
    overall_srcc_med = np.median(srcc_all_overall)
    overall_plcc_med = np.median(plcc_all_overall)

    print(f'\nTraining complete! Overall Median Test SRCC: {overall_srcc_med:.4f}')
    
    results_summary_path = os.path.join(dataset_model_type_save_dir, "_per_distortion_summary_results.txt")
    with open(results_summary_path, 'w') as f:
        f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Model Type: {config.model_type}, Backbone: {config.backbone_type}\n")
        f.write(f"Overall 10-Fold Median SRCC: {overall_srcc_med:.4f}, PLCC: {overall_plcc_med:.4f}\n\n")
        
        f.write("=== Per-Distortion Medians ===\n")
        for dist_name in distortion_patterns.keys():
            d_srcc_med = np.median(srcc_all_dist[dist_name])
            d_plcc_med = np.median(plcc_all_dist[dist_name])
            f.write(f"{dist_name.ljust(10)}: Median SRCC = {d_srcc_med:.4f}, Median PLCC = {d_plcc_med:.4f}\n")
            print(f"Median {dist_name}: SRCC = {d_srcc_med:.4f}, PLCC = {d_plcc_med:.4f}")
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standard Train and exact Per-Distortion evaluation test")
    parser.add_argument('--dataset', type=str, required=True, help='Source Dataset (e.g. live or csiq)')
    parser.add_argument('--model_type', type=str, default='direct')
    parser.add_argument('--backbone_type', type=str, default='vit16', choices=['resnet50', 'vit16', 'vit32', 'swin_base', 'swin_tiny'])
    parser.add_argument('--backbone_strategy', type=str, default='finetune_all', choices=['freeze_all', 'freeze_bn', 'finetune_all'])
    
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-3)
    parser.add_argument('--backbone_lr', type=float, default=1e-5)
    parser.add_argument('--T_max', type=int, default=40)
    parser.add_argument('--eta_min', type=float, default=1e-7)
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mse'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=20)
    
    parser.add_argument('--patch_size', type=int, default=224)
    parser.add_argument('--train_patch_num', type=int, default=25)
    parser.add_argument('--test_patch_num', type=int, default=25)
    
    parser.add_argument('--train_test_num', type=int, default=10)
    parser.add_argument('--model_save_base_dir', type=str, default='/root/autodl-tmp/checkpoints_perDistortion')
    
    parser.add_argument('--feature_size', type=int, default=14)
    parser.add_argument('--fc_intermediate_dim', type=int, default=256)
    parser.add_argument('--predictor_hidden_dim', type=int, default=256)
    
    config = parser.parse_args()
    main(config)
