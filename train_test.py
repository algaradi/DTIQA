import os
import argparse
import random
import numpy as np
import torch
from config.config import cfg

from core.solver import IQASolver


def main(config):
    is_cross = getattr(config, 'cross_dataset', None) is not None
    if config.dataset not in cfg.img_num:
        raise ValueError(f"Dataset {config.dataset} not configured in folder settings.")
    if is_cross and config.cross_dataset not in cfg.img_num:
        raise ValueError(f"Cross dataset {config.cross_dataset} not configured in folder settings.")
    
    sel_num = cfg.img_num[config.dataset]

    srcc_all = np.zeros(config.train_test_num, dtype=float)
    plcc_all = np.zeros(config.train_test_num, dtype=float)

    if is_cross:
        save_dir_name = f"{config.dataset}_to_{config.cross_dataset}_{config.model_type}"
        print(f'Cross-Dataset Train on {config.dataset} and Test on {config.cross_dataset} using {config.backbone_type} model...')
    else:
        save_dir_name = f"{config.dataset}_{config.model_type}"
        print(f'Standard Train/Test on {config.dataset} dataset using {config.backbone_type} model for {config.train_test_num} rounds...')

    dataset_model_type_save_dir = os.path.join(config.model_save_base_dir, save_dir_name)
    if not os.path.exists(dataset_model_type_save_dir):
        os.makedirs(dataset_model_type_save_dir)
    
    for i in range(config.train_test_num):
        print(f'Round {i + 1}/{config.train_test_num}')
        random.shuffle(sel_num)
        
        split_idx = int(round(0.8 * len(sel_num)))
        train_index = sel_num[0:split_idx]
        test_index = sel_num[split_idx:len(sel_num)]

        current_round_save_dir = os.path.join(dataset_model_type_save_dir, f"round_{i + 1}")
        if not os.path.exists(current_round_save_dir):
            os.makedirs(current_round_save_dir)

        np.save(os.path.join(current_round_save_dir, "train_index.npy"), np.array(train_index))
        np.save(os.path.join(current_round_save_dir, "test_index.npy"), np.array(test_index))

        kwargs = {}
        if is_cross:
            cross_index = list(cfg.img_num[config.cross_dataset])
            np.save(os.path.join(current_round_save_dir, "cross_test_index.npy"), np.array(cross_index))
            kwargs = {
                'cross_dataset': config.cross_dataset,
                'cross_path': cfg.folder_path[config.cross_dataset],
                'cross_idx': cross_index
            }

        solver = IQASolver(config, cfg.folder_path[config.dataset], train_index, test_index,
                           save_dir_for_round=current_round_save_dir, **kwargs)
        
        if is_cross:
            solver.train()
            srcc_all[i], plcc_all[i] = solver.test(solver.cross_test_data)
        else:
            srcc_all[i], plcc_all[i] = solver.train()

    srcc_med = np.median(srcc_all)
    plcc_med = np.median(plcc_all)

    print(f'\nComplete! Median SRCC: {srcc_med:.4f}, Median PLCC: {plcc_med:.4f} over {config.train_test_num} rounds.')

    results_summary_path = os.path.join(dataset_model_type_save_dir, "_overall_summary_results.txt")
    with open(results_summary_path, 'w') as f:
        if is_cross:
            f.write(f"Source Dataset: {config.dataset} -> Target Dataset: {config.cross_dataset}\n")
        else:
            f.write(f"Dataset: {config.dataset}\n")
        f.write(f"Model Type: {config.model_type}\n")
        f.write(f"Backbone: {config.backbone_type}\n")
        f.write(f"Train/Test Rounds: {config.train_test_num}\n")
        f.write(f"Median SRCC: {srcc_med:.4f}\n")
        f.write(f"Median PLCC: {plcc_med:.4f}\n")
        f.write("\nIndividual Round Results (SRCC, PLCC):\n")
        for r_idx in range(config.train_test_num):
            f.write(f"Round {r_idx + 1}: SRCC={srcc_all[r_idx]:.4f}, PLCC={plcc_all[r_idx]:.4f}\n")
    print(f"Overall results summary saved to {results_summary_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Standard Train and Test script for Intra-dataset evaluation")
    parser.add_argument('--dataset', type=str, default='live', help='Support datasets: livec|koniq-10k|bid|live|csiq|tid2013')
    parser.add_argument('--cross_dataset', type=str, default=None, help='Target dataset for cross-dataset evaluation. If None, performs standard intra-dataset evaluation.')
    parser.add_argument('--model_type', type=str, default='direct', help='Model type.')
    parser.add_argument('--backbone_type', type=str, default='vit16', choices=['resnet50', 'vit16', 'vit32', 'swin_base', 'swin_tiny'])
    parser.add_argument('--backbone_strategy', type=str, default='finetune_all', choices=['freeze_all', 'freeze_bn', 'finetune_all'])
    
    # Training configurations
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate for trainable parameters')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay for AdamW optimizer')
    parser.add_argument('--backbone_lr', type=float, default=1e-5, help='Learning rate for backbone parameters')
    parser.add_argument('--T_max', type=int, default=40, help='Maximum number of epochs for cosine annealing')
    parser.add_argument('--eta_min', type=float, default=1e-7, help='Minimum learning rate for cosine annealing')
    parser.add_argument('--loss_type', type=str, default='l1', choices=['l1', 'mse'], help='Loss function type')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of data loading workers')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs for training')
    
    # Patch configurations
    parser.add_argument('--patch_size', type=int, default=224, help='Crop size for training & testing image patches')
    parser.add_argument('--train_patch_num', type=int, default=25, help='Number of sample patches from training image')
    parser.add_argument('--test_patch_num', type=int, default=25, help='Number of sample patches from testing image')
    
    parser.add_argument('--train_test_num', type=int, default=10, help='Number of train-test rounds (e.g. 10 folds)')
    parser.add_argument('--model_save_base_dir', type=str, default='/root/autodl-tmp/checkpoints', help='Base directory for saving models')
    
    # DPAtten specific dimensions
    parser.add_argument('--feature_size', type=int, default=14, help='Spatial size for pooled features (e.g. 14 for ViT-16/ResNet)')
    parser.add_argument('--fc_intermediate_dim', type=int, default=256, help='Intermediate FC dimension in MLPs')
    parser.add_argument('--predictor_hidden_dim', type=int, default=256, help='Hidden dimension in final predictor MLP')
    config = parser.parse_args()
    main(config)
