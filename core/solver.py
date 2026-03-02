import torch
from scipy import stats
import numpy as np
import models  # Assuming models.py contains the build_model factory and DirectQualityModel
from datasets import data_loader
import os
import random  # Add
import numpy as np  # Add


class IQASolver(object):
    """Solver for training and testing DirectQualityModel"""

    def __init__(self, config, path, train_idx, test_idx, save_dir_for_round, cross_dataset=None, cross_path=None, cross_idx=None):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Dataset scaling definitions for [0, 1] normalization bounding
        self.dataset_scales = {
            'live': (0.0, 100.0),
            'csiq': (0.0, 1.0),
            'tid2013': (0.0, 9.0),
            'livec': (0.0, 100.0),
            'koniq-10k': (1.0, 5.0),
            'bid': (0.0, 5.0)
        }
        self.epochs = config.epochs
        self.test_patch_num = config.test_patch_num
        self.model_save_dir = save_dir_for_round

        if config.model_type != 'direct':
            raise ValueError(f"This solver now only supports 'direct' model type, got {config.model_type}")

        model_kwargs = {
            'feature_size': config.feature_size,
            'fc_intermediate_dim': config.fc_intermediate_dim,
            'predictor_hidden_dim': config.predictor_hidden_dim,
            'backbone_type': getattr(config, 'backbone_type', 'resnet50')
        }

        self.model, _ = models.build_model(model_type='direct', **model_kwargs)
        self.model = self.model.to(self.device)


        # Layered learning rate configuration
        backbone_params = []
        head_params = []
        
        # Backbone fine-tuning strategy
        backbone_strategy = getattr(config, 'backbone_strategy', 'finetune_all')  # Default: fine-tune all
        # Options: 'freeze_all', 'freeze_bn', 'finetune_all'
        
        # Parameter grouping for backbone and head
        for name, param in self.model.named_parameters():
            if 'backbone' in name:
                backbone_params.append(param)
            else:
                head_params.append(param)
        
        # Calculate actual parameter counts
        backbone_param_count = sum(p.numel() for p in backbone_params)
        head_param_count = sum(p.numel() for p in head_params)
        total_param_count = backbone_param_count + head_param_count
        
        # Process backbone parameters based on strategy
        if backbone_strategy == 'freeze_all':
            print("🔒 FREEZING ALL BACKBONE PARAMETERS")
            for param in backbone_params:
                param.requires_grad = False
            trainable_params = head_params
            param_groups = [
                {'params': head_params, 'lr': config.lr, 'name': 'head'}
            ]
        elif backbone_strategy == 'freeze_bn':
            print("🔒 FREEZING BACKBONE BATCHNORM LAYERS ONLY")
            frozen_count = 0
            for name, param in self.model.named_parameters():
                if 'backbone' in name and ('bn' in name or 'norm' in name):
                    param.requires_grad = False
                    frozen_count += param.numel()
            print(f"Frozen BN parameters: {frozen_count:,}")
            trainable_params = [p for p in self.model.parameters() if p.requires_grad]
            param_groups = [
                {'params': backbone_params, 'lr': config.backbone_lr, 'name': 'backbone'},
                {'params': head_params, 'lr': config.lr, 'name': 'head'}
            ]
        else:  # finetune_all
            print("🔓 FINE-TUNING ALL BACKBONE PARAMETERS")
            trainable_params = backbone_params + head_params
            param_groups = [
                {'params': backbone_params, 'lr': config.backbone_lr, 'name': 'backbone'},
                {'params': head_params, 'lr': config.lr, 'name': 'head'}
            ]
        
        print(f"Backbone parameters: {backbone_param_count:,} ({len(backbone_params)} tensors)")
        print(f"Head parameters: {head_param_count:,} ({len(head_params)} tensors)")
        print(f"Total parameters: {total_param_count:,}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")
        
        if backbone_strategy != 'freeze_all':
            print(f"Backbone LR: {config.backbone_lr }")
        print(f"Head LR: {config.lr:.2e}")

        # trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())

        # Initialize optimizer with layered learning rates
        print(f"Initializing optimizer with layered learning rates")
        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=config.weight_decay)

        # Using CosineAnnealingLR scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.T_max,
            eta_min=config.eta_min
        )


        # Choose loss function based on config.loss_type (default to 'l1' if not set)
        loss_type = config.loss_type
        if loss_type == 'mse':
            self.loss_fn = torch.nn.MSELoss().to(self.device)
            print("Using MSE Loss")
        else:
            self.loss_fn = torch.nn.L1Loss().to(self.device)
            print("Using L1 Loss")


        num_workers = getattr(config, 'num_workers', 8)
        train_loader = data_loader.DataLoader(config.dataset, path, train_idx, config.patch_size,
                                              config.train_patch_num, batch_size=config.batch_size, istrain=True, num_workers=num_workers)
        test_loader = data_loader.DataLoader(config.dataset, path, test_idx, config.patch_size,
                                             config.test_patch_num, batch_size=config.batch_size, istrain=False, num_workers=num_workers)
        self.train_data = train_loader.get_data()
        self.test_data = test_loader.get_data()

        # Cross-dataset loading
        self.cross_test_data = None
        if cross_dataset and cross_path and cross_idx:
            print(f"Loading cross-test dataset: {cross_dataset}")
            cross_loader = data_loader.DataLoader(cross_dataset, cross_path, cross_idx, config.patch_size,
                                                 config.test_patch_num, batch_size=config.batch_size, istrain=False, num_workers=num_workers)
            self.cross_test_data = cross_loader.get_data()

    def normalize_scores(self, scores, dataset_name):
        """
        Normalize scores to range [0,1] using min-max normalization.

        Args:
            scores: Tensor or numpy array of scores to normalize
            dataset_name: Used to lookup the specific scale bounds of the dataset

        Returns:
            Normalized scores in same format as input
        """
        min_score, max_score = self.dataset_scales.get(dataset_name.lower(), (0.0, 100.0)) # Default to [0,100] if not found
        
        if isinstance(scores, torch.Tensor):
            return (scores - min_score) / (max_score - min_score)
        elif isinstance(scores, np.ndarray):
            return (scores - min_score) / (max_score - min_score)
        else:
            raise TypeError("Input must be either torch.Tensor or numpy.ndarray")

    def train(self):
        best_srcc = 0.0;
        best_plcc = 0.0;
        best_epoch = 0
        best_model_path = ""
        patience = 10  # Early stopping patience
        patience_counter = 0
        print('Epoch\tTrain_Loss\tTrain_SRCC\tTest_SRCC\tTest_PLCC\tTest_SRCC_C\tTest_PLCC_C\tLR')
        self.model.train(True)
        for t in range(self.epochs):
            epoch_loss = [];
            pred_scores_epoch = [];
            gt_scores_epoch = []

            # Use tqdm for progress bar
            from tqdm import tqdm
            pbar = tqdm(self.train_data, desc=f"Epoch {t+1}/{self.epochs}", ncols=80)
            for img, label in pbar:
                img = img.to(self.device, dtype=torch.float)
                label = label.to(self.device, dtype=torch.float)
            
                # Normalize labels based on source dataset bounds to match [0, 1] output prediction
                label = self.normalize_scores(label.squeeze(), self.config.dataset)

                self.optimizer.zero_grad()
                pred = self.model(img)
                # normalize scores to [0,100]
                # pred = self.normalize_scores(pred.squeeze(), min_score=0, max_score=100)
                # loss function
                # specific normalization for some datasets
                
                # loss function
                loss = self.loss_fn(pred,label)
                epoch_loss.append(loss.item())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                # Update progress bar with current loss
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})


                            # Ensure pred is at least 1-dimensional before converting to list
                if pred.ndim == 0:  # If it's a scalar tensor
                    pred_scores_epoch.append(pred.item())
                else:
                    pred_scores_epoch.extend(pred.detach().cpu().tolist())
    
                if label.ndim == 0: # Do the same for label if it could also be a scalar
                    gt_scores_epoch.append(label.item())
                else:
                    gt_scores_epoch.extend(label.detach().cpu().tolist())

            # Step the scheduler at the end of each epoch
            self.scheduler.step()

            train_srcc_value, _ = stats.spearmanr(np.array(pred_scores_epoch), np.array(gt_scores_epoch))
            # Handle potential NaN from spearmanr if all predictions or labels are identical
            train_srcc = np.abs(train_srcc_value) if not np.isnan(train_srcc_value) else 0.0

            avg_epoch_loss = sum(epoch_loss) / len(epoch_loss) if epoch_loss else float('inf')
            current_lr = self.optimizer.param_groups[0]['lr']

            test_srcc, test_plcc = self.test(self.test_data)
            self.model.train(True)

            if test_srcc > best_srcc:
                best_srcc = test_srcc
                best_plcc = test_plcc
                best_epoch = t + 1
                patience_counter = 0  # Reset patience counter
                model_path = os.path.join(self.model_save_dir,
                                        f"{self.config.dataset}_{self.config.model_type}_best_epoch{t + 1}_srcc{test_srcc:.4f}_plcc{test_plcc:.4f}.pth")
                state_dict = {
                    'epoch': t + 1, 'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(), 'best_srcc': best_srcc, 'best_plcc': best_plcc,
                }
                best_model_path = model_path
                torch.save(state_dict, best_model_path)
                print(f"Best model saved to {best_model_path}")
            else:
                patience_counter += 1
                
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping at epoch {t + 1} due to no improvement for {patience} epochs")
                break
            # Cross-Test Per Epoch
            cross_srcc = 0.0
            cross_plcc = 0.0
            if self.cross_test_data:
                cross_srcc, cross_plcc = self.test(self.cross_test_data)

            print(
                f'{t + 1}\t{avg_epoch_loss:.4f}\t\t{train_srcc:.4f}\t\t{test_srcc:.4f}\t\t{test_plcc:.4f}\t\t{cross_srcc:.4f}\t\t{cross_plcc:.4f}\tLR: {current_lr:.2e}')

        final_model_name = f"{self.config.dataset}_{self.config.model_type}_final_epoch{self.epochs}_best_srcc{best_srcc:.4f}_plcc{best_plcc:.4f}.pth"
        final_model_path = os.path.join(self.model_save_dir, final_model_name)
        final_state = {
            'epoch': self.epochs, 'state_dict': self.model.state_dict(),
            'best_epoch': best_epoch, 'best_srcc': best_srcc, 'best_plcc': best_plcc,
        }
        torch.save(final_state, final_model_path)
        print(f"Final model saved to {final_model_path}")
        print(f'Best test SRCC {best_srcc:.6f}, PLCC {best_plcc:.6f} at epoch {best_epoch}')
        
        # Restore the best validation model before returning
        if best_model_path and os.path.exists(best_model_path):
            checkpoint = torch.load(best_model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['state_dict'])
        
        return best_srcc, best_plcc

    def test(self, data):
        self.model.eval()
        pred_scores_patches = []
        gt_scores_patches = []
        from tqdm import tqdm
        # Only show progress bar if data seems substantial or to reassure user
        pbar = tqdm(data, desc="Testing", leave=False, ncols=80)
        with torch.no_grad():
            for img, label in pbar:
                img = img.to(self.device, dtype=torch.float)
                label = label.to(self.device, dtype=torch.float)
                pred = self.model(img)

                # Look up which dataset we are actually testing right now (cross vs val)
                test_dataset_name = getattr(self.config, 'cross_dataset', self.config.dataset) if data == getattr(self, 'cross_test_data', None) else self.config.dataset
                
                label = self.normalize_scores(label, test_dataset_name)

                # Fix for TypeError: 'float' object is not iterable
                if pred.ndim == 0:
                    pred_list = [pred.item()]
                else:
                    pred_list = pred.detach().cpu().numpy().flatten().tolist()
                
                if label.ndim == 0:
                    label_list = [label.item()]
                else:
                    label_list = label.detach().cpu().numpy().flatten().tolist()


                pred_scores_patches.extend(pred_list)
                gt_scores_patches.extend(label_list)

        if not pred_scores_patches or not gt_scores_patches:
            print("Warning: No scores to evaluate in test method.")
            return 0.0, 0.0

        num_images = len(gt_scores_patches) // self.test_patch_num
        if num_images == 0 and len(gt_scores_patches) > 0:
            num_images = 1

        pred_scores_img_avg = np.array([])
        gt_scores_img_avg = np.array([])

        if num_images > 0:
            num_total_patches_to_consider = num_images * self.test_patch_num
            pred_scores_img_avg = np.mean(np.reshape(np.array(pred_scores_patches[:num_total_patches_to_consider]),
                                                     (num_images, self.test_patch_num)), axis=1)
            gt_scores_img_avg = np.mean(np.reshape(np.array(gt_scores_patches[:num_total_patches_to_consider]),
                                                   (num_images, self.test_patch_num)), axis=1)
        else:
            if pred_scores_patches:
                pred_scores_img_avg = np.mean(np.array(pred_scores_patches))
                gt_scores_img_avg = np.mean(np.array(gt_scores_patches))

        # Ensure results are array-like for stats functions and have enough points
        if not isinstance(pred_scores_img_avg,
                          np.ndarray) or pred_scores_img_avg.size == 0: pred_scores_img_avg = np.array(
            [0.0])  # fallback for safety
        if not isinstance(gt_scores_img_avg, np.ndarray) or gt_scores_img_avg.size == 0: gt_scores_img_avg = np.array(
            [0.0])

        if pred_scores_img_avg.ndim == 0: pred_scores_img_avg = np.array([pred_scores_img_avg.item()])
        if gt_scores_img_avg.ndim == 0: gt_scores_img_avg = np.array([gt_scores_img_avg.item()])

        if len(pred_scores_img_avg) < 2 or len(gt_scores_img_avg) < 2:
            return 0.0, 0.0  # Return 0 if not enough data points for correlation

        test_srcc, _ = stats.spearmanr(pred_scores_img_avg, gt_scores_img_avg)
        test_plcc, _ = stats.pearsonr(pred_scores_img_avg, gt_scores_img_avg)
        
        # In IQA, it is standard to report the absolute magnitude of correlation, 
        # especially for cross-dataset evaluation where datasets mix MOS (higher is better) 
        # and DMOS (lower is better), resulting in inherently negative correlations.
        test_srcc = np.abs(test_srcc) if not np.isnan(test_srcc) else 0.0
        test_plcc = np.abs(test_plcc) if not np.isnan(test_plcc) else 0.0
        
        return test_srcc, test_plcc


if __name__ == '__main__':
    import argparse
    import sys
    sys.path.append('..')
    print("Testing IQASolver Initialization...")
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone_type', default='resnet50')
    parser.add_argument('--model_type', default='direct')
    parser.add_argument('--feature_size', default=14)
    parser.add_argument('--fc_intermediate_dim', default=256)
    parser.add_argument('--predictor_hidden_dim', default=256)
    parser.add_argument('--loss_type', default='l1')
    parser.add_argument('--lr', default=5e-5)
    parser.add_argument('--backbone_lr', default=1e-5)
    parser.add_argument('--weight_decay', default=1e-3)
    parser.add_argument('--T_max', default=40)
    parser.add_argument('--eta_min', default=1e-7)
    parser.add_argument('--batch_size', default=2)
    parser.add_argument('--epochs', default=2)
    parser.add_argument('--patch_size', default=224)
    parser.add_argument('--train_patch_num', default=1)
    parser.add_argument('--test_patch_num', default=1)
    parser.add_argument('--num_workers', default=0)
    parser.add_argument('--backbone_strategy', default='finetune_all')
    config = parser.parse_args([])
    try:
        # Note: solver initialization hits folder path directly, might fail if none exists.
        solver = IQASolver(config, 'dummy_path', [0], [1], 'dummy_save')
        print("Solver structure instantiated successfully!")
    except Exception as e:
        print(f"Solver init note: {e}")
