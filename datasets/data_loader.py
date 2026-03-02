import torch
import torchvision
from . import folders
import numpy as np
import random

class DataLoader(object):
    """Dataset class for IQA databases"""

    def __init__(self, dataset, path, img_indx, patch_size, patch_num, batch_size=1, istrain=True, num_workers=0):

        self.batch_size = batch_size
        self.istrain = istrain
        self.num_workers = num_workers
        
        if (dataset == 'live') | (dataset == 'csiq') | (dataset == 'tid2013') | (dataset == 'livec'):
            # Train transforms
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.RandomVerticalFlip(),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
                # Alternative transforms with proportional resize to 224 - uncomment to use
                # transforms = torchvision.transforms.Compose([
                #     torchvision.transforms.RandomHorizontalFlip(),
                #     torchvision.transforms.RandomVerticalFlip(),  # 随机垂直翻转
                #     torchvision.transforms.RandomRotation(degrees=15),  # 随机旋转±15度
                #     torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色增强
                #     torchvision.transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),  # 随机仿射变换
                #     torchvision.transforms.Resize((224,224)),  # Resize shorter side to 224, keeping aspect ratio
                #     torchvision.transforms.ToTensor(),
                #     torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                #                                     std=(0.229, 0.224, 0.225))
                # ])
            # Test transforms
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))
                ])
        elif dataset == 'koniq-10k':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 384)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
        elif dataset == 'bid':
            if istrain:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.RandomHorizontalFlip(),
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])
            else:
                transforms = torchvision.transforms.Compose([
                    torchvision.transforms.Resize((512, 512)),
                    torchvision.transforms.RandomCrop(size=patch_size),
                    torchvision.transforms.ToTensor(),
                    torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                                     std=(0.229, 0.224, 0.225))])

        if dataset == 'live':
            self.data = folders.LIVEFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'livec':
            self.data = folders.LIVEChallengeFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'csiq':
            self.data = folders.CSIQFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'koniq-10k':
            self.data = folders.Koniq_10kFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'bid':
            self.data = folders.BIDFolder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        elif dataset == 'tid2013':
            self.data = folders.TID2013Folder(
                root=path, index=img_indx, transform=transforms, patch_num=patch_num)
        
    def get_data(self):
        
       
        if self.istrain:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        else:
            dataloader = torch.utils.data.DataLoader(
                self.data, batch_size=1, shuffle=False, num_workers=self.num_workers)
        
        return dataloader

if __name__ == '__main__':
    import sys
    sys.path.append('..')
    from config.config import cfg
    print("Testing DataLoader Object Initializations...")
    try:
        loader = DataLoader('live', cfg.folder_path['live'], list(range(5)), 224, 1, batch_size=2, istrain=True, num_workers=0)
        print("DataLoader instance successfully created.")
    except Exception as e:
        print(f"DataLoader instance creation notice (likely due to path not existing): {e}")
