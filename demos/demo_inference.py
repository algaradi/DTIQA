import os
import argparse
import torch
import numpy as np
from PIL import Image
from torchvision import transforms


import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model

def get_transform(patch_size=224):
    return transforms.Compose([
        transforms.Resize((patch_size, patch_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def main():
    parser = argparse.ArgumentParser(description="DTIQA Demo: Predict quality score of an image")
    parser.add_argument('--image_path', type=str, required=True, help='Path to the input image')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model checkpoint (.pth)')
    parser.add_argument('--backbone_type', type=str, default='vit16', choices=['resnet50', 'vit16', 'vit32', 'swin_base', 'swin_tiny'])
    parser.add_argument('--patch_size', type=int, default=224, help='Image resize dimension (default: 224)')
    parser.add_argument('--feature_size', type=int, default=7, help='Spatial size of feature map (e.g., 7 for ViT-16@224px, 14 for ResNet)')
    args = parser.parse_args()

    if not os.path.exists(args.image_path):
        raise FileNotFoundError(f"Image not found: {args.image_path}")
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {args.model_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading image from {args.image_path}...")
    img = Image.open(args.image_path).convert('RGB')
    transform = get_transform(args.patch_size)
    img_tensor = transform(img).unsqueeze(0).to(device)

    print(f"Building model with backbone {args.backbone_type}...")
    model_kwargs = {
        'feature_size': args.feature_size,
        'fc_intermediate_dim': 256,
        'predictor_hidden_dim': 256,
        'backbone_type': args.backbone_type
    }
    model, _ = build_model(model_type='direct', **model_kwargs)

    print(f"Loading weights from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    print("Predicting quality score...")
    with torch.no_grad():
        score = model(img_tensor).item()

    print(f"\n{'='*40}")
    print(f"Predicted Quality Score: {score:.4f}")
    print(f"{'='*40}\n")

if __name__ == '__main__':
    main()
