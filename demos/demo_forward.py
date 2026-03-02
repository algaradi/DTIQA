import torch
import argparse
import time

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import build_model

def main():
    parser = argparse.ArgumentParser(description="DTIQA Forward Pass and Architecture Check")
    parser.add_argument('--backbone', type=str, default='resnet50', 
                        choices=['resnet50', 'vit16', 'vit32', 'swin_base', 'swin_tiny'],
                        help='Backbone type to test')
    parser.add_argument('--batch_size', type=int, default=2)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️ Testing environment: {device}")

    if 'vit32' in args.backbone:
        feature_size = 7
    elif 'vit16' in args.backbone:
        feature_size = 14
    elif 'swin' in args.backbone:
        feature_size = 7
    else:  # resnet50
        feature_size = 14

    model_kwargs = {
        'feature_size': feature_size,
        'fc_intermediate_dim': 256,
        'predictor_hidden_dim': 256,
        'backbone_type': args.backbone
    }

    try:
        print(f"🏗️ Building DTIQA model with backbone '{args.backbone}'...")
        t0 = time.time()
        model, _ = build_model(model_type='direct', **model_kwargs)
        model = model.to(device)
        model.eval()
        t1 = time.time()
        print(f"✅ Model built and moved to {device} in {t1 - t0:.2f} seconds.")

        # Create dummy input tensor
        dummy_input = torch.randn(args.batch_size, 3, 224, 224).to(device)
        print(f"\n📦 Created dummy input tensor with shape: {dummy_input.shape}")

        print("🚀 Running forward pass...")
        t2 = time.time()
        with torch.no_grad():
            output = model(dummy_input)
        t3 = time.time()
        print(f"✅ Forward pass completed in {t3 - t2:.2f} seconds.")

        print(f"\n{'='*40}")
        print(f"🎉 ARCHITECTURE TEST SUCCESSFUL! 🎉")
        print(f"{'='*40}")
        print(f"Output shape: {output.shape} (Expected: torch.Size([{args.batch_size}]))")
        print(f"Sample predicted quality scores: {output.cpu().numpy()}")
        print(f"{'='*40}\n")

    except Exception as e:
        print(f"\n❌ ARCHITECTURE TEST FAILED!")
        print(f"Error during execution: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
