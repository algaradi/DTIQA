import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torch.utils.model_zoo as model_zoo
from torchvision.ops import DeformConv2d

try:
    import timm
except ImportError:
    print("Warning: timm not installed. ViT backbones might not work fully.")
    timm = None

# Official ResNet50 URL (for fallback or custom use if needed, though we use torchvision style loading)
model_urls = {
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
}

# ============================================================================
# ResNet Family Backbones
# ============================================================================

def weights_init_kaiming(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None: init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        init.constant_(m.weight, 1.0); init.constant_(m.bias, 0.0)

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False); self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False); self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False); self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True); self.downsample = downsample; self.stride = stride
    def forward(self, x):
        residual = x; out = self.conv1(x); out = self.bn1(out); out = self.relu(out)
        out = self.conv2(out); out = self.bn2(out); out = self.relu(out); out = self.conv3(out); out = self.bn3(out)
        if self.downsample is not None: residual = self.downsample(x)
        out += residual; out = self.relu(out); return out

class SimplifiedResNetBackbone(nn.Module):
    """ Simplified ResNetBackbone for DirectQualityModel: only outputs l1-l4 features. """
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], lda_out_channels=None, in_chn=None):
        super(SimplifiedResNetBackbone, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.apply(weights_init_kaiming) 

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False), 
                                     nn.BatchNorm2d(planes * block.expansion))
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        l1 = self.layer1(x); l2 = self.layer2(l1); l3 = self.layer3(l2); l4 = self.layer4(l3)
        return {'l1': l1, 'l2': l2, 'l3': l3, 'l4': l4}

def resnet50_backbone(pretrained=True, **kwargs):
    model = SimplifiedResNetBackbone(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        try:
            from torchvision.models import resnet50, ResNet50_Weights
            # Use torchvision weights if possible for better connectivity
            print("Loading ResNet50 weights from torchvision...")
            tv_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
            state_dict = tv_model.state_dict()
            # Filter matches
            model_dict = model.state_dict()
            state_dict = {k: v for k, v in state_dict.items() if k in model_dict and model_dict[k].shape == v.shape}
            model_dict.update(state_dict)
            model.load_state_dict(model_dict)
        except Exception as e:
            print(f"Torchvision load failed, trying legacy url: {e}")
            try:
                save_model = model_zoo.load_url(model_urls['resnet50'])
                model_dict = model.state_dict()
                state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys() and model_dict[k].shape == v.shape}
                model_dict.update(state_dict)
                model.load_state_dict(model_dict)
            except Exception as e2:
                 print(f"Error loading ResNet50 weights: {e2}. Using Kaiming init.")
    return model

# ============================================================================
# Transformer Backbones (ViT & Swin)
# ============================================================================

class ViTBackbone(nn.Module):
    """
    ViT Backbone wrapper using Torchvision (Official PyTorch Weights).
    Solves download issues in China by using official PyTorch CDN.
    """
    def __init__(self, model_name='vit_base_patch16_224', pretrained=True):
        super(ViTBackbone, self).__init__()
        import torchvision.models as models
        
        # Map model_name to torchvision function
        # vit_base_patch16_224 -> vit_b_16
        # vit_base_patch32_224 -> vit_b_32
        if 'patch16' in model_name:
            tv_model_func = models.vit_b_16
        elif 'patch32' in model_name:
            tv_model_func = models.vit_b_32
        else:
            tv_model_func = models.vit_b_16 # Default
            
        weights = 'DEFAULT' if pretrained else None
        
        try:
            print(f"Loading {model_name} from torchvision (download.pytorch.org)...")
            self.model = tv_model_func(weights=weights)
        except Exception as e:
            print(f"Error loading from torchvision: {e}")
            if pretrained:
                print("Fallback: Using random initialization.")
                self.model = tv_model_func(weights=None)
            else:
                raise e

        # ViT-Base has 12 blocks. We take outputs from blocks 2, 5, 8, 11 (0-indexed)
        self.hook_indices = [2, 5, 8, 11]
        self.features = {}
        
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # Register hooks on torchvision ViT encoder layers
        # Structure: model.encoder.layers (Sequential of EncoderBlock)
        for i, idx in enumerate(self.hook_indices):
            self.model.encoder.layers[idx].register_forward_hook(get_activation(f'l{i+1}'))
            
    def forward(self, x):
        self.features = {}
        _ = self.model(x)
        outputs = {}
        for k, v in self.features.items():
            # v shape: [B, N_tokens, C] (Torchvision ViT includes CLS token at index 0)
            # Remove CLS token and reshape
            B, N, C = v.shape
            # N - 1 is number of patches
            H = W = int(math.sqrt(N - 1))
            
            # v[:, 1:, :] -> Take only patch tokens
            # permute(0, 2, 1) -> [B, C, N_patches]
            # reshape -> [B, C, H, W]
            v_img = v[:, 1:, :].permute(0, 2, 1).reshape(B, C, H, W)
            outputs[k] = v_img
            
        return outputs

class SwinBackbone(nn.Module):
    """
    Swin Transformer Backbone using Torchvision (Official PyTorch Weights).
    Solves download issues in China by using official PyTorch CDN.
    """
    def __init__(self, model_name='swin_base_patch4_window7_224', pretrained=True):
        super(SwinBackbone, self).__init__()
        import torchvision.models as models
        
        # Select correct torchvision model
        # swin_b = Base (Embed dim 128), swin_t = Tiny (Embed dim 96)
        if 'base' in model_name:
            tv_model_func = models.swin_b
        elif 'tiny' in model_name:
            tv_model_func = models.swin_t
        else:
            tv_model_func = models.swin_b # Default fallback
            
        weights = 'DEFAULT' if pretrained else None
        
        try:
            print(f"Loading {model_name} from torchvision (download.pytorch.org)...")
            self.model = tv_model_func(weights=weights)
        except Exception as e:
            print(f"Error loading from torchvision: {e}")
            if pretrained:
                print("Fallback: Using random initialization.")
                self.model = tv_model_func(weights=None)
            else:
                raise e

        self.features = {}
        def get_activation(name):
            def hook(model, input, output):
                self.features[name] = output
            return hook

        # Register hooks for Swin in Torchvision
        # Structure: features[0]=Embed, 1=Stage1, 3=Stage2, 5=Stage3, 7=Stage4
        target_layers = [1, 3, 5, 7] 
        for i, layer_idx in enumerate(target_layers):
             if layer_idx < len(self.model.features):
                 self.model.features[layer_idx].register_forward_hook(get_activation(f'l{i+1}'))
        
    def forward(self, x):
        self.features = {}
        _ = self.model(x)
        
        outputs = {}
        for i in range(4):
            key = f'l{i+1}'
            if key in self.features:
                out = self.features[key]
                # Torchvision Swin stages return channels-last (B, H, W, C)
                # We simply permute to channels-first (B, C, H, W) for the following CNN blocks
                if out.ndim == 4:
                     out = out.permute(0, 3, 1, 2)
                outputs[key] = out
            else:
                outputs[key] = None

        return outputs

def build_backbone(backbone_type='resnet50', pretrained=True):
    if backbone_type == 'resnet50':
        return resnet50_backbone(pretrained=pretrained)
    elif backbone_type == 'vit16':
        return ViTBackbone(model_name='vit_base_patch16_224', pretrained=pretrained)
    elif backbone_type == 'vit32':
        return ViTBackbone(model_name='vit_base_patch32_224', pretrained=pretrained)
    elif backbone_type == 'swin_base':
        return SwinBackbone(model_name='swin_base_patch4_window7_224', pretrained=pretrained)
    elif backbone_type == 'swin_tiny':
        return SwinBackbone(model_name='swin_tiny_patch4_window7_224', pretrained=pretrained)
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")

        

if __name__ == '__main__':
    print("Testing Backbone Models...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    for b_type in ['resnet50', 'vit16', 'swin_tiny']:
        print(f"\n--- Testing {b_type} ---")
        try:
            model = build_backbone(backbone_type=b_type, pretrained=False).to(device)
            out = model(dummy_input)
            print(f"{b_type} output dictionary keys: {list(out.keys())}")
            for k, v in out.items():
                print(f"  {k} shape: {v.shape}")
        except Exception as e:
            print(f"Failed {b_type}: {e}")
