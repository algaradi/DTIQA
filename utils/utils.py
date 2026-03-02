"""
Utility functions for DTIQA model visualization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import os


def preprocess_image(img_path, size=(224, 224)):
    """Preprocess image for model input"""
    transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img.resize(size))
    img_tensor = transform(img).unsqueeze(0)
    
    return img_tensor, img_np


def denormalize_image(tensor):
    """Denormalize image tensor for display"""
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    
    if tensor.is_cuda:
        mean = mean.cuda()
        std = std.cuda()
    
    denormalized = tensor * std + mean
    denormalized = torch.clamp(denormalized, 0, 1)
    
    if len(denormalized.shape) == 4:
        denormalized = denormalized[0]
    denormalized = denormalized.cpu().numpy()
    denormalized = np.transpose(denormalized, (1, 2, 0))
    
    return denormalized


def overlay_heatmap(img, heatmap, alpha=0.6, colormap=cv2.COLORMAP_JET):
    """Overlay heatmap on image"""
    # Ensure image is in correct format
    if img.max() > 1:
        img = img / 255.0
    
    # Resize heatmap
    if heatmap.shape != img.shape[:2]:
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    
    # Normalize heatmap
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Apply color mapping
    heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), colormap)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB) / 255.0
    
    # Overlay
    output = alpha * heatmap_colored + (1 - alpha) * img
    output = np.clip(output, 0, 1)
    
    return output


def compute_gradcam(activations, gradients):
    """Compute GradCAM"""
    # Global average pooling of gradients for weights
    weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
    
    # Weighted combination
    cam = torch.sum(weights * activations, dim=1, keepdim=True)
    
    # ReLU activation
    cam = F.relu(cam)
    
    # Normalization
    cam_min = cam.min()
    cam_max = cam.max()
    cam = (cam - cam_min) / (cam_max - cam_min + 1e-8)
    
    return cam[0, 0].cpu().numpy()


def save_feature_map(feature_map, save_path, title="Feature Map", cmap='viridis'):
    """Save single feature map"""
    plt.figure(figsize=(8, 6))
    plt.imshow(feature_map, cmap=cmap)
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_figure(images, titles, save_path, figsize=(15, 5)):
    """Create comparison figure"""
    n_images = len(images)
    fig, axes = plt.subplots(1, n_images, figsize=figsize)
    
    if n_images == 1:
        axes = [axes]
    
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def show_cam_on_image(img, mask, use_rgb=True, colormap=cv2.COLORMAP_JET, alpha=0.4):
    """Overlay CAM heatmap on image"""
    try:
        # Validate inputs
        if not isinstance(mask, np.ndarray) or mask.size == 0:
            print("Warning: Invalid mask provided to show_cam_on_image")
            # Return original image if mask is invalid
            return np.uint8(img * 255) if np.max(img) <= 1 else img.astype(np.uint8)
        
        # Ensure mask is 2D
        if len(mask.shape) > 2:
            mask = np.mean(mask, axis=0)
        
        # Ensure mask has valid values
        mask = np.nan_to_num(mask)  # Replace NaN with 0
        mask = np.clip(mask, 0, 1)  # Clip values to [0,1] range
        
        # Resize mask to image size
        if mask.shape != img.shape[:2]:
            try:
                mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            except Exception as e:
                print(f"Error resizing mask: {e}")
                # Return original image if resize fails
                return np.uint8(img * 255) if np.max(img) <= 1 else img.astype(np.uint8)
        
        # Create heatmap
        try:
            heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
            if use_rgb:
                heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap = np.float32(heatmap) / 255
        except Exception as e:
            print(f"Error creating heatmap: {e}")
            # Return original image if heatmap creation fails
            return np.uint8(img * 255) if np.max(img) <= 1 else img.astype(np.uint8)
        
        # Ensure input image is in correct range
        if np.max(img) > 1:
            img = img.astype(np.float32) / 255
        
        # Overlay image
        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
            img = np.repeat(img, 3, axis=2)
        
        cam = alpha * heatmap + (1 - alpha) * img
        cam = cam / np.max(cam) if np.max(cam) > 0 else cam
        return np.uint8(255 * cam)
    
    except Exception as e:
        print(f"Error in show_cam_on_image: {e}")
        # Return original image if any error occurs
        return np.uint8(img * 255) if np.max(img) <= 1 else img.astype(np.uint8)


def visualize_attention_weights(weights, save_path, title="Attention Weights"):
    """Visualize attention weights"""
    if isinstance(weights, torch.Tensor):
        weights = weights.detach().cpu().numpy()
    
    # Handle different weight shapes
    if len(weights.shape) == 4:  # [B, C, H, W]
        weights = weights[0]  # Take first batch
        if weights.shape[0] > 1:
            weights = np.mean(weights, axis=0)  # Average all channels
        else:
            weights = weights[0]
    elif len(weights.shape) == 3:  # [B, H, W] or [C, H, W]
        weights = weights[0] if weights.shape[0] == 1 else np.mean(weights, axis=0)
    
    # Normalize weights
    weights = (weights - weights.min()) / (weights.max() - weights.min() + 1e-8)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(weights, cmap='hot', interpolation='nearest')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_comparison_visualization(original_img, cam_maps, titles, save_path):
    """Create comparison visualization"""
    num_maps = len(cam_maps)
    if num_maps == 0:
        print(f"No CAM maps to visualize, skipping {save_path}")
        return
        
    fig, axes = plt.subplots(1, num_maps + 1, figsize=(4 * (num_maps + 1), 4))
    
    # Ensure axes is array form (when only 2 images, axes is not an array)
    if num_maps + 1 == 1:
        axes = [axes]
    elif not isinstance(axes, np.ndarray):
        axes = [axes]
        
    # Show original image
    axes[0].imshow(original_img)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # Show each CAM
    for i, (cam, title) in enumerate(zip(cam_maps, titles)):
        try:
            # Validate the cam
            if not isinstance(cam, np.ndarray):
                print(f"Warning: CAM for {title} is not a numpy array, attempting to convert")
                try:
                    cam = np.array(cam)
                except:
                    raise ValueError(f"Cannot convert CAM for {title} to numpy array")
            
            if cam.size == 0:
                raise ValueError(f"CAM for {title} is empty")
                
            # Handle different dimensions
            if len(cam.shape) > 2:
                cam = np.mean(cam, axis=0)
            
            # Check for NaN or Inf values
            if np.isnan(cam).any() or np.isinf(cam).any():
                print(f"Warning: CAM for {title} contains NaN or Inf values, replacing with zeros")
                cam = np.nan_to_num(cam)
            
            # Normalize if needed
            if cam.min() != cam.max():
                cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
            
            # Resize
            if cam.shape != original_img.shape[:2]:
                try:
                    cam = cv2.resize(cam, (original_img.shape[1], original_img.shape[0]))
                except Exception as resize_error:
                    print(f"Error resizing CAM for {title}: {resize_error}")
                    # Create a default heatmap
                    cam = np.ones((original_img.shape[0], original_img.shape[1])) * 0.5
            
            # Overlay display
            overlay = show_cam_on_image(original_img, cam, use_rgb=True)
            axes[i + 1].imshow(overlay)
            axes[i + 1].set_title(title)
            axes[i + 1].axis('off')
        except Exception as e:
            print(f"Error visualizing {title}: {e}")
            # Show original image as fallback
            axes[i + 1].imshow(original_img)
            axes[i + 1].set_title(f'{title} (Error)')
            axes[i + 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_feature_statistics(features_dict, save_path):
    """Save feature statistics"""
    stats = {}
    for name, feature in features_dict.items():
        try:
            if isinstance(feature, torch.Tensor):
                feature = feature.detach().cpu().numpy()
            elif not isinstance(feature, np.ndarray):
                # If not tensor or numpy array, try to convert
                try:
                    feature = np.array(feature)
                except:
                    stats[name] = {'error': f'Cannot convert to numpy array, type: {type(feature)}'}
                    continue
            
            stats[name] = {
                'shape': feature.shape,
                'mean': np.mean(feature),
                'std': np.std(feature),
                'min': np.min(feature),
                'max': np.max(feature)
            }
        except Exception as e:
            stats[name] = {'error': f'Failed to compute stats: {str(e)}'}
            continue
    
    with open(save_path, 'w') as f:
        f.write("Feature Statistics (DTIQA Architecture):\n")
        f.write("=" * 50 + "\n")
        f.write("Architecture: ResNet50 + GlobalLocalRouter + Bidirectional Transformer\n")
        f.write("=" * 50 + "\n")
        for name, stat in stats.items():
            f.write(f"\n{name}:\n")
            if 'error' in stat:
                f.write(f"  Error: {stat['error']}\n")
            else:
                f.write(f"  Shape: {stat['shape']}\n")
                f.write(f"  Mean: {stat['mean']:.6f}\n")
                f.write(f"  Std: {stat['std']:.6f}\n")
                f.write(f"  Min: {stat['min']:.6f}\n")
                f.write(f"  Max: {stat['max']:.6f}\n")


class AdvancedGradCAM:
    """Enhanced GradCAM for DTIQA model architecture"""
    def __init__(self, model, use_cuda=True):
        self.model = model.eval()
        self.use_cuda = use_cuda
        if self.use_cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
        
        # Store activations and gradients
        self.activations = {}
        self.gradients = {}
    
    def get_gradcam(self, input_tensor, target_layer_name):
        """Get GradCAM for specified layer, optimized for DTIQA model"""
        self.model.zero_grad()
        self.activations.clear()
        self.gradients.clear()
        
        # Get target module
        target_module = self._get_target_module(target_layer_name)
        if target_module is None:
            print(f"Target module {target_layer_name} not found")
            return None
        
        # Register activation hook
        def forward_hook(module, input, output):
            if isinstance(output, dict):
                # For dictionary output, select first value
                self.activations[target_layer_name] = list(output.values())[0].detach()
            elif isinstance(output, tuple):
                # For tuple output, select first element
                self.activations[target_layer_name] = output[0].detach()
            else:
                self.activations[target_layer_name] = output.detach()
        
        # Register gradient hook - using full_backward_hook
        def backward_hook(module, grad_input, grad_output):
            if isinstance(grad_output, tuple) and len(grad_output) > 0:
                if isinstance(grad_output[0], torch.Tensor):
                    self.gradients[target_layer_name] = grad_output[0].detach()
        
        forward_handle = target_module.register_forward_hook(forward_hook)
        backward_handle = target_module.register_full_backward_hook(backward_hook)
        
        try:
            # Forward pass
            input_tensor = input_tensor.clone().detach().requires_grad_(True)
            output = self.model(input_tensor)
            
            # Calculate loss
            if isinstance(output, torch.Tensor):
                if output.numel() == 1:
                    loss = output
                else:
                    loss = output.mean()
            else:
                loss = torch.tensor(output, requires_grad=True).mean()
            
            # Backward pass
            loss.backward(retain_graph=True)
            
            # Get activations and gradients
            if target_layer_name in self.activations and target_layer_name in self.gradients:
                activations = self.activations[target_layer_name]
                gradients = self.gradients[target_layer_name]
                
                # Calculate GradCAM
                return self._compute_gradcam(activations, gradients)
            else:
                print(f"Failed to capture activations or gradients for {target_layer_name}")
                return self._compute_feature_importance_fallback(target_layer_name, input_tensor)
                
        except Exception as e:
            print(f"Error computing GradCAM for {target_layer_name}: {e}")
            return self._compute_feature_importance_fallback(target_layer_name, input_tensor)
        finally:
            forward_handle.remove()
            backward_handle.remove()
    
    def _get_target_module(self, target_layer_name):
        """Get target module for DTIQA architecture"""
        # Define module mapping for updated DTIQA architecture
        module_mapping = {
            'backbone': self.model.backbone,
            'DPAtten': self.model.DPAtten,
            'atten_c_pool': self.model.atten_c_pool,
            'predictor': self.model.predictor,
        }
        
        # ResNet backbone layers
        try:
            module_mapping.update({
                'backbone_layer1': self.model.backbone.layer1,
                'backbone_layer2': self.model.backbone.layer2,
                'backbone_layer3': self.model.backbone.layer3,
                'backbone_layer4': self.model.backbone.layer4,
            })
        except:
            pass
        
        # DPAtten components
        try:
            # DRM modules
            for i in range(4):
                module_mapping[f'drm_{i}'] = self.model.DPAtten.drm_modules[i]
                
            # Self-attention encoders
            for i in range(4):
                module_mapping[f'SA_S_{i}'] = self.model.DPAtten.SA_S[i]
                module_mapping[f'SA_D_{i}'] = self.model.DPAtten.SA_D[i]
            
            # Cross-scale decoders
            module_mapping['cross_topdown_L4_L3'] = self.model.DPAtten.cross_topdown_L4_L3
            module_mapping['cross_topdown_L3_L2'] = self.model.DPAtten.cross_topdown_L3_L2
            module_mapping['cross_topdown_L2_L1'] = self.model.DPAtten.cross_topdown_L2_L1
            module_mapping['cross_bottomup_L1_L2'] = self.model.DPAtten.cross_bottomup_L1_L2
            module_mapping['cross_bottomup_L2_L3'] = self.model.DPAtten.cross_bottomup_L2_L3
            module_mapping['cross_bottomup_L3_L4'] = self.model.DPAtten.cross_bottomup_L3_L4
            
            # These attributes don't exist in the new architecture
            # module_mapping['fine'] = self.model.DPAtten.fine
            # module_mapping['semantic'] = self.model.DPAtten.semantic
        except:
            pass
        
        return module_mapping.get(target_layer_name)
    
    def _compute_gradcam(self, activations, gradients):
        """Compute GradCAM"""
        # Ensure tensor is on CPU and detached
        if activations.is_cuda:
            activations = activations.cpu()
        if gradients.is_cuda:
            gradients = gradients.cpu()
        
        # Ensure no gradients required
        activations = activations.detach()
        gradients = gradients.detach()
        
        # Process 4D tensor [B, C, H, W]
        if len(activations.shape) == 4:
            # Calculate gradient weights (global average pooling)
            weights = torch.mean(gradients, dim=(2, 3), keepdim=True)  # [B, C, 1, 1]
            
            # Calculate CAM
            cam = torch.sum(weights * activations, dim=1, keepdim=True)  # [B, 1, H, W]
            cam = F.relu(cam)  # Keep only positive values
            cam = cam.squeeze().numpy()  # Convert to numpy
            
        # Process 3D tensor [C, H, W]
        elif len(activations.shape) == 3:
            weights = torch.mean(gradients, dim=(1, 2), keepdim=True)
            cam = torch.sum(weights * activations, dim=0)
            cam = F.relu(cam)
            cam = cam.numpy()
            
        # Process 2D tensor [H, W]
        elif len(activations.shape) == 2:
            cam = F.relu(activations * gradients)
            cam = cam.numpy()
            
        # Other cases
        else:
            # Safely get scalar value
            result = torch.mean(F.relu(activations * gradients))
            if hasattr(result, 'item'):
                cam_value = result.item()
            else:
                cam_value = float(result)
            cam = np.array([[cam_value]])
            
        return self._normalize_cam(cam)
    
    def _compute_feature_importance_fallback(self, target_layer_name, input_tensor):
        """Fallback when GradCAM fails"""
        print(f"Using fallback feature importance for {target_layer_name}")
        
        target_module = self._get_target_module(target_layer_name)
        if target_module is None:
            return np.ones((14, 14)) * 0.5  # Return default map
        
        # Special handling for global_analyzer components which may not have spatial dimensions
        if 'global_analyzer' in target_layer_name:
            return np.ones((14, 14)) * 0.5  # Return default map for global_analyzer
        
        # Simple activation strength as importance metric
        activations = {}
        def hook_fn(module, input, output):
            if isinstance(output, dict):
                activations['output'] = list(output.values())[0].detach()
            elif isinstance(output, tuple):
                activations['output'] = output[0].detach()
            else:
                activations['output'] = output.detach()
        
        hook = target_module.register_forward_hook(hook_fn)
        
        try:
            with torch.no_grad():
                _ = self.model(input_tensor)
            
            if 'output' in activations:
                output = activations['output'].cpu()
                if len(output.shape) == 4:  # [B, C, H, W]
                    cam = torch.mean(output[0], dim=0).numpy()
                elif len(output.shape) == 3:  # [C, H, W]
                    cam = torch.mean(output, dim=0).numpy()
                elif len(output.shape) == 2:  # [H, W]
                    cam = output.numpy()
                else:
                    # For outputs without spatial dimensions (e.g., global_analyzer)
                    # Generate a default heatmap
                    cam = np.ones((14, 14)) * 0.5
            else:
                cam = np.ones((14, 14)) * 0.5
                
        except Exception as e:
            print(f"Fallback also failed for {target_layer_name}: {e}")
            cam = np.ones((14, 14)) * 0.5
        finally:
            hook.remove()
            
        return self._normalize_cam(cam)
    
    def _normalize_cam(self, cam):
        """Normalize CAM to [0,1] range"""
        if isinstance(cam, (int, float)):
            return np.array([[cam]])
        
        try:
            cam = np.array(cam)
            if cam.size == 0 or not isinstance(cam, np.ndarray):
                # Handle empty or non-array inputs
                return np.ones((14, 14)) * 0.5
                
            cam = cam - np.min(cam)
            cam_max = np.max(cam)
            if cam_max > 0:
                cam = cam / cam_max
            return cam
        except Exception as e:
            print(f"Error normalizing CAM: {e}")
            return np.ones((14, 14)) * 0.5

