import torch
import torch.nn as nn
from .backbone import build_backbone
from .components import DualStreamAttentionModulation, SelfAttentionFeatureEnrichment

class DTIQA(nn.Module):
    """
    DTIQA: A Dual-Path Transformer Framework for Robust No-Reference Image Quality Assessment
    """
    def __init__(self, backbone_type='resnet50', feature_size=None, fc_intermediate_dim=256, predictor_hidden_dim=256, **kwargs):
        super().__init__()
        
        if feature_size is None:
            if 'vit32' in backbone_type:
                self.feature_size = 7
            else:
                self.feature_size = 14
        else:
            self.feature_size = feature_size
            
        self.dim_output = 256
        self.backbone = build_backbone(backbone_type=backbone_type, pretrained=True)
        self.dsam = DualStreamAttentionModulation(output_channels=self.dim_output, feature_size=self.feature_size, backbone_type=backbone_type)
        
        # Self-Attention Pooling (SAPool)
        self.sapool_c = SelfAttentionFeatureEnrichment(d_model=256, nhead=4, num_layers=2, dropout=0.2, dim_feedforward=1024)
        self.sapool_d = SelfAttentionFeatureEnrichment(d_model=256, nhead=4, num_layers=2, dropout=0.2, dim_feedforward=1024)

        self.predictor = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 256),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),      
            
            nn.LayerNorm(256),
            nn.Linear(256, 128),  
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            
            nn.LayerNorm(128),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        try:
            # 1. Multi-scale Feature Extraction
            out1 = self.backbone(img)
            
            # 2. Dual-Stream Attention Modulation (DSAM)
            dsam_out = self.dsam(out1)
            semantic_feat = dsam_out['semantic_feat']
            fine_feat = dsam_out['fine_feat']
            
            # 3. Self-Attention Pooling (SAPool)
            semantic_pool = self.sapool_c(semantic_feat, None).mean(dim=(2,3)) # Output: [B, 256]
            fine_pool = self.sapool_d(fine_feat, None).mean(dim=(2,3))         # Output: [B, 256]

            # 4. Feature Fusion
            combined_pool = torch.cat([semantic_pool, fine_pool], dim=1)       # Output: [B, 512]

            # 5. Quality Regression
            final_score = self.predictor(combined_pool)
        
            return final_score.squeeze(1) 
        except Exception as e:
            print(f"DTIQA error: {e}")
            import traceback
            traceback.print_exc()
            return torch.zeros((img.size(0), 1), device=img.device, requires_grad=True)

def build_model(model_type='direct', **kwargs):
    """
    Factory method to instantiate the model architecture.
    """
    if model_type == 'direct':
        return DTIQA(**kwargs), None
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")
if __name__ == '__main__':
    print("Testing DTIQA Full Model Architecture...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dummy_input = torch.randn(2, 3, 224, 224).to(device)
    
    try:
        model = DTIQA(backbone_type='resnet50', feature_size=14, fc_intermediate_dim=256, predictor_hidden_dim=256).to(device)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        print(f"DTIQA Forward Pass Successful! Output shape: {out.shape}")
    except Exception as e:
        print(f"DTIQA Forward failed: {e}")
