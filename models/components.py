import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class GlobalLocalGatedDecomposition(nn.Module):
    """
    Global-Local Gated Decomposition (GLGD).
    Separates feature representation into two complementary streams: 
    a global semantic stream and a local distortion stream using logic from the paper.
    """
    def __init__(self, in_channels, out_channels, feature_size=14, ksz=3):
        super().__init__()
        self.feature_size = feature_size
        
        # Main Projection
        self.projection = nn.Conv2d(in_channels, out_channels * 3, 1)
        self.activation = nn.GELU()
        
        # Linear transform
        self.linear_transform = nn.Conv2d(out_channels, out_channels, 1)
        
        # Gate Generators (bottleneck for sigmoid mask, multi-channel output)
        def create_gate_generator(input_channels):
            return nn.Sequential(
                nn.Conv2d(input_channels, 64, kernel_size=1),
                nn.GELU(),
                nn.Conv2d(64, 64, kernel_size=ksz, padding=1),
                nn.GELU(),
                nn.Conv2d(64, out_channels, kernel_size=1),
                nn.Sigmoid()
            )
        
        self.content_gate_generator = create_gate_generator(out_channels)
        self.distortion_gate_generator = create_gate_generator(out_channels * 2) 

        self.pool_c = nn.AdaptiveAvgPool2d(self.feature_size)
        self.pool_d = nn.AdaptiveAvgPool2d(self.feature_size)
     
    def forward(self, x):
        # Split pathways
        x_main, x_c, x_d = self.projection(x).chunk(3, dim=1)
        x_main_act = self.activation(x_main)
        
        # Semantic Stream Gating
        content_mask = self.content_gate_generator(x_c) 
        transformed = self.activation(self.linear_transform(x_main_act)) 
        c_feat = x_main_act + (content_mask * transformed) 
        
        # Distortion Stream Interaction
        x_d2 = torch.cat([x_d, c_feat], dim=1)
        
        # Distortion Stream Gating
        distortion_mask = self.distortion_gate_generator(x_d2)
        d_feat_gate = x_main_act + (distortion_mask * transformed) 
        
        c_pooled = self.pool_c(c_feat)
        d_pooled = self.pool_d(d_feat_gate)
        return c_pooled, d_pooled


class SelfAttentionFeatureEnrichment(nn.Module):
    """
    Standard Transformer Encoder for Self-Attention Feature Enrichment.
    """
    def __init__(self, d_model, nhead=8, num_layers=1, dropout=0.1, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, x, pos_encoding=None):
        B, C, H, W = x.shape
        x_seq = x.view(B, C, H*W).permute(0, 2, 1)  # [B, H*W, C]
        
        if pos_encoding is not None:
            if x_seq.size(1) != pos_encoding.size(1):
                pos_2d = pos_encoding.permute(0, 2, 1).view(1, C, int(math.sqrt(pos_encoding.size(1))), int(math.sqrt(pos_encoding.size(1))))
                pos_resized = F.interpolate(pos_2d, size=(H, W), mode='bilinear', align_corners=False)
                pos_encoding = pos_resized.view(1, C, H*W).permute(0, 2, 1)
            x_seq = x_seq + pos_encoding
        
        x_encoded = self.transformer_encoder(x_seq)  
        x_encoded = self.norm(x_encoded)
        x_out = x_encoded.permute(0, 2, 1).view(B, C, H, W)
        return x_out


class CrossScaleAttentionLayer(nn.TransformerDecoderLayer):
    """
    Specific Transformer Decoder Layer tailored to ONLY perform cross-attention.
    """
    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1, activation='gelu', layer_norm_eps=1e-5, batch_first=True, norm_first=True):
        super().__init__(d_model, nhead, dim_feedforward=dim_feedforward, dropout=dropout, activation=activation, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None, tgt_is_causal=None, memory_is_causal=None):
        x = tgt
        if self.norm_first:
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
            x = self.norm3(x + self._ff_block(x))
        return x


class CrossScaleAttention(nn.Module):
    """
    Standard Transformer Decoder tailored for Cross-Scale Interaction.
    Aligns with CSATD and CSABU operations detailed in the paper.
    """
    def __init__(self, d_model, nhead=8, num_layers=1, dropout=0.1, dim_feedforward=512):
        super().__init__()
        self.d_model = d_model
        
        decoder_layer = CrossScaleAttentionLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer=decoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.query_proj = nn.Linear(d_model, d_model)
        self.memory_proj = nn.Linear(d_model, d_model)
        
    def forward(self, query, memory, pos_encoding=None):
        B, C, H, W = query.shape
        B_m, C_m, H_m, W_m = memory.shape
        
        if (H_m, W_m) != (H, W):
            memory = F.interpolate(memory, size=(H, W), mode='bilinear', align_corners=False)
        
        query_seq = query.view(B, C, H*W).permute(0, 2, 1)  
        memory_seq = memory.view(B, C_m, H*W).permute(0, 2, 1)  
        
        query_seq = self.query_proj(query_seq)
        memory_seq = self.memory_proj(memory_seq)
        
        if pos_encoding is not None:
            if query_seq.size(1) != pos_encoding.size(1):
                pos_2d = pos_encoding.permute(0, 2, 1).view(1, C, int(math.sqrt(pos_encoding.size(1))), int(math.sqrt(pos_encoding.size(1))))
                pos_resized = F.interpolate(pos_2d, size=(H, W), mode='bilinear', align_corners=False)
                pos_encoding = pos_resized.view(1, C, H*W).permute(0, 2, 1)
            
            query_seq = query_seq + pos_encoding
            memory_seq = memory_seq + pos_encoding
        
        x_decoded = self.transformer_decoder(query_seq, memory_seq) 
        x_decoded = self.norm(x_decoded)
        x_out = x_decoded.permute(0, 2, 1).view(B, C, H, W)
        return x_out


class DualStreamAttentionModulation(nn.Module):
    """
    Dual-Stream Attention Modulation (DSAM)
    Replaces DPAtten. Implements GLGD across scales natively, enriches features via Self-Attention,
    and then conducts Cross-Scale Top-Down (CSATD) and Cross-Scale Bottom-Up (CSABU) refinements.
    """
    def __init__(self, output_channels=256, feature_size=14, **kwargs):
        super().__init__()
        self.output_channels = output_channels
        self.feature_size = feature_size
        
        outchannels = 256
        cr_num_head = 4
        sa_num_head = 4
        cr_num_layers = 2
        sa_num_layers = 2
        cr_dim_feedforward = outchannels * 4
        sa_dim_feedforward = outchannels * 4
        cr_dropout = 0.25
        sa_dropout = 0.2
        
        self.unified_pos_encoding_c = nn.Parameter(torch.randn(1, self.feature_size * self.feature_size, outchannels))
        init.trunc_normal_(self.unified_pos_encoding_c, std=0.02)
        
        if 'vit' in str(kwargs.get('backbone_type', 'resnet')).lower():
            in_channels_list = [768, 768, 768, 768]
        elif 'swin_tiny' in str(kwargs.get('backbone_type', 'resnet')).lower():
            in_channels_list = [96, 192, 384, 768]
        elif 'swin' in str(kwargs.get('backbone_type', 'resnet')).lower(): 
             in_channels_list = [128, 256, 512, 1024]
        else: 
            in_channels_list = [256, 512, 1024, 2048]

        self.glgd_modules = nn.ModuleList([
            GlobalLocalGatedDecomposition(in_channels, outchannels, feature_size) for in_channels in in_channels_list
        ])
        
        self.SA_S = nn.ModuleList([
            SelfAttentionFeatureEnrichment(outchannels, sa_num_head, sa_num_layers, sa_dropout, sa_dim_feedforward) for _ in range(4)
        ])
        
        self.SA_D = nn.ModuleList([
            SelfAttentionFeatureEnrichment(outchannels, sa_num_head, sa_num_layers, sa_dropout, sa_dim_feedforward) for _ in range(4)
        ])
        
        # Cross-Scale Top-Down (CSATD)
        self.csatd_L4_L3 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)
        self.csatd_L3_L2 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)
        self.csatd_L2_L1 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)
        
        # Cross-Scale Bottom-Up (CSABU)
        self.csabu_L1_L2 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)
        self.csabu_L2_L3 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)
        self.csabu_L3_L4 = CrossScaleAttention(outchannels, cr_num_head, cr_num_layers, cr_dropout, cr_dim_feedforward)

    def bidirectional_generator(self, multi_scale_features):
        semantic_features = []
        fine_grained_features = []
        for i in range(len(multi_scale_features)):
            semantic, fine = self.glgd_modules[i](multi_scale_features[i])
            semantic_features.append(self.SA_S[i](semantic, self.unified_pos_encoding_c))
            fine_grained_features.append(self.SA_D[i](fine, self.unified_pos_encoding_c))
        return semantic_features, fine_grained_features
    
    def forward(self, out1):
        l1, l2, l3, l4 = out1['l1'], out1['l2'], out1['l3'], out1['l4']
        multi_scale_features = [l1, l2, l3, l4]
        
        semantic_feats, fine_feats = self.bidirectional_generator(multi_scale_features)
        
        # Reverse semantic list explicitly as required for top-down processing order
        semantic_features = list(reversed(semantic_feats))
        fine_features = fine_feats

        def coarse_to_fine(semantic_features):
            # Top-down semantic (CSATD)
            sl4, sl3, sl2, sl1 = semantic_features
            step_td3 = self.csatd_L4_L3(sl4, sl3, None)
            step_td2 = self.csatd_L3_L2(step_td3, sl2, None)
            semantic_final = self.csatd_L2_L1(step_td2, sl1, None)
            return semantic_final
        
        def fine_to_coarse(fine_features):
            # Bottom-up distortion (CSABU)
            fl4, fl3, fl2, fl1 = fine_features
            step_bu1 = self.csabu_L1_L2(fl1, fl2, None)
            step_bu2 = self.csabu_L2_L3(step_bu1, fl3, None)
            fine_final = self.csabu_L3_L4(step_bu2, fl4, None)
            return fine_final
        
        coarse_to_fine_feat = coarse_to_fine(semantic_features)
        fine_to_coarse_feat = fine_to_coarse(fine_features)
        
        return {
            'semantic_feat': coarse_to_fine_feat, 
            'fine_feat': fine_to_coarse_feat
        }

if __name__ == '__main__':
    print("Testing DTIQA Core Components...")
    
    # Test GLGD
    print("\n--- Testing GlobalLocalGatedDecomposition ---")
    glgd = GlobalLocalGatedDecomposition(in_channels=256, out_channels=256, feature_size=14)
    dummy_feat = torch.randn(2, 256, 14, 14)
    out_c, out_d = glgd(dummy_feat)
    print(f"GLGD Semantic flow shape: {out_c.shape}")
    print(f"GLGD Fine-grained flow shape: {out_d.shape}")
    
    # Test DSAM
    print("\n--- Testing DualStreamAttentionModulation ---")
    dsam = DualStreamAttentionModulation(output_channels=256, feature_size=14, backbone_type='resnet50')
    dummy_out1 = {
        'l1': torch.randn(2, 256, 14, 14),
        'l2': torch.randn(2, 512, 14, 14),
        'l3': torch.randn(2, 1024, 14, 14),
        'l4': torch.randn(2, 2048, 14, 14)
    }
    dsam_out = dsam(dummy_out1)
    print(f"DSAM Semantic Feat shape: {dsam_out['semantic_feat'].shape}")
    print(f"DSAM Fine Feat shape: {dsam_out['fine_feat'].shape}")
