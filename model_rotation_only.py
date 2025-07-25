import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.nn.attention import sdpa_kernel, SDPBackend

class DinoV2RotationTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=9):  # rot_dim=9 for single rotation (we'll use twice)
        super().__init__()
        self.encoder = timm.create_model('facebook/dinov2-base', pretrained=True, num_classes=0)
        encoder_dim = self.encoder.embed_dim
        
        # New: Embedders for rotations (project 9D rotation to match encoder_dim)
        self.rot_embedder = nn.Linear(rot_dim, encoder_dim)  # Used for each rotation
        
        # Projector now takes 3*encoder_dim (since rotations are fused early)
        self.projector = nn.Linear(encoder_dim * 3, embed_dim)
        
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.regressor = nn.Linear(embed_dim, 9)  # Output 9D absolute for p3

    def forward(self, img1, img2, img3, input_abs):
        # Split input_abs into abs_p1 (first 9) and abs_p2 (next 9)
        abs_p1 = input_abs[:, :9]  # (B, 9)
        abs_p2 = input_abs[:, 9:]  # (B, 9)
        
        # Extract image features
        feat1 = self.encoder(img1)  # (B, encoder_dim)
        feat2 = self.encoder(img2)
        feat3 = self.encoder(img3)
        
        # New: Early fusion - embed rotations and add to their image features
        embedded_p1 = self.rot_embedder(abs_p1)  # (B, encoder_dim)
        embedded_p2 = self.rot_embedder(abs_p2)
        fused_feat1 = feat1 + embedded_p1  # Add to img1 features
        fused_feat2 = feat2 + embedded_p2  # Add to img2 features
        # feat3 has no rotation (since we're predicting it), so leave as-is
        
        # Concatenate fused features
        combined = torch.cat([fused_feat1, fused_feat2, feat3], dim=1)  # (B, 3*encoder_dim)
        
        projected = self.projector(combined).unsqueeze(0)  # (1, B, embed_dim)
        transformer_out = self.transformer(projected).squeeze(0)
        return self.regressor(transformer_out)