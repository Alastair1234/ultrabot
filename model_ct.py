import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.nn.attention import sdpa_kernel, SDPBackend  # For Flash Attention API

# --------------- 1. The Model Definition (Extended for Position) ---------------
class DinoV2RotationPositionTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=9, pos_dim=3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-base')
        encoder_dim = self.encoder.config.hidden_size
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Separate embedders for rotation and position
        self.rot_embedder = nn.Linear(rot_dim, encoder_dim)
        self.pos_embedder = nn.Linear(pos_dim, encoder_dim)
        nn.init.xavier_uniform_(self.rot_embedder.weight, gain=0.1)
        nn.init.zeros_(self.rot_embedder.bias)
        nn.init.xavier_uniform_(self.pos_embedder.weight, gain=0.1)
        nn.init.zeros_(self.pos_embedder.bias)
        
        # Add layer normalization
        self.layer_norm1 = nn.LayerNorm(encoder_dim)
        self.layer_norm2 = nn.LayerNorm(encoder_dim)
        
        self.projector = nn.Linear(encoder_dim * 3, embed_dim)
        nn.init.xavier_uniform_(self.projector.weight, gain=0.1)
        nn.init.zeros_(self.projector.bias)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim),
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        # Separate heads for rotation and position
        self.rotation_head = nn.Linear(embed_dim, 9)
        self.position_head = nn.Linear(embed_dim, 3)
        nn.init.xavier_uniform_(self.rotation_head.weight, gain=0.01)
        nn.init.zeros_(self.rotation_head.bias)
        nn.init.xavier_uniform_(self.position_head.weight, gain=0.01)
        nn.init.zeros_(self.position_head.bias)

    def forward(self, img1, img2, img3, input_abs):
        # Get features and apply layer norm
        feat1 = self.layer_norm1(self.encoder(img1).last_hidden_state[:, 0])
        feat2 = self.layer_norm1(self.encoder(img2).last_hidden_state[:, 0])
        feat3 = self.layer_norm1(self.encoder(img3).last_hidden_state[:, 0])
        
        # Split input into rotations and positions for points 1 and 2
        # input_abs shape: [batch, 24] = [rot1(9) + pos1(3) + rot2(9) + pos2(3)]
        abs_rot1, abs_pos1 = input_abs[:, :9], input_abs[:, 9:12]
        abs_rot2, abs_pos2 = input_abs[:, 12:21], input_abs[:, 21:24]
        
        # Clamp inputs to prevent extreme values
        abs_rot1 = torch.clamp(abs_rot1, -10, 10)
        abs_rot2 = torch.clamp(abs_rot2, -10, 10)
        abs_pos1 = torch.clamp(abs_pos1, -1000, 1000)  # Reasonable position bounds
        abs_pos2 = torch.clamp(abs_pos2, -1000, 1000)
        
        # Embed rotations and positions separately, then combine
        rot_emb1 = self.layer_norm2(self.rot_embedder(abs_rot1))
        pos_emb1 = self.layer_norm2(self.pos_embedder(abs_pos1))
        rot_emb2 = self.layer_norm2(self.rot_embedder(abs_rot2))
        pos_emb2 = self.layer_norm2(self.pos_embedder(abs_pos2))
        
        # Fuse features (additive fusion)
        fused_feat1 = feat1 + rot_emb1 + pos_emb1
        fused_feat2 = feat2 + rot_emb2 + pos_emb2
        
        # Apply dropout
        fused_feat1 = self.dropout(fused_feat1)
        fused_feat2 = self.dropout(fused_feat2)
        feat3 = self.dropout(feat3)
        
        # Combine and project
        combined = torch.cat([fused_feat1, fused_feat2, feat3], dim=1)
        projected = self.projector(combined).unsqueeze(1)
        
        # Apply transformer
        transformer_out = self.transformer(projected).squeeze(1)
        
        # Separate heads for rotation and position
        rotation_output = self.rotation_head(transformer_out)
        position_output = self.position_head(transformer_out)
        
        # Apply constraints
        # Normalize rotation output to prevent extreme values
        rot_norm = torch.norm(rotation_output, dim=1, keepdim=True)
        max_rot_norm = 3.0
        rotation_output = rotation_output * torch.clamp(max_rot_norm / (rot_norm + 1e-8), max=1.0)
        
        # Concatenate outputs: [rotation(9) + position(3)]
        output = torch.cat([rotation_output, position_output], dim=1)
        
        return output