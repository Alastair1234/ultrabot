import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor
from torch.nn.attention import sdpa_kernel, SDPBackend  # For Flash Attention API

class DinoV2PairTransformer(nn.Module):
    def __init__(self,
                 output_dim=6,  # 6 for 6D rotation representation
                 vision_model='facebook/webssl-dino300m-full2b-224',  # Pretrained DinoV2 model
                 hidden_dim=768,  # Hidden dimension for projector and transformer
                 nhead=8,  # Number of attention heads
                 num_layers=2,  # Number of transformer layers
                 delta_input_dim=6):  # Input dim for delta_p1_p2 (6D now)
        super().__init__()

        # Image processor for preprocessing (resizes to 224x224, no rescaling)
        self.processor = AutoImageProcessor.from_pretrained(
            vision_model, 
            do_rescale=False, 
            use_fast=True,  # Faster processing
            size={'height': 224, 'width': 224}  # Explicit size for consistency
        )

        # Load the pretrained DinoV2 encoder
        self.encoder = AutoModel.from_pretrained(vision_model)
        encoder_dim = self.encoder.config.hidden_size  # e.g., 768 for many Dino models

        # Projector: Combines pair embeddings (4 * encoder_dim) + delta_input_dim to hidden_dim
        self.projector = nn.Linear(4 * encoder_dim + delta_input_dim, hidden_dim)

        # Transformer encoder layer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            batch_first=True  # Batch dimension first
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Regressor head: From transformer output to final predictions
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, img1, img2, img3, delta_p1_p2):
        # Preprocess images (assumes img1/2/3 are tensors in [batch, C, H, W])
        inputs_1 = self.processor(img1, return_tensors="pt").to(img1.device)
        inputs_2 = self.processor(img2, return_tensors="pt").to(img2.device)
        inputs_3 = self.processor(img3, return_tensors="pt").to(img3.device)

        # Extract CLS embeddings (global features) from each image
        embed_1 = self.encoder(**inputs_1).last_hidden_state[:, 0, :]
        embed_2 = self.encoder(**inputs_2).last_hidden_state[:, 0, :]
        embed_3 = self.encoder(**inputs_3).last_hidden_state[:, 0, :]

        # Pair embeddings: Cat for (img1, img2) and (img2, img3)
        pair_embed_1_2 = torch.cat([embed_1, embed_2], dim=1)
        pair_embed_2_3 = torch.cat([embed_2, embed_3], dim=1)

        # Combine with delta_p1_p2
        combined_embed = torch.cat([pair_embed_1_2, pair_embed_2_3, delta_p1_p2], dim=1)

        # Project to hidden_dim and add sequence dim for transformer (treat as seq len=1)
        projected_embed = self.projector(combined_embed).unsqueeze(1)

        # Apply transformer with Flash Attention for efficiency
        with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            transformer_out = self.transformer_encoder(projected_embed).squeeze(1)

        # Regress to output (no normalization needed for 6D)
        outputs = self.regressor(transformer_out)
        return outputs
