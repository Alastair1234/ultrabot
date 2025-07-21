import torch
import torch.nn as nn
from transformers import AutoModel, AutoImageProcessor

class DinoV2PairTransformer(nn.Module):
    def __init__(self, 
                 output_dim=7, 
                 vision_model='facebook/webssl-dino2b-full2b-224',
                 hidden_dim=768, 
                 nhead=8, 
                 num_layers=2,
                 delta_input_dim=7):
        super().__init__()

        self.processor = AutoImageProcessor.from_pretrained(vision_model, do_rescale=False, use_fast=True)
        self.encoder = AutoModel.from_pretrained(vision_model)
        encoder_dim = self.encoder.config.hidden_size

        # Now we have embeddings of img pairs + delta input
        self.projector = nn.Linear(4 * encoder_dim + delta_input_dim, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, output_dim)
        )

    def forward(self, img1, img2, img3, delta_p1_p2):
        inputs_1 = self.processor(img1, return_tensors="pt").to(img1.device)
        inputs_2 = self.processor(img2, return_tensors="pt").to(img2.device)
        inputs_3 = self.processor(img3, return_tensors="pt").to(img3.device)

        embed_1 = self.encoder(**inputs_1).last_hidden_state[:, 0, :]
        embed_2 = self.encoder(**inputs_2).last_hidden_state[:, 0, :]
        embed_3 = self.encoder(**inputs_3).last_hidden_state[:, 0, :]

        pair_embed_1_2 = torch.cat([embed_1, embed_2], dim=1)
        pair_embed_2_3 = torch.cat([embed_2, embed_3], dim=1)

        # Explicitly include delta_p1_p2 embedding
        combined_embed = torch.cat([pair_embed_1_2, pair_embed_2_3, delta_p1_p2], dim=1)

        projected_embed = self.projector(combined_embed).unsqueeze(1)

        transformer_out = self.transformer_encoder(projected_embed).squeeze(1)

        return self.regressor(transformer_out)
