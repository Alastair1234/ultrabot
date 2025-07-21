import os
import json
import torch
import numpy as np
import cv2
import random
import datetime
import argparse
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from matplotlib.backends.backend_pdf import PdfPages
import subprocess  # For querying nvidia-smi
import sklearn.neighbors as sknn  # For KNN sampling

import warnings  # For suppressing warnings
from torch.amp import GradScaler, autocast  # Updated for PyTorch 2.5+

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable CuDNN benchmarking and TF32 for performance (no quality loss on H200)
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True

HISTORY_LEN = 4  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA_DATA = 0.5 
LOW_RES_SIZE = (64, 64)  # Low-res for world model (physics prediction)

class CTSequenceDataset(Dataset):
    """
    Loads sequences of ultrasound images and their corresponding poses.
    Focuses on low-res only for world model training.
    Updated: Uses KNN-based sampling for pseudo-contiguous sequences based on pose similarity (for random/unordered poses).
    """
    def __init__(self, patient_dirs, sequences_per_patient=500, is_val=False, single_patient=False, use_knn=True):
        self.sequences = []
        self.is_val = is_val
        self.single_patient = single_patient
        self.use_knn = use_knn  # Flag: True for KNN (random poses), False for contiguous (sequential data)
        
        for patient_dir in tqdm(patient_dirs, desc=f"Loading {'Val' if is_val else 'Train'} Patient Data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')

            if not os.path.exists(info_path) or not os.path.isdir(images_dir):
                continue
            with open(info_path) as f:
                points = json.load(f)['PointInfos']
            if len(points) < HISTORY_LEN + 1:
                continue

            # Handle splitting and indices
            if self.single_patient:
                # Deterministic split for single patient to avoid leakage
                split_ratio = 0.8
                split_point = int(len(points) * split_ratio)
                if is_val:
                    available_indices = list(range(split_point, len(points)))
                else:
                    available_indices = list(range(0, split_point))
                num_sequences = min(sequences_per_patient, len(available_indices))
                if num_sequences < 1:
                    print(f"Warning: Not enough points for {'val' if is_val else 'train'} split in {patient_dir}. Skipping.")
                    continue
                selected_start_indices = random.sample(available_indices, num_sequences)
                # For KNN: Extract points and poses for this split only (prevents leakage)
                split_points = [points[idx] for idx in available_indices]
                split_poses = np.array([[
                    p['Position']['x'], p['Position']['y'], p['Position']['z'],
                    p['RotationQuaternion']['x'], p['RotationQuaternion']['y'], p['RotationQuaternion']['z'], p['RotationQuaternion']['w']
                ] for p in split_points])
                # Map back to original indices for sequence building
                split_to_original = {i: available_indices[i] for i in range(len(available_indices))}
            else:
                num_sequences = sequences_per_patient
                selected_start_indices = random.sample(range(len(points)), num_sequences)
                split_points = points  # Use all
                split_poses = np.array([[
                    p['Position']['x'], p['Position']['y'], p['Position']['z'],
                    p['RotationQuaternion']['x'], p['RotationQuaternion']['y'], p['RotationQuaternion']['z'], p['RotationQuaternion']['w']
                ] for p in split_points])
                split_to_original = {i: i for i in range(len(split_points))}

            # Build KNN model if enabled (fit on split_poses to avoid leakage)
            if self.use_knn:
                if len(split_poses) < HISTORY_LEN + 1:
                    continue
                knn = sknn.NearestNeighbors(n_neighbors=HISTORY_LEN + 2, metric='euclidean')  # +2 for safety (skip self)
                knn.fit(split_poses)

            for start_idx in selected_start_indices:
                if self.use_knn:
                    # Find nearest neighbors in the split
                    split_start_idx = available_indices.index(start_idx) if self.single_patient else start_idx
                    distances, indices = knn.kneighbors([split_poses[split_start_idx]], n_neighbors=HISTORY_LEN + 2)
                    # Build sequence indices (skip self if it's first)
                    seq_split_indices = indices[0][1:HISTORY_LEN + 2]  # Skip self, take next
                    if len(seq_split_indices) < HISTORY_LEN + 1:
                        continue
                    # Map back to original points indices
                    seq_original_indices = [split_to_original[int(idx)] for idx in seq_split_indices[:HISTORY_LEN + 1]]
                    sequence_points = [points[idx] for idx in seq_original_indices]
                else:
                    # Fallback: Contiguous sampling (assuming ordered data)
                    end_idx = start_idx + HISTORY_LEN + 1
                    if end_idx > len(points):
                        continue
                    sequence_points = points[start_idx:end_idx]
                
                self.sequences.append((images_dir, sequence_points))

    def _calc_delta_pose(self, p1, p2):
        pos1 = np.array([p1['Position']['x'], p1['Position']['y'], p1['Position']['z']])
        pos2 = np.array([p2['Position']['x'], p2['Position']['y'], p2['Position']['z']])
        q1 = np.array([p1['RotationQuaternion']['x'], p1['RotationQuaternion']['y'], p1['RotationQuaternion']['z'], p1['RotationQuaternion']['w']])
        q2 = np.array([p2['RotationQuaternion']['x'], p2['RotationQuaternion']['y'], p2['RotationQuaternion']['z'], p2['RotationQuaternion']['w']])
        pos_diff = pos2 - pos1
        if np.all(q1 == 0): q1[3] = 1.0
        if np.all(q2 == 0): q2[3] = 1.0
        quat_diff = (R.from_quat(q2) * R.from_quat(q1).inv()).as_quat()
        return np.concatenate([pos_diff, quat_diff])

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        images_dir, sequence_points = self.sequences[idx]
        context_imgs_low, context_deltas = [], []

        def load_img(pt):
            path = os.path.join(images_dir, pt['FileName'])
            img = cv2.imread(path)
            low_img = cv2.resize(img, LOW_RES_SIZE, interpolation=cv2.INTER_AREA)  # Low-res only
            low_img = (cv2.cvtColor(low_img, cv2.COLOR_BGR2RGB) / 255.0) * 2.0 - 1.0
            return torch.tensor(low_img).permute(2, 0, 1).float()  # To float (FP32); autocast handles FP16

        for i in range(HISTORY_LEN):
            low_img = load_img(sequence_points[i])
            delta = self._calc_delta_pose(sequence_points[i], sequence_points[i+1])
            context_imgs_low.append(low_img)
            context_deltas.append(torch.tensor(delta, dtype=torch.float))  # To float (FP32)

        target_low = load_img(sequence_points[HISTORY_LEN])
        return torch.stack(context_imgs_low), torch.stack(context_deltas), target_low

def get_karras_conditioners(sigmas):
    c_skip = SIGMA_DATA**2 / (sigmas**2 + SIGMA_DATA**2)
    c_out = sigmas * SIGMA_DATA / (sigmas**2 + SIGMA_DATA**2).sqrt()
    c_in = 1 / (sigmas**2 + SIGMA_DATA**2).sqrt()
    return c_skip, c_out, c_in

def train_epoch(model, loader, criterion, optimizer, scaler, accum_steps=8):
    model.train()
    total_loss = 0
    for batch_idx, (context_imgs_low, context_deltas, target_low) in enumerate(tqdm(loader, desc='Training', leave=False)):
        context_imgs_low, context_deltas, target_low = (
            context_imgs_low.to(DEVICE), context_deltas.to(DEVICE), target_low.to(DEVICE)
        )
        optimizer.zero_grad()  # Zero at start of accum cycle
        
        sigmas = (torch.randn(target_low.shape[0], device=DEVICE) * 1.2 - 1.2).exp() 
        noise = torch.randn_like(target_low)
        noisy_target_low = target_low + noise * sigmas.view(-1, 1, 1, 1)

        c_skip, c_out, c_in = get_karras_conditioners(sigmas)
        c_skip, c_out, c_in = [c.view(-1, 1, 1, 1) for c in (c_skip, c_out, c_in)]

        with autocast(device_type='cuda', dtype=torch.float16):  # Updated
            model_output = model(context_imgs_low, context_deltas, noisy_target_low * c_in, sigmas)
            target_for_loss = (target_low - c_skip * noisy_target_low) / c_out
            loss = criterion(model_output, target_for_loss) / accum_steps  # Normalize loss for accum
            
        scaler.scale(loss).backward()
        
        # Gradient accumulation
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps  # Adjust back for reporting
        
    torch.cuda.empty_cache()  # Clear after epoch
    peak_mem = torch.cuda.max_memory_allocated(DEVICE) / 1e9
    print(f"Peak memory usage: {peak_mem:.2f} GB")
    return total_loss / len(loader), peak_mem  # Return peak_mem for PDF

def denormalize_img(img_tensor):
    img = (img_tensor.clamp(-1, 1) + 1) / 2.0
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

# Function to get GPU stats from nvidia-smi
def get_gpu_stats():
    try:
        output = subprocess.check_output(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,power.draw,temperature.gpu', '--format=csv,noheader']).decode('utf-8').strip()
        util, mem_used, mem_total, power, temp = output.split(', ')
        return {
            'GPU Util': util.strip(),
            'Memory': f"{mem_used.strip()} / {mem_total.strip()}",
            'Power': power.strip(),
            'Temp': temp.strip()
        }
    except Exception as e:
        return {'Error': str(e)}

@torch.no_grad()
def sample_k_diffusion(model, context_imgs_low, context_deltas, num_steps=20, sigma_max=20.0, sigma_min=0.002, rho=7.0, sub_batch_size=1):
    model.eval()
    generated = []
    for i in range(0, context_imgs_low.shape[0], sub_batch_size):  # Process in sub-batches
        sub_imgs = context_imgs_low[i:i+sub_batch_size]
        sub_deltas = context_deltas[i:i+sub_batch_size]
        B, _, C, H, W = sub_imgs.shape
        
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        steps = torch.linspace(1, 0, num_steps, device=DEVICE)
        sigmas = (max_inv_rho + steps * (min_inv_rho - max_inv_rho)) ** rho
        sigmas = torch.cat([sigmas, torch.tensor([0.0], device=DEVICE)])

        x = torch.randn(B, C, H, W, device=DEVICE) * sigmas[0]

        for j in tqdm(range(num_steps), desc="Sampling", leave=False):
            sigma_i = sigmas[j].repeat(B)
            sigma_next = sigmas[j+1].repeat(B)
            
            c_skip, c_out, c_in = get_karras_conditioners(sigma_i)
            c_skip, c_out, c_in = [c.view(-1, 1, 1, 1) for c in (c_skip, c_out, c_in)]

            with autocast(device_type='cuda', dtype=torch.float16):  # Added: Ensure FP16 for Flash Attention
                model_output = model(sub_imgs, sub_deltas, x * c_in, sigma_i)
            denoised = c_skip * x + c_out * model_output

            d = (x - denoised) / sigma_i.view(-1, 1, 1, 1)
            dt = sigma_next.view(-1, 1, 1, 1) - sigma_i.view(-1, 1, 1, 1)
            x = x + d * dt
        
        generated.append(x)
    return torch.cat(generated)

@torch.no_grad()
def eval_and_plot(model, loader, epoch, run_timestamp, train_loss, peak_mem):  # Updated: Use run_timestamp for filenames
    model.eval()
    pdf_dir = './pdf_outputs'
    os.makedirs(pdf_dir, exist_ok=True)
    pdf_path = os.path.join(pdf_dir, f'generated_epoch_{epoch}_{run_timestamp}.pdf')
    
    try:
        # Get a batch and also compute deltas for visualization
        context_imgs_low, context_deltas, target_low = next(iter(loader))
        # Compute human-readable delta strings for PDF overlays
        delta_strings = []
        for i in range(context_imgs_low.shape[0]):  # Per batch item
            seq_deltas = []
            for j in range(HISTORY_LEN):
                delta = context_deltas[i, j].cpu().numpy()  # [pos_diff (3), quat_diff (4)]
                pos_str = f"ΔPos: {delta[0:3].round(2)}"
                quat_str = f"ΔQuat: {delta[3:7].round(2)}"
                seq_deltas.append(f"{pos_str}\n{quat_str}")
            delta_strings.append(seq_deltas)
    except StopIteration:
        print("Validation loader is empty. Cannot plot results.")
        return

    context_imgs_low, context_deltas = context_imgs_low.to(DEVICE), context_deltas.to(DEVICE)
    
    generated_low = sample_k_diffusion(model, context_imgs_low, context_deltas, sub_batch_size=1)  # Low sub-batch for memory

    with PdfPages(pdf_path) as pdf:
        for i in range(min(context_imgs_low.shape[0], 5)):
            fig, axes = plt.subplots(1, HISTORY_LEN + 2, figsize=(20, 4))
            fig.suptitle(f"Epoch {epoch} - Example {i+1} (Low-Res World Model)")
            
            for j in range(HISTORY_LEN):
                axes[j].imshow(denormalize_img(context_imgs_low[i, j].cpu()))  # Move to CPU for plotting
                axes[j].set_title(f"Context {j+1} (Low-Res)")
                axes[j].axis('off')
                
                # Add transition (delta) text "between" frames (overlay on right side of each context frame)
                if j < HISTORY_LEN:
                    axes[j].text(1.05, 0.5, delta_strings[i][j], transform=axes[j].transAxes, 
                                 fontsize=8, va='center', ha='left', bbox=dict(facecolor='white', alpha=0.5))

            axes[HISTORY_LEN].imshow(denormalize_img(target_low[i].cpu()))
            axes[HISTORY_LEN].set_title("Ground Truth (Low-Res)")
            axes[HISTORY_LEN].axis('off')

            axes[HISTORY_LEN + 1].imshow(denormalize_img(generated_low[i].cpu()))
            axes[HISTORY_LEN + 1].set_title("Generated (Low-Res)")
            axes[HISTORY_LEN + 1].axis('off')
            
            pdf.savefig(fig)
            plt.close(fig)
        
        # Add GPU stats page
        stats = get_gpu_stats()
        stats_text = f"Epoch {epoch} GPU Stats:\n" \
                     f"Train Loss: {train_loss:.4f}\n" \
                     f"Peak Memory: {peak_mem:.2f} GB\n" \
                     f"GPU Util: {stats.get('GPU Util', 'N/A')}\n" \
                     f"Memory Usage: {stats.get('Memory', 'N/A')}\n" \
                     f"Power Draw: {stats.get('Power', 'N/A')}\n" \
                     f"GPU Temp: {stats.get('Temp', 'N/A')}"
        
        fig_stats = plt.figure(figsize=(8, 6))
        plt.text(0.1, 0.5, stats_text, fontsize=12, va='center')
        plt.axis('off')
        plt.title(f"Epoch {epoch} Performance Stats")
        pdf.savefig(fig_stats)
        plt.close(fig_stats)
    
    torch.cuda.empty_cache()  # Clear after eval

# =================================================================================
#  --- Model Definition (Inlined from model_csgo_adapted.py) ---
# =================================================================================

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from flash_attn import flash_attn_func  # Correct import for efficient, exact attention (fixes OOM)
# NOTE: Ensure flash-attn is installed: pip install flash-attn

# =================================================================================
#  --- Building Blocks (Unchanged and Correct) ---
# =================================================================================

class GroupNorm(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        num_groups = max(1, in_channels // 32)
        self.norm = nn.GroupNorm(num_groups, in_channels, eps=1e-5)
    def forward(self, x: Tensor) -> Tensor: return self.norm(x)

class AdaGroupNorm(nn.Module):
    def __init__(self, in_channels: int, cond_channels: int):
        super().__init__()
        self.in_channels, self.num_groups = in_channels, max(1, in_channels // 32)
        self.linear = nn.Linear(cond_channels, in_channels * 2)
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        x_norm = F.group_norm(x, self.num_groups, eps=1e-5)
        scale, shift = self.linear(cond)[:, :, None, None].chunk(2, dim=1)
        return x_norm * (1 + scale) + shift

class SelfAttention2d(nn.Module):
    def __init__(self, in_channels: int, head_dim: int = 8):
        super().__init__()
        self.n_head = max(1, in_channels // head_dim)
        assert in_channels % self.n_head == 0
        self.norm = GroupNorm(in_channels)
        self.qkv_proj = nn.Conv2d(in_channels, in_channels * 3, 1)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        nn.init.zeros_(self.out_proj.weight), nn.init.zeros_(self.out_proj.bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv_proj(x_norm).view(n, self.n_head * 3, c // self.n_head, h * w).transpose(2, 3)
        q, k, v = qkv.chunk(3, dim=1)  # Each: (B, n_head, H*W, head_dim)

        # Reshape for Flash Attention: (B, H*W, n_head, head_dim) and ensure contiguous for backward pass
        q = q.permute(0, 2, 1, 3).contiguous()  # (B, H*W, n_head, head_dim)
        k = k.permute(0, 2, 1, 3).contiguous()
        v = v.permute(0, 2, 1, 3).contiguous()

        # Flash Attention (exact replacement: computes softmax(qk) @ v efficiently, no OOM)
        y = flash_attn_func(
            q, k, v,
            softmax_scale=1 / math.sqrt(k.size(-1)),  # Same scale as original
            causal=False  # No causal mask (assuming self-attn over flattened image; set True if needed)
        )

        # Reshape back to original format
        y = y.permute(0, 2, 1, 3).reshape(n, c, h, w)  # (B, n_head, H*W, head_dim) -> (B, C, H, W)
        return x + self.out_proj(y)

class FourierFeatures(nn.Module):
    def __init__(self, cond_channels: int):
        super().__init__(); assert cond_channels % 2 == 0
        self.register_buffer("weight", torch.randn(1, cond_channels // 2))
    def forward(self, input: Tensor) -> Tensor:
        f = 2 * math.pi * input.unsqueeze(1) @ self.weight
        return torch.cat([f.cos(), f.sin()], dim=-1)

class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, cond_channels: int, attn: bool):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.norm1, self.conv1 = AdaGroupNorm(in_channels, cond_channels), nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.norm2, self.conv2 = AdaGroupNorm(out_channels, cond_channels), nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.attn = SelfAttention2d(out_channels) if attn else nn.Identity()
        nn.init.zeros_(self.conv2.weight)
    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        r = self.proj(x)
        x = self.conv1(F.silu(self.norm1(x, cond)))
        x = self.conv2(F.silu(self.norm2(x, cond)))
        return self.attn(x + r)

class UNet(nn.Module):
    def __init__(self, cond_channels: int, depths: List[int], channels: List[int], attn_depths: List[int]):
        super().__init__()
        self.d_blocks, self.u_blocks = nn.ModuleList(), nn.ModuleList()
        self.downsamples, self.upsamples = nn.ModuleList(), nn.ModuleList()
        
        # -- ENCODER INITIALIZATION --
        for i, (d, c, a) in enumerate(zip(depths, channels, attn_depths)):
            in_c = channels[i-1] if i > 0 else c
            self.downsamples.append(nn.Conv2d(in_c, c, 3, 2, 1) if i > 0 else nn.Identity())
            self.d_blocks.append(nn.ModuleList([ResBlock(c, c, cond_channels, a) for _ in range(d)]))

        # -- BOTTLENECK --
        self.mid_block = ResBlock(channels[-1], channels[-1], cond_channels, attn=True)
        reversed_params = list(reversed(list(zip(depths, channels, attn_depths))))[:-1]
        for i, (d, c, a) in enumerate(reversed_params):
            in_c = channels[len(channels) - 1 - i]
            out_c = channels[len(channels) - 2 - i] if i < len(channels) - 1 else c
            self.upsamples.append(nn.ConvTranspose2d(in_c, out_c, 2, 2) if i < len(channels) - 1 else nn.Identity())
            self.u_blocks.append(nn.ModuleList(
                [ResBlock(out_c * 2, out_c, cond_channels, a)] + 
                [ResBlock(out_c, out_c, cond_channels, a) for _ in range(d - 1)]
            ))

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        skips = []
        # Encoder pass
        for i, (downsample, blocks) in enumerate(zip(self.downsamples, self.d_blocks)):
            x = downsample(x)
            for block in blocks:
                x = block(x, cond)
            skips.append(x)
        
        # Bottleneck pass
        x = self.mid_block(x, cond)

        # Decoder pass
        for i, (upsample, blocks) in enumerate(zip(self.upsamples, self.u_blocks)):
            x = upsample(x)
            skip = skips[len(skips) - 2 - i]
            x = torch.cat((x, skip), dim=1)
            for block in blocks:
                x = block(x, cond)
        return x

# =================================================================================
#  --- InnerModel and UltrasoundDenoiser Adapter (Unchanged) ---
# =================================================================================

class InnerModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.is_upsampler = cfg.is_upsampler
        self.noise_emb = FourierFeatures(cfg.cond_channels)
        self.noise_cond_emb = FourierFeatures(cfg.cond_channels)
        self.act_emb = None if self.is_upsampler else nn.Sequential(
            nn.Linear(cfg.num_actions, cfg.cond_channels),
        )
        self.cond_proj = nn.Sequential(
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
            nn.SiLU(),
            nn.Linear(cfg.cond_channels, cfg.cond_channels),
        )
        in_channels = (cfg.num_steps_conditioning + int(self.is_upsampler) + 1) * cfg.img_channels
        self.conv_in = nn.Conv2d(in_channels, cfg.channels[0], 3, 1, 1)
        self.unet = UNet(cfg.cond_channels, cfg.depths, cfg.channels, [bool(a) for a in cfg.attn_depths])
        self.norm_out = GroupNorm(cfg.channels[0])
        self.conv_out = nn.Conv2d(cfg.channels[0], cfg.img_channels, 3, 1, 1)
        nn.init.zeros_(self.conv_out.weight)

    def forward(self, noisy_next_obs: Tensor, c_noise: Tensor, c_noise_cond: Tensor, obs: Tensor, act: Optional[Tensor]) -> Tensor:
        act_emb = 0 if self.is_upsampler else self.act_emb(act)
        cond = self.cond_proj(self.noise_emb(c_noise) + self.noise_cond_emb(c_noise_cond) + act_emb)
        x = self.conv_in(torch.cat((obs, noisy_next_obs), dim=1))
        x = self.unet(x, cond)
        x = self.conv_out(F.silu(self.norm_out(x)))
        return x

class UltrasoundDenoiser(nn.Module):
    def __init__(self, history_len=4, pose_dim=7, **kwargs):
        super().__init__()
        self.pose_dim = pose_dim

        class InnerModelConfig:
            img_channels = 3
            num_steps_conditioning = history_len
            num_actions = history_len * pose_dim 
            is_upsampler = False
            cond_channels = kwargs.get("cond_channels", 4096)  # Upsized for richer conditioning
            depths = kwargs.get("depths", [3, 4, 4, 3])  # Upsized for deeper features
            channels = kwargs.get("channels", [320, 640, 1280, 1280])  # Upsized for more capacity
            attn_depths = kwargs.get("attn_depths", [False, False, True, True])  # Kept; attention in deeper layers

        self.cfg = InnerModelConfig()
        self.inner_model = InnerModel(self.cfg)
        self.noise_previous_obs = False

    def forward(self, context_imgs, context_deltas, noisy_target_img, sigmas):
        B, T, C, H, W = context_imgs.shape
        obs_cond = context_imgs.view(B, T * C, H, W)
        act_cond = context_deltas.view(B, T * self.pose_dim)
        c_noise_cond = sigmas
        c_noise_prev_obs = torch.zeros_like(sigmas)
        if self.noise_previous_obs:
            raise NotImplementedError("noise_previous_obs regularization not implemented.")

        return self.inner_model(
            noisy_next_obs=noisy_target_img,
            c_noise=c_noise_cond,
            c_noise_cond=c_noise_prev_obs,
            obs=obs_cond,
            act=act_cond
        )

# =================================================================================
#                                --- MAIN SCRIPT ---
# =================================================================================

def main(args):
    # Handle single patient mode
    if args.use_single_patient or args.single_patient is not None:
        if args.single_patient is None:
            # Auto-select the first available patient from train dir
            train_dir = os.path.join(args.data_root, 'train')
            if not os.path.exists(train_dir):
                raise ValueError(f"Train directory not found: {train_dir}")
            patient_dirs = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
            if not patient_dirs:
                raise ValueError(f"No patient directories found in {train_dir}")
            selected_patient = os.path.join(train_dir, patient_dirs[0])
            print(f"Single patient mode enabled (auto-selected): Using {selected_patient} for both train and val.")
        else:
            selected_patient = args.single_patient
            print(f"Single patient mode enabled: Using {selected_patient} for both train and val.")
        
        train_patient_dirs = [selected_patient]
        val_patient_dirs = [selected_patient]  # Eval on same patient
        args.val_sequences = 50  # Reduce val sequences to avoid too much overlap/heavy eval; adjust as needed
    else:
        train_patient_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train')) if os.path.isdir(os.path.join(args.data_root, 'train', d))]
        val_patient_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val')) if os.path.isdir(os.path.join(args.data_root, 'val', d))]
    
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    checkpoint_dir = './checkpoints'
    os.makedirs(checkpoint_dir, exist_ok=True)  # New: Separate dir for checkpoints
    print(f"DEVICE: {DEVICE}")
    print(f"Run Timestamp: {run_timestamp}")
    print(f"PDFs will be saved to: ./pdf_outputs/")
    print(f"Checkpoints will be saved to: {checkpoint_dir}/")

    train_dataset = CTSequenceDataset(train_patient_dirs, args.sequences_per_patient, single_patient=(args.use_single_patient or args.single_patient is not None), use_knn=True)
    val_dataset = CTSequenceDataset(val_patient_dirs, args.val_sequences, is_val=True, single_patient=(args.use_single_patient or args.single_patient is not None), use_knn=True)
    
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Check data paths.")
        return
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Check data paths.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)

    # World model only (low-res physics predictor)
    model = UltrasoundDenoiser(
        history_len=HISTORY_LEN,
        pose_dim=7,
        cond_channels=2048,  # Balanced for richness without excess VRAM
        channels=[256, 512, 1024, 1024],  # Powerful for dynamics; scale down if OOM (e.g., [128, 256, 512, 512])
        depths=[2, 3, 3, 2],  # Deeper in middle for feature extraction
        attn_depths=[False, True, True, True]  # Attention where it matters (deeper layers)
    ).to(DEVICE)  # No .half() - keep FP32 params; autocast handles FP16 compute
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    if DEVICE.type == 'cuda':
        scaler = GradScaler(device='cuda', init_scale=65536)  # Higher init_scale for FP16 stability
    else:
        scaler = GradScaler()
        print("Warning: Running on CPU - mixed precision disabled.")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss, peak_mem = train_epoch(model, train_loader, criterion, optimizer, scaler, accum_steps=8)
        print(f"Epoch {epoch} - Average Training Loss: {train_loss:.4f}")

        if (epoch % args.checkpoint_freq == 0 or epoch == args.epochs) and len(val_loader) > 0:
            print("Running validation and plotting...")
            eval_and_plot(model, val_loader, epoch, run_timestamp, train_loss, peak_mem)  # Updated: Pass timestamp
            
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}_{run_timestamp}.pth")  # New: Separate dir
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
        torch.cuda.empty_cache()  # Clear after each epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model for ultrasound sequence prediction.")
    parser.add_argument('--data_root', default='./ct_data_random_angle', help="Root directory of the CT data")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size. Lower if you run out of VRAM.")
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--sequences_per_patient', default=1000, type=int, help="How many sequences to sample from each patient for training")
    parser.add_argument('--val_sequences', default=100, type=int, help="How many sequences to sample from each patient for validation")
    parser.add_argument('--checkpoint_freq', default=5, type=int, help="Frequency of saving checkpoints and plotting validation results")
    parser.add_argument('--single_patient', default=None, help="Path to single patient directory (e.g., './ct_data_random_angle/train/patient_001'). If provided, train and val use only this directory.")
    parser.add_argument('--use_single_patient', action='store_true', help="Enable single patient mode. If --single_patient is not specified, automatically picks the first available patient from data_root/train.")
    args = parser.parse_args()
    
    main(args)
