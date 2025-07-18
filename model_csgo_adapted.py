import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List, Optional
from flash_attn import flash_attn_func  # Correct import for efficient, exact attention (fixes OOM)

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
            cond_channels = kwargs.get("cond_channels", 2048)
            depths = kwargs.get("depths", [2, 2, 2, 2])
            channels = kwargs.get("channels", [128, 256, 512, 1024])
            attn_depths = kwargs.get("attn_depths", [False, False, True, True])

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