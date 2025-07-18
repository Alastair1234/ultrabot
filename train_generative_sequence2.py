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

import warnings  # For suppressing warnings
from torch.amp import GradScaler, autocast  # Updated for PyTorch 2.5+

from model_csgo_adapted import UltrasoundDenoiser

# Suppress FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)

# Enable CuDNN benchmarking for performance
torch.backends.cudnn.benchmark = True

HISTORY_LEN = 4  
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SIGMA_DATA = 0.5 
IMAGE_SIZE = (256, 256)  # New: Resize images to this for memory savings (adjust lower if OOM)

class CTSequenceDataset(Dataset):
    """
    Loads sequences of ultrasound images and their corresponding poses.
    """
    def __init__(self, patient_dirs, sequences_per_patient=500, is_val=False):
        self.sequences = []
        self.is_val = is_val
        
        for patient_dir in tqdm(patient_dirs, desc=f"Loading {'Val' if is_val else 'Train'} Patient Data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')

            if not os.path.exists(info_path) or not os.path.isdir(images_dir):
                continue
            with open(info_path) as f:
                points = json.load(f)['PointInfos']

            if len(points) < HISTORY_LEN + 1:
                continue

            for _ in range(sequences_per_patient):
                start_idx = random.randint(0, len(points) - (HISTORY_LEN + 1))
                end_idx = start_idx + HISTORY_LEN + 1
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
        context_imgs, context_deltas = [], []

        def load_img(pt):
            path = os.path.join(images_dir, pt['FileName'])
            img = cv2.imread(path)
            img = cv2.resize(img, IMAGE_SIZE)  # New: Resize for memory savings
            img = (cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.0) * 2.0 - 1.0
            return torch.tensor(img).permute(2, 0, 1).half()  # New: To float16 for memory

        for i in range(HISTORY_LEN):
            img = load_img(sequence_points[i])
            delta = self._calc_delta_pose(sequence_points[i], sequence_points[i+1])
            context_imgs.append(img)
            context_deltas.append(torch.tensor(delta, dtype=torch.float16))  # New: To float16

        target_img = load_img(sequence_points[HISTORY_LEN])
        return torch.stack(context_imgs), torch.stack(context_deltas), target_img

def get_karras_conditioners(sigmas):
    c_skip = SIGMA_DATA**2 / (sigmas**2 + SIGMA_DATA**2)
    c_out = sigmas * SIGMA_DATA / (sigmas**2 + SIGMA_DATA**2).sqrt()
    c_in = 1 / (sigmas**2 + SIGMA_DATA**2).sqrt()
    return c_skip, c_out, c_in

def train_epoch(model, loader, criterion, optimizer, scaler, accum_steps=4):  # New: accum_steps param
    model.train()
    total_loss = 0
    for batch_idx, (context_imgs, context_deltas, target_img) in enumerate(tqdm(loader, desc='Training', leave=False)):
        context_imgs, context_deltas, target_img = (
            context_imgs.to(DEVICE), context_deltas.to(DEVICE), target_img.to(DEVICE)
        )
        optimizer.zero_grad()  # Zero at start of accum cycle
        
        sigmas = (torch.randn(target_img.shape[0], device=DEVICE) * 1.2 - 1.2).exp() 
        noise = torch.randn_like(target_img)
        noisy_target_img = target_img + noise * sigmas.view(-1, 1, 1, 1)

        c_skip, c_out, c_in = get_karras_conditioners(sigmas)
        c_skip, c_out, c_in = [c.view(-1, 1, 1, 1) for c in (c_skip, c_out, c_in)]

        with autocast(device_type='cuda', dtype=torch.float16):  # Updated
            model_output = model(context_imgs, context_deltas, noisy_target_img * c_in, sigmas)
            target_for_loss = (target_img - c_skip * noisy_target_img) / c_out
            loss = criterion(model_output, target_for_loss) / accum_steps  # New: Normalize loss for accum
            
        scaler.scale(loss).backward()
        
        # New: Gradient accumulation
        if (batch_idx + 1) % accum_steps == 0 or (batch_idx + 1) == len(loader):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item() * accum_steps  # Adjust back for reporting
        
    torch.cuda.empty_cache()  # New: Clear after epoch
    print(f"Peak memory usage: {torch.cuda.max_memory_allocated(DEVICE) / 1e9:.2f} GB")  # New: Track peak VRAM
    return total_loss / len(loader)

def denormalize_img(img_tensor):
    img = (img_tensor.clamp(-1, 1) + 1) / 2.0
    img = img.permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)

@torch.no_grad()
def sample_k_diffusion(model, context_imgs, context_deltas, num_steps=20, sigma_max=20.0, sigma_min=0.002, rho=7.0, sub_batch_size=1):  # New: sub_batch_size for memory
    model.eval()
    generated = []
    for i in range(0, context_imgs.shape[0], sub_batch_size):  # New: Process in sub-batches
        sub_imgs = context_imgs[i:i+sub_batch_size]
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

            model_output = model(sub_imgs, sub_deltas, x * c_in, sigma_i)
            denoised = c_skip * x + c_out * model_output

            d = (x - denoised) / sigma_i.view(-1, 1, 1, 1)
            dt = sigma_next.view(-1, 1, 1, 1) - sigma_i.view(-1, 1, 1, 1)
            x = x + d * dt
        
        generated.append(x)
    return torch.cat(generated)

@torch.no_grad()
def eval_and_plot(model, loader, epoch, save_dir):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'generated_epoch_{epoch}.pdf')
    
    try:
        context_imgs, context_deltas, target_img = next(iter(loader))
    except StopIteration:
        print("Validation loader is empty. Cannot plot results.")
        return

    context_imgs, context_deltas = context_imgs.to(DEVICE), context_deltas.to(DEVICE)
    
    generated_img = sample_k_diffusion(model, context_imgs, context_deltas, sub_batch_size=1)  # New: Low sub-batch for memory

    with PdfPages(pdf_path) as pdf:
        for i in range(min(context_imgs.shape[0], 5)):
            fig, axes = plt.subplots(1, HISTORY_LEN + 2, figsize=(20, 4))
            fig.suptitle(f"Epoch {epoch} - Example {i+1}")
            
            for j in range(HISTORY_LEN):
                axes[j].imshow(denormalize_img(context_imgs[i, j]))
                axes[j].set_title(f"Context {j+1}")
                axes[j].axis('off')
            
            axes[HISTORY_LEN].imshow(denormalize_img(target_img[i]))
            axes[HISTORY_LEN].set_title("Ground Truth")
            axes[HISTORY_LEN].axis('off')

            axes[HISTORY_LEN + 1].imshow(denormalize_img(generated_img[i]))
            axes[HISTORY_LEN + 1].set_title("Generated")
            axes[HISTORY_LEN + 1].axis('off')
            
            pdf.savefig(fig)
            plt.close(fig)
    torch.cuda.empty_cache()  # New: Clear after eval

# =================================================================================
#                                --- MAIN SCRIPT ---
# =================================================================================

def main(args):
    train_patient_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_patient_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]
    
    output_dir = f"./training_runs_sequence/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    print(f"DEVICE: {DEVICE}")
    print(f"Outputting to: {output_dir}")

    train_dataset = CTSequenceDataset(train_patient_dirs, args.sequences_per_patient)
    val_dataset = CTSequenceDataset(val_patient_dirs, args.val_sequences, is_val=True)
    
    if len(train_dataset) == 0:
        print("Error: Training dataset is empty. Check data paths.")
        return
    if len(val_dataset) == 0:
        print("Warning: Validation dataset is empty. Check data paths.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)  # New: Lower workers, no persistent
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=False, persistent_workers=False)

    model = UltrasoundDenoiser(
        history_len=HISTORY_LEN,
        pose_dim=7,
        cond_channels=1024,  # Small config for memory (uncomment below for large if testing)
        channels=[64, 128, 256, 512],
        depths=[1, 1, 2, 2],
        attn_depths=[False, False, True, False]
        # For large (high-VRAM only): 
        # cond_channels=2048,
        # channels=[128, 256, 512, 1024],
        # depths=[2, 2, 2, 2],
        # attn_depths=[False, False, True, True]
    ).to(DEVICE).half()  # New: Model to float16 for memory
    
    print(f"Model has {sum(p.numel() for p in model.parameters())/1e6:.2f}M parameters")

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.MSELoss()
    
    if DEVICE.type == 'cuda':
        scaler = GradScaler(device='cuda')  # Updated
    else:
        scaler = GradScaler()
        print("Warning: Running on CPU - mixed precision disabled.")

    for epoch in range(1, args.epochs + 1):
        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler, accum_steps=4)  # New: Pass accum_steps
        print(f"Epoch {epoch} - Average Training Loss: {train_loss:.4f}")

        if (epoch % args.checkpoint_freq == 0 or epoch == args.epochs) and len(val_loader) > 0:
            print("Running validation and plotting...")
            eval_and_plot(model, val_loader, epoch, output_dir)
            
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
            print(f"Checkpoint saved for epoch {epoch}.")
        torch.cuda.empty_cache()  # New: Clear after each epoch

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a diffusion model for ultrasound sequence prediction.")
    parser.add_argument('--data_root', default='./ct_data_random_angle', help="Root directory of the CT data")
    parser.add_argument('--epochs', default=100, type=int)
    parser.add_argument('--batch_size', default=1, type=int, help="Batch size. Lower if you run out of VRAM.")  # New: Default to 1
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--sequences_per_patient', default=1000, type=int, help="How many sequences to sample from each patient for training")
    parser.add_argument('--val_sequences', default=100, type=int, help="How many sequences to sample from each patient for validation")
    parser.add_argument('--checkpoint_freq', default=5, type=int, help="Frequency of saving checkpoints and plotting validation results")
    args = parser.parse_args()
    
    main(args)