# ==============================================================================
# File: model_rotation_final_smooth.py
#
# Description:
# This version adds an Exponential Moving Average (EMA) to the loss display
# in the progress bar. This provides a much smoother, less "jumpy" view of the
# training trend, while still logging the true, raw loss for accurate analysis.
# ==============================================================================

import os
import json
import torch
import numpy as np
import cv2
import random
import datetime
import argparse
import logging
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch import optim, amp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModel, AutoImageProcessor
from torch.optim.lr_scheduler import CosineAnnealingLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------- 1. The Model Definition (Unchanged) ---------------
class DinoV2RotationTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        encoder_dim = self.encoder.config.hidden_size
        self.rot_embedder = nn.Linear(rot_dim, encoder_dim)
        self.projector = nn.Linear(encoder_dim * 3, embed_dim)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim),
            activation='gelu', batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        self.regressor = nn.Linear(embed_dim, 9)

    def forward(self, img1, img2, img3, input_abs):
        feat1 = self.encoder(img1).last_hidden_state[:, 0]
        feat2 = self.encoder(img2).last_hidden_state[:, 0]
        feat3 = self.encoder(img3).last_hidden_state[:, 0]
        abs_p1, abs_p2 = input_abs[:, :9], input_abs[:, 9:]
        fused_feat1 = feat1 + self.rot_embedder(abs_p1)
        fused_feat2 = feat2 + self.rot_embedder(abs_p2)
        combined = torch.cat([fused_feat1, fused_feat2, feat3], dim=1)
        projected = self.projector(combined).unsqueeze(1)
        transformer_out = self.transformer(projected).squeeze(1)
        return self.regressor(transformer_out)

# --------------- 2. Dataset and Preprocessing (Unchanged) ---------------
class RotationDataset(Dataset):
    def __init__(self, patient_dirs, pairs_per_patient=1000):
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.data_triplets = []
        for patient_dir in tqdm(patient_dirs, desc="Loading Data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')
            if not os.path.exists(info_path): continue
            with open(info_path) as f: points = json.load(f)['PointInfos']
            if len(points) < 3: continue
            for _ in range(pairs_per_patient): self.data_triplets.append((images_dir, *random.sample(points, 3)))
        all_quats = [self.get_quat(p) for _, p1, p2, p3 in self.data_triplets for p in [p1, p2, p3]]
        self.absolutes = np.array([self.quat_to_9d(q) for q in all_quats])
        self.mean, self.std = self.absolutes.mean(axis=0), self.absolutes.std(axis=0) + 1e-8
    def get_quat(self, pt_info): return np.array([pt_info['RotationQuaternion'][k] for k in 'xyzw'])
    def quat_to_9d(self, quat): return R.from_quat(quat).as_matrix().reshape(9)
    def __len__(self): return len(self.data_triplets)
    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]
        def load_img(pt): return self.image_processor(cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB), return_tensors="pt").pixel_values.squeeze(0)
        img1, img2, img3 = load_img(p1), load_img(p2), load_img(p3)
        abs_p1, abs_p2, abs_p3_label = [(self.quat_to_9d(self.get_quat(p)) - self.mean) / self.std for p in (p1, p2, p3)]
        input_abs = np.concatenate([abs_p1, abs_p2])
        return (img1, img2, img3, torch.tensor(input_abs).float(), torch.tensor(abs_p3_label).float())

# --------------- 3. Loss Function and Metrics (Unchanged) ---------------
def nine_d_to_rotmat(nine_d):
    b, _ = nine_d.shape
    mats = nine_d.view(b, 3, 3).to(torch.float32)
    U, _, Vt = torch.linalg.svd(mats)
    rotation_matrix = U @ Vt
    det = torch.det(rotation_matrix.to(torch.float32))
    Vt_clone = Vt.clone()
    Vt_clone[det < 0, 2, :] = -Vt_clone[det < 0, 2, :]
    return U @ Vt_clone

def angular_distance(pred_9d, target_9d):
    pred_rot, target_rot = nine_d_to_rotmat(pred_9d), nine_d_to_rotmat(target_9d)
    relative = torch.bmm(pred_rot.transpose(1, 2), target_rot)
    traces = torch.einsum('bii->b', relative)
    return torch.acos(torch.clamp((traces - 1) / 2, -1.0 + 1e-7, 1.0 - 1e-7)) * (180.0 / np.pi)

class RotationLoss(nn.Module):
    def __init__(self, angular_weight=0.1):
        super().__init__()
        self.angular_weight = angular_weight
    def forward(self, pred_9d, target_9d):
        pred_rot, target_rot = nine_d_to_rotmat(pred_9d), nine_d_to_rotmat(target_9d)
        chordal_loss = torch.mean(torch.norm(pred_rot - target_rot, p='fro', dim=(1,2)) ** 2)
        angular_loss_term = torch.mean(angular_distance(pred_9d, target_9d).to(device))
        return chordal_loss + self.angular_weight * angular_loss_term

# --------------- 4. Evaluation and Plotting (Unchanged from robust version) ---------------
def eval_epoch(model, loader, criterion, writer, step, mean, std, output_dir):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            img1, img2, img3, input_abs, abs_p3 = [b.to(device) for b in batch]
            output = model(img1, img2, img3, input_abs)
            loss = criterion(output, abs_p3)
            if torch.isnan(loss):
                logging.warning(f"NaN loss detected during validation at step {step}. Skipping batch.")
                continue
            total_loss += loss.item()
            all_preds.append(output.cpu().numpy()); all_labels.append(abs_p3.cpu().numpy())
    if not all_preds:
        logging.error("All validation batches resulted in NaN. Cannot generate report.")
        return
    avg_loss = total_loss / len(all_preds)
    preds_np, labels_np = np.vstack(all_preds), np.vstack(all_labels)
    ang_errors = angular_distance(torch.tensor(preds_np), torch.tensor(labels_np)).numpy()
    logging.info(f"Validation @ Step {step}: Avg Loss={avg_loss:.4f}, Mean Angular Err={np.mean(ang_errors):.2f}°")
    writer.add_scalar('Loss/val', avg_loss, step); writer.add_scalar('Metrics/mean_angular_error', np.mean(ang_errors), step)
    writer.add_histogram('Angular_Errors_val', ang_errors, step)
    pdf_path = os.path.join(output_dir, 'plots', f'results_step_{step}.pdf'); os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
    errors_denorm = angular_distance(torch.tensor(preds_np * std + mean), torch.tensor(labels_np * std + mean)).numpy()
    with PdfPages(pdf_path) as pdf: plt.figure(); plt.hist(errors_denorm, bins=50); plt.title(f'Angular Error @ Step {step} (Mean: {np.mean(errors_denorm):.2f})'); pdf.savefig(); plt.close()

# --------------- 5. Main Training Loop (MODIFIED) ---------------
def main(args):
    output_dir = f"./rotation_training_runs/{datetime.datetime.now().strftime('%Y%b%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')), logging.StreamHandler()])
    logging.info(f"Starting run. Outputting to {output_dir}\nArgs: {args}")

    train_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]
    if args.single_patient:
        logging.warning("--- RUNNING IN SINGLE PATIENT MODE FOR DEBUGGING ---")
        train_dirs, val_dirs = train_dirs[:1], val_dirs[:1]

    train_dataset = RotationDataset(train_dirs, args.pairs_per_patient)
    val_dataset = RotationDataset(val_dirs, args.val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DinoV2RotationTransformer().to(device)
    model.encoder.gradient_checkpointing_enable()

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': args.lr_head}
    ]
    optimizer = optim.AdamW(param_groups)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps)
    criterion = RotationLoss()
    scaler = amp.GradScaler()
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    train_iter = iter(train_loader)
    pbar = tqdm(range(args.total_steps), desc="Training")
    
    # --- SMOOTHING FIX ---
    # Initialize the exponential moving average for the loss
    ema_loss = 0.0
    smoothing_factor = 0.9

    for step in pbar:
        try: batch = next(train_iter)
        except StopIteration: train_iter = iter(train_loader); batch = next(train_iter)
        img1, img2, img3, input_abs, abs_p3 = [b.to(device) for b in batch]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        with amp.autocast(device_type='cuda'):
            outputs = model(img1, img2, img3, input_abs)
            loss = criterion(outputs, abs_p3)
        
        raw_loss = loss.item() # Get the raw loss value
        if not np.isfinite(raw_loss): # Check with numpy isfinite for compatibility
            logging.error(f"Non-finite loss detected at step {step}: {raw_loss}. Stopping training.")
            break

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        # --- SMOOTHING FIX ---
        # Update the EMA of the loss
        if step == 0:
            ema_loss = raw_loss
        else:
            ema_loss = (smoothing_factor * ema_loss) + ((1 - smoothing_factor) * raw_loss)
        
        # Display the smoothed loss in the progress bar
        pbar.set_postfix(smooth_loss=f"{ema_loss:.4f}")
        
        # Log the RAW loss to TensorBoard for accurate tracking
        writer.add_scalar('Loss/train', raw_loss, step)
        writer.add_scalar('LR/head', optimizer.param_groups[1]['lr'], step)
        writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], step)

        if step > 0 and (step % args.val_freq == 0 or step == args.total_steps - 1):
            eval_epoch(model, val_loader, criterion, writer, step, val_dataset.mean, val_dataset.std, output_dir)
            torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_step_{step}.pth'))

    pbar.close(); writer.close()
    logging.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DinoV2-based Rotation Transformer.")
    parser.add_argument('--data_root', type=str, default='../ct_data_rotation_only')
    parser.add_argument('--lr_head', type=float, default=5e-5, help='Peak learning rate for the new layers.')
    parser.add_argument('--lr_backbone', type=float, default=5e-6, help='Peak learning rate for the backbone.')
    parser.add_argument('--batch_size', type=int, default=64) # Default batch size is 16
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pairs_per_patient', type=int, default=1000)
    parser.add_argument('--val_pairs', type=int, default=200)
    parser.add_argument('--single_patient', action='store_true', help='Debug with one patient.')
    args = parser.parse_args()
    main(args)