# ==============================================================================
# File: model_rotation_position_stable.py
#
# Description:
# Extended stable rotation model to also handle positions
# Predicting 9D rotation + 3D position (12D total) using absolute values
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

# --------------- 2. Dataset and Preprocessing (Extended for Position) ---------------
class RotationPositionDataset(Dataset):
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
        
        # Collect all rotations and positions for normalization
        all_rots = []
        all_positions = []
        for _, p1, p2, p3 in self.data_triplets:
            for p in [p1, p2, p3]:
                all_rots.append(self.quat_to_9d(self.get_quat(p)))
                all_positions.append(self.get_position(p))
        
        self.rot_absolutes = np.array(all_rots)
        self.pos_absolutes = np.array(all_positions)
        
        # Separate normalization for rotations and positions
        self.rot_mean, self.rot_std = self.rot_absolutes.mean(axis=0), self.rot_absolutes.std(axis=0) + 1e-8
        self.pos_mean, self.pos_std = self.pos_absolutes.mean(axis=0), self.pos_absolutes.std(axis=0) + 1e-8
        
        logging.info(f"Rotation normalization - Mean: {self.rot_mean[:3]}, Std: {self.rot_std[:3]}")
        logging.info(f"Position normalization - Mean: {self.pos_mean}, Std: {self.pos_std}")
    
    def get_quat(self, pt_info): 
        return np.array([pt_info['RotationQuaternion'][k] for k in 'xyzw'])
    
    def get_position(self, pt_info):
        return np.array([pt_info['Position'][k] for k in 'xyz'])
    
    def quat_to_9d(self, quat): 
        return R.from_quat(quat).as_matrix().reshape(9)
    
    def __len__(self): 
        return len(self.data_triplets)
    
    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]
        def load_img(pt): 
            return self.image_processor(cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB), return_tensors="pt").pixel_values.squeeze(0)
        
        img1, img2, img3 = load_img(p1), load_img(p2), load_img(p3)
        
        # Get rotations and positions
        rot_p1 = (self.quat_to_9d(self.get_quat(p1)) - self.rot_mean) / self.rot_std
        rot_p2 = (self.quat_to_9d(self.get_quat(p2)) - self.rot_mean) / self.rot_std
        rot_p3_label = (self.quat_to_9d(self.get_quat(p3)) - self.rot_mean) / self.rot_std
        
        pos_p1 = (self.get_position(p1) - self.pos_mean) / self.pos_std
        pos_p2 = (self.get_position(p2) - self.pos_mean) / self.pos_std
        pos_p3_label = (self.get_position(p3) - self.pos_mean) / self.pos_std
        
        # Combine rotations and positions: [rot1, pos1, rot2, pos2]
        input_abs = np.concatenate([rot_p1, pos_p1, rot_p2, pos_p2])
        
        # Combine rotation and position labels: [rot3, pos3]
        combined_label = np.concatenate([rot_p3_label, pos_p3_label])
        
        return (img1, img2, img3, torch.tensor(input_abs).float(), torch.tensor(combined_label).float())

# --------------- 3. Loss Function and Metrics (Extended for Position) ---------------
def gram_schmidt_batch(matrices):
    """Apply Gram-Schmidt orthogonalization to batch of 3x3 matrices."""
    batch_size = matrices.shape[0]
    A = matrices.view(batch_size, 3, 3)
    
    u1 = A[:, :, 0]
    e1 = u1 / (torch.norm(u1, dim=1, keepdim=True) + 1e-8)
    
    u2 = A[:, :, 1] - torch.sum(e1 * A[:, :, 1], dim=1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + 1e-8)
    
    u3 = A[:, :, 2] - torch.sum(e1 * A[:, :, 2], dim=1, keepdim=True) * e1 - torch.sum(e2 * A[:, :, 2], dim=1, keepdim=True) * e2
    e3 = u3 / (torch.norm(u3, dim=1, keepdim=True) + 1e-8)
    
    Q = torch.stack([e1, e2, e3], dim=2)
    
    det = torch.det(Q)
    Q[det < 0, :, 2] = -Q[det < 0, :, 2]
    
    return Q

def nine_d_to_rotmat(nine_d):
    """Convert 9D representation to rotation matrix using Gram-Schmidt."""
    nine_d_clamped = torch.clamp(nine_d, -10, 10)
    rot_matrices = gram_schmidt_batch(nine_d_clamped)
    return rot_matrices

def angular_distance(pred_9d, target_9d):
    """Compute angular distance with stable implementation."""
    pred_rot = nine_d_to_rotmat(pred_9d)
    target_rot = nine_d_to_rotmat(target_9d)
    
    relative = torch.bmm(pred_rot.transpose(1, 2), target_rot)
    traces = torch.diagonal(relative, dim1=1, dim2=2).sum(dim=1)
    traces = torch.clamp(traces, -3.0 + 1e-6, 3.0 - 1e-6)
    
    cos_angle = (traces - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    
    angles_rad = torch.acos(cos_angle)
    angles_deg = angles_rad * (180.0 / np.pi)
    
    angles_deg = torch.where(torch.isnan(angles_deg), torch.tensor(180.0, device=angles_deg.device), angles_deg)
    
    return angles_deg

def position_distance(pred_pos, target_pos):
    """Compute Euclidean distance for positions."""
    return torch.norm(pred_pos - target_pos, dim=1)

class RotationPositionLoss(nn.Module):
    def __init__(self, angular_weight=0.1, position_weight=1.0, l2_reg=1e-4):
        super().__init__()
        self.angular_weight = angular_weight
        self.position_weight = position_weight
        self.l2_reg = l2_reg

    def orthogonality_loss(self, nine_d):
        """Encourage orthogonal matrices."""
        batch_size = nine_d.shape[0]
        matrices = nine_d.view(batch_size, 3, 3)
        
        AtA = torch.bmm(matrices.transpose(1, 2), matrices)
        I = torch.eye(3, device=nine_d.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        ortho_loss = torch.mean(torch.norm(AtA - I, p='fro', dim=(1, 2)) ** 2)
        return ortho_loss

    def forward(self, pred_12d, target_12d):
        # Split predictions and targets
        pred_rot, pred_pos = pred_12d[:, :9], pred_12d[:, 9:]
        target_rot, target_pos = target_12d[:, :9], target_12d[:, 9:]
        
        # Clamp predictions to prevent explosion
        pred_rot = torch.clamp(pred_rot, -10, 10)
        pred_pos = torch.clamp(pred_pos, -1000, 1000)
        
        # Rotation losses
        pred_rot_mat = nine_d_to_rotmat(pred_rot)
        target_rot_mat = nine_d_to_rotmat(target_rot)
        
        chordal_loss = torch.mean(torch.norm(pred_rot_mat - target_rot_mat, p='fro', dim=(1, 2)) ** 2)
        angular_errors = angular_distance(pred_rot, target_rot)
        angular_loss_term = torch.mean(angular_errors)
        ortho_loss = self.orthogonality_loss(pred_rot)
        
        # Position loss (simple MSE)
        position_loss = torch.mean((pred_pos - target_pos) ** 2)
        
        # L2 regularization
        l2_loss = torch.mean(pred_12d ** 2)
        
        # Combine losses
        total_loss = (chordal_loss + self.angular_weight * angular_loss_term + 0.01 * ortho_loss + 
                     self.position_weight * position_loss + self.l2_reg * l2_loss)
        
        if torch.isnan(total_loss):
            logging.warning("NaN detected in loss computation. Using MSE fallback.")
            total_loss = torch.mean((pred_12d - target_12d) ** 2)
        
        return total_loss

# --------------- 4. Evaluation and Plotting (Extended for Position) ---------------
def eval_epoch(model, loader, criterion, writer, step, rot_mean, rot_std, pos_mean, pos_std, output_dir):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            img1, img2, img3, input_abs, combined_label = [b.to(device) for b in batch]
            
            try:
                output = model(img1, img2, img3, input_abs)
                loss = criterion(output, combined_label)
                
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss in validation batch. Skipping.")
                    continue
                    
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_labels.append(combined_label.cpu().numpy())
                valid_batches += 1
                
            except Exception as e:
                logging.warning(f"Error in validation batch: {e}. Skipping.")
                continue
    
    if valid_batches == 0:
        logging.error("No valid validation batches. Cannot generate report.")
        return
    
    avg_loss = total_loss / valid_batches
    preds_np, labels_np = np.vstack(all_preds), np.vstack(all_labels)
    
    try:
        # Split rotation and position predictions/labels
        pred_rot, pred_pos = preds_np[:, :9], preds_np[:, 9:]
        label_rot, label_pos = labels_np[:, :9], labels_np[:, 9:]
        
        # Compute rotation metrics
        ang_errors = angular_distance(torch.tensor(pred_rot), torch.tensor(label_rot)).numpy()
        ang_errors = ang_errors[np.isfinite(ang_errors)]
        
        # Compute position metrics (denormalized)
        pred_pos_denorm = pred_pos * pos_std + pos_mean
        label_pos_denorm = label_pos * pos_std + pos_mean
        pos_errors = np.linalg.norm(pred_pos_denorm - label_pos_denorm, axis=1)
        
        if len(ang_errors) > 0 and len(pos_errors) > 0:
            mean_ang_error = np.mean(ang_errors)
            mean_pos_error = np.mean(pos_errors)
            
            logging.info(f"Validation @ Step {step}: Avg Loss={avg_loss:.4f}, "
                        f"Mean Angular Err={mean_ang_error:.2f}°, Mean Pos Err={mean_pos_error:.2f}mm")
            
            writer.add_scalar('Loss/val', avg_loss, step)
            writer.add_scalar('Metrics/mean_angular_error', mean_ang_error, step)
            writer.add_scalar('Metrics/mean_position_error', mean_pos_error, step)
            writer.add_histogram('Angular_Errors_val', ang_errors, step)
            writer.add_histogram('Position_Errors_val', pos_errors, step)
            
            # Generate plots
            pdf_path = os.path.join(output_dir, 'plots', f'results_step_{step}.pdf')
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            
            with PdfPages(pdf_path) as pdf:
                # Angular error plot
                plt.figure(figsize=(10, 4))
                plt.subplot(1, 2, 1)
                plt.hist(ang_errors, bins=50)
                plt.title(f'Angular Error @ Step {step}\n(Mean: {mean_ang_error:.2f}°)')
                plt.xlabel('Angular Error (degrees)')
                plt.ylabel('Count')
                
                # Position error plot
                plt.subplot(1, 2, 2)
                plt.hist(pos_errors, bins=50)
                plt.title(f'Position Error @ Step {step}\n(Mean: {mean_pos_error:.2f}mm)')
                plt.xlabel('Position Error (mm)')
                plt.ylabel('Count')
                
                plt.tight_layout()
                pdf.savefig()
                plt.close()
        else:
            logging.warning("No valid errors computed in validation.")
            
    except Exception as e:
        logging.error(f"Error computing validation metrics: {e}")

# --------------- 5. Main Training Loop (Updated for Position) ---------------
def main(args):
    output_dir = f"./rotation_position_training_runs/{datetime.datetime.now().strftime('%Y%b%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')), logging.StreamHandler()])
    logging.info(f"Starting rotation+position training. Outputting to {output_dir}\nArgs: {args}")

    train_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]
    if args.single_patient:
        logging.warning("--- RUNNING IN SINGLE PATIENT MODE FOR DEBUGGING ---")
        train_dirs, val_dirs = train_dirs[:1], val_dirs[:1]

    train_dataset = RotationPositionDataset(train_dirs, args.pairs_per_patient)
    val_dataset = RotationPositionDataset(val_dirs, args.val_pairs)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = DinoV2RotationPositionTransformer().to(device)
    model.encoder.gradient_checkpointing_enable()

    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': args.lr_head, 'weight_decay': 1e-4}
    ]
    optimizer = optim.AdamW(param_groups, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-7)
    criterion = RotationPositionLoss(position_weight=args.position_weight)
    scaler = amp.GradScaler(init_scale=512.0)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Initial validation
    logging.info("Running initial validation...")
    eval_epoch(model, val_loader, criterion, writer, 0, 
               val_dataset.rot_mean, val_dataset.rot_std, 
               val_dataset.pos_mean, val_dataset.pos_std, output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_step_0.pth'))

    train_iter = iter(train_loader)
    pbar = tqdm(range(1, args.total_steps + 1), desc="Training")
    
    ema_loss = None
    smoothing_factor = 0.9
    consecutive_nan_count = 0
    max_consecutive_nans = 10

    for step in pbar:
        try:
            batch = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            batch = next(train_iter)
            
        img1, img2, img3, input_abs, combined_label = [b.to(device) for b in batch]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            with amp.autocast(device_type='cuda'):
                outputs = model(img1, img2, img3, input_abs)
                loss = criterion(outputs, combined_label)
            
            raw_loss = loss.item()
            
            if not np.isfinite(raw_loss) or raw_loss > 1000:
                consecutive_nan_count += 1
                logging.warning(f"Problematic loss at step {step}: {raw_loss}. Count: {consecutive_nan_count}")
                
                if consecutive_nan_count >= max_consecutive_nans:
                    logging.error(f"Too many consecutive problematic losses. Stopping training.")
                    break
                continue
            else:
                consecutive_nan_count = 0
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if grad_norm > 2000.0:
                logging.warning(f"Very large gradient norm: {grad_norm:.2f} at step {step}")
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if ema_loss is None:
                ema_loss = raw_loss
            else:
                ema_loss = smoothing_factor * ema_loss + (1 - smoothing_factor) * raw_loss
            
            pbar.set_postfix(smooth_loss=f"{ema_loss:.4f}", grad_norm=f"{grad_norm:.2f}")
            
            writer.add_scalar('Loss/train', raw_loss, step)
            writer.add_scalar('Metrics/grad_norm', grad_norm, step)
            writer.add_scalar('LR/head', optimizer.param_groups[1]['lr'], step)
            writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], step)

            if step % args.val_freq == 0 or step == args.total_steps:
                logging.info(f"Running validation at step {step}...")
                eval_epoch(model, val_loader, criterion, writer, step, 
                          val_dataset.rot_mean, val_dataset.rot_std,
                          val_dataset.pos_mean, val_dataset.pos_std, output_dir)
                torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_step_{step}.pth'))
                
        except Exception as e:
            logging.error(f"Error in training step {step}: {e}")
            consecutive_nan_count += 1
            if consecutive_nan_count >= max_consecutive_nans:
                logging.error("Too many consecutive errors. Stopping training.")
                break
            continue

    pbar.close()
    writer.close()
    logging.info("Training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DinoV2-based Rotation+Position Transformer.")
    parser.add_argument('--data_root', type=str, default='../ct_data_rotation_only')
    parser.add_argument('--lr_head', type=float, default=1e-5, help='Peak learning rate for the new layers.')
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Peak learning rate for the backbone.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pairs_per_patient', type=int, default=1000)
    parser.add_argument('--val_pairs', type=int, default=200)
    parser.add_argument('--position_weight', type=float, default=1.0, help='Weight for position loss relative to rotation loss')
    parser.add_argument('--single_patient', action='store_true', help='Debug with one patient.')
    args = parser.parse_args()
    main(args)