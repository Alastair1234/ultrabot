# ==============================================================================
# File: model_rotation_stable.py
#
# Description:
# This version prevents NaN/infinity issues through:
# 1. Output normalization and constraints
# 2. Gradient clipping and loss scaling
# 3. Better initialization
# 4. Regularization techniques
# 5. EMA smoothing for loss display
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

# --------------- 1. The Model Definition (STABILIZED) ---------------
class DinoV2RotationTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=9):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        encoder_dim = self.encoder.config.hidden_size
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # Better initialization for rotation embedder
        self.rot_embedder = nn.Linear(rot_dim, encoder_dim)
        nn.init.xavier_uniform_(self.rot_embedder.weight, gain=0.1)
        nn.init.zeros_(self.rot_embedder.bias)
        
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
        
        # Constrained output layer with smaller initialization
        self.regressor = nn.Linear(embed_dim, 9)
        nn.init.xavier_uniform_(self.regressor.weight, gain=0.01)  # Much smaller gain
        nn.init.zeros_(self.regressor.bias)

    def forward(self, img1, img2, img3, input_abs):
        # Get features and apply layer norm to prevent explosion
        feat1 = self.layer_norm1(self.encoder(img1).last_hidden_state[:, 0])
        feat2 = self.layer_norm1(self.encoder(img2).last_hidden_state[:, 0])
        feat3 = self.layer_norm1(self.encoder(img3).last_hidden_state[:, 0])
        
        # Split and constrain input rotations
        abs_p1, abs_p2 = input_abs[:, :9], input_abs[:, 9:]
        abs_p1 = torch.clamp(abs_p1, -10, 10)  # Prevent extreme inputs
        abs_p2 = torch.clamp(abs_p2, -10, 10)
        
        # Embed rotations with normalization
        rot_emb1 = self.layer_norm2(self.rot_embedder(abs_p1))
        rot_emb2 = self.layer_norm2(self.rot_embedder(abs_p2))
        
        # Fuse features
        fused_feat1 = feat1 + rot_emb1
        fused_feat2 = feat2 + rot_emb2
        
        # Apply dropout
        fused_feat1 = self.dropout(fused_feat1)
        fused_feat2 = self.dropout(fused_feat2)
        feat3 = self.dropout(feat3)
        
        # Combine and project
        combined = torch.cat([fused_feat1, fused_feat2, feat3], dim=1)
        projected = self.projector(combined).unsqueeze(1)
        
        # Apply transformer
        transformer_out = self.transformer(projected).squeeze(1)
        
        # Constrained output - prevent explosion
        raw_output = self.regressor(transformer_out)
        
        # Normalize output to prevent extreme values
        output_norm = torch.norm(raw_output, dim=1, keepdim=True)
        max_norm = 3.0  # Reasonable bound for normalized 9D rotations
        output = raw_output * torch.clamp(max_norm / (output_norm + 1e-8), max=1.0)
        
        return output

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

# --------------- 3. Loss Function and Metrics (STABILIZED) ---------------
def gram_schmidt_batch(matrices):
    """Apply Gram-Schmidt orthogonalization to batch of 3x3 matrices."""
    batch_size = matrices.shape[0]
    # Reshape to (batch, 3, 3)
    A = matrices.view(batch_size, 3, 3)
    
    # Gram-Schmidt process
    u1 = A[:, :, 0]  # First column
    e1 = u1 / (torch.norm(u1, dim=1, keepdim=True) + 1e-8)
    
    u2 = A[:, :, 1] - torch.sum(e1 * A[:, :, 1], dim=1, keepdim=True) * e1
    e2 = u2 / (torch.norm(u2, dim=1, keepdim=True) + 1e-8)
    
    u3 = A[:, :, 2] - torch.sum(e1 * A[:, :, 2], dim=1, keepdim=True) * e1 - torch.sum(e2 * A[:, :, 2], dim=1, keepdim=True) * e2
    e3 = u3 / (torch.norm(u3, dim=1, keepdim=True) + 1e-8)
    
    # Stack to form orthogonal matrix
    Q = torch.stack([e1, e2, e3], dim=2)
    
    # Ensure proper rotation (det = +1)
    det = torch.det(Q)
    Q[det < 0, :, 2] = -Q[det < 0, :, 2]
    
    return Q

def nine_d_to_rotmat(nine_d):
    """Convert 9D representation to rotation matrix using Gram-Schmidt."""
    batch_size = nine_d.shape[0]
    
    # Clamp input to prevent extreme values
    nine_d_clamped = torch.clamp(nine_d, -10, 10)
    
    # Use Gram-Schmidt instead of SVD to avoid convergence issues
    rot_matrices = gram_schmidt_batch(nine_d_clamped)
    
    return rot_matrices

def angular_distance(pred_9d, target_9d):
    """Compute angular distance with stable implementation."""
    pred_rot = nine_d_to_rotmat(pred_9d)
    target_rot = nine_d_to_rotmat(target_9d)
    
    # Compute relative rotation
    relative = torch.bmm(pred_rot.transpose(1, 2), target_rot)
    
    # Extract trace with numerical stability
    traces = torch.diagonal(relative, dim1=1, dim2=2).sum(dim=1)
    traces = torch.clamp(traces, -3.0 + 1e-6, 3.0 - 1e-6)
    
    # Compute angle
    cos_angle = (traces - 1) / 2
    cos_angle = torch.clamp(cos_angle, -1.0 + 1e-7, 1.0 - 1e-7)
    
    angles_rad = torch.acos(cos_angle)
    angles_deg = angles_rad * (180.0 / np.pi)
    
    # Replace any remaining NaN with 180 degrees (worst case)
    angles_deg = torch.where(torch.isnan(angles_deg), torch.tensor(180.0, device=angles_deg.device), angles_deg)
    
    return angles_deg

class RotationLoss(nn.Module):
    def __init__(self, angular_weight=0.1, l2_reg=1e-4):
        super().__init__()
        self.angular_weight = angular_weight
        self.l2_reg = l2_reg

    def orthogonality_loss(self, nine_d):
        """Encourage orthogonal matrices."""
        batch_size = nine_d.shape[0]
        matrices = nine_d.view(batch_size, 3, 3)
        
        # Compute AᵀA - I
        AtA = torch.bmm(matrices.transpose(1, 2), matrices)
        I = torch.eye(3, device=nine_d.device).unsqueeze(0).expand(batch_size, -1, -1)
        
        ortho_loss = torch.mean(torch.norm(AtA - I, p='fro', dim=(1, 2)) ** 2)
        return ortho_loss

    def forward(self, pred_9d, target_9d):
        # Clamp predictions to prevent explosion
        pred_9d = torch.clamp(pred_9d, -10, 10)
        
        # Primary losses
        pred_rot = nine_d_to_rotmat(pred_9d)
        target_rot = nine_d_to_rotmat(target_9d)
        
        # Frobenius norm loss (more stable than before)
        chordal_loss = torch.mean(torch.norm(pred_rot - target_rot, p='fro', dim=(1, 2)) ** 2)
        
        # Angular loss with stability check
        angular_errors = angular_distance(pred_9d, target_9d)
        angular_loss_term = torch.mean(angular_errors)
        
        # Orthogonality regularization
        ortho_loss = self.orthogonality_loss(pred_9d)
        
        # L2 regularization on predictions
        l2_loss = torch.mean(pred_9d ** 2)
        
        # Check for NaN in any component
        total_loss = chordal_loss + self.angular_weight * angular_loss_term + 0.01 * ortho_loss + self.l2_reg * l2_loss
        
        if torch.isnan(total_loss):
            logging.warning("NaN detected in loss computation. Using MSE fallback.")
            # Fallback to simple MSE
            total_loss = torch.mean((pred_9d - target_9d) ** 2)
        
        return total_loss

# --------------- 4. Evaluation and Plotting (UPDATED) ---------------
def eval_epoch(model, loader, criterion, writer, step, mean, std, output_dir):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            img1, img2, img3, input_abs, abs_p3 = [b.to(device) for b in batch]
            
            try:
                output = model(img1, img2, img3, input_abs)
                loss = criterion(output, abs_p3)
                
                # Skip batch if loss is not finite
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss in validation batch. Skipping.")
                    continue
                    
                total_loss += loss.item()
                all_preds.append(output.cpu().numpy())
                all_labels.append(abs_p3.cpu().numpy())
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
        ang_errors = angular_distance(torch.tensor(preds_np), torch.tensor(labels_np)).numpy()
        # Remove any NaN errors for statistics
        ang_errors = ang_errors[np.isfinite(ang_errors)]
        
        if len(ang_errors) > 0:
            mean_error = np.mean(ang_errors)
            logging.info(f"Validation @ Step {step}: Avg Loss={avg_loss:.4f}, Mean Angular Err={mean_error:.2f}°")
            writer.add_scalar('Loss/val', avg_loss, step)
            writer.add_scalar('Metrics/mean_angular_error', mean_error, step)
            writer.add_histogram('Angular_Errors_val', ang_errors, step)
            
            # Generate plot
            pdf_path = os.path.join(output_dir, 'plots', f'results_step_{step}.pdf')
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            errors_denorm = angular_distance(torch.tensor(preds_np * std + mean), torch.tensor(labels_np * std + mean)).numpy()
            errors_denorm = errors_denorm[np.isfinite(errors_denorm)]
            
            if len(errors_denorm) > 0:
                with PdfPages(pdf_path) as pdf:
                    plt.figure()
                    plt.hist(errors_denorm, bins=50)
                    plt.title(f'Angular Error @ Step {step} (Mean: {np.mean(errors_denorm):.2f})')
                    pdf.savefig()
                    plt.close()
        else:
            logging.warning("No valid angular errors computed in validation.")
            
    except Exception as e:
        logging.error(f"Error computing validation metrics: {e}")

# --------------- 5. Main Training Loop (STABILIZED) ---------------
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

    # Use different learning rates and add weight decay
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': args.lr_head, 'weight_decay': 1e-4}
    ]
    optimizer = optim.AdamW(param_groups, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-7)
    criterion = RotationLoss()
    scaler = amp.GradScaler(init_scale=512.0)  # Lower initial scale
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    train_iter = iter(train_loader)
    pbar = tqdm(range(args.total_steps), desc="Training")
    
    # EMA for smooth loss display
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
            
        img1, img2, img3, input_abs, abs_p3 = [b.to(device) for b in batch]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            with amp.autocast(device_type='cuda'):
                outputs = model(img1, img2, img3, input_abs)
                loss = criterion(outputs, abs_p3)
            
            raw_loss = loss.item()
            
            # Check for problematic loss values
            if not np.isfinite(raw_loss) or raw_loss > 1000:
                consecutive_nan_count += 1
                logging.warning(f"Problematic loss at step {step}: {raw_loss}. Count: {consecutive_nan_count}")
                
                if consecutive_nan_count >= max_consecutive_nans:
                    logging.error(f"Too many consecutive problematic losses. Stopping training.")
                    break
                continue
            else:
                consecutive_nan_count = 0  # Reset counter
            
            # Backward pass with gradient scaling
            scaler.scale(loss).backward()
            
            # Unscale and check gradients
            scaler.unscale_(optimizer)
            
            # More aggressive gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            # Check if gradients are reasonable
            if grad_norm > 10.0:
                logging.warning(f"Large gradient norm: {grad_norm:.2f} at step {step}")
            
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            # Update EMA loss
            if ema_loss is None:
                ema_loss = raw_loss
            else:
                ema_loss = smoothing_factor * ema_loss + (1 - smoothing_factor) * raw_loss
            
            pbar.set_postfix(smooth_loss=f"{ema_loss:.4f}", grad_norm=f"{grad_norm:.2f}")
            
            # Log metrics
            writer.add_scalar('Loss/train', raw_loss, step)
            writer.add_scalar('Metrics/grad_norm', grad_norm, step)
            writer.add_scalar('LR/head', optimizer.param_groups[1]['lr'], step)
            writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], step)

            # Validation
            if step > 0 and (step % args.val_freq == 0 or step == args.total_steps - 1):
                eval_epoch(model, val_loader, criterion, writer, step, val_dataset.mean, val_dataset.std, output_dir)
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
    parser = argparse.ArgumentParser(description="Train a DinoV2-based Rotation Transformer.")
    parser.add_argument('--data_root', type=str, default='../ct_data_rotation_only')
    parser.add_argument('--lr_head', type=float, default=1e-5, help='Peak learning rate for the new layers.')  # Lower LR
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Peak learning rate for the backbone.')  # Lower LR
    parser.add_argument('--batch_size', type=int, default=32)  # Smaller batch size for stability
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pairs_per_patient', type=int, default=1000)
    parser.add_argument('--val_pairs', type=int, default=200)
    parser.add_argument('--single_patient', action='store_true', help='Debug with one patient.')
    args = parser.parse_args()
    main(args)