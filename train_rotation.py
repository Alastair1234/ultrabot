# ==============================================================================
# File: model_rotation_delta_6d_complete.py
#
# Description:
# Complete 6D delta rotation prediction model
# Based on "On the Continuity of Rotation Representations in Neural Networks"
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

# --------------- 6D ROTATION UTILITIES ---------------
def rotation_matrix_to_6d(rot_matrices):
    """Convert rotation matrices to 6D representation (first two columns)."""
    # rot_matrices: (..., 3, 3) -> (..., 6)
    return rot_matrices[..., :, :2].reshape(*rot_matrices.shape[:-2], 6)

def rotation_6d_to_matrix(rot_6d):
    """Convert 6D representation back to rotation matrix using Gram-Schmidt."""
    batch_size = rot_6d.shape[0]
    
    # Reshape to get first two columns
    a1 = rot_6d[:, :3]  # First column
    a2 = rot_6d[:, 3:]  # Second column
    
    # Normalize first vector
    b1 = a1 / (torch.norm(a1, dim=1, keepdim=True) + 1e-8)
    
    # Make second vector orthogonal to first
    b2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = b2 / (torch.norm(b2, dim=1, keepdim=True) + 1e-8)
    
    # Third column is cross product
    b3 = torch.cross(b1, b2, dim=1)
    
    # Stack to form rotation matrix [b1, b2, b3]
    rot_matrix = torch.stack([b1, b2, b3], dim=2)
    
    return rot_matrix

def angular_distance_6d(pred_6d, target_6d):
    """Compute angular distance between 6D representations."""
    pred_rot = rotation_6d_to_matrix(pred_6d)
    target_rot = rotation_6d_to_matrix(target_6d)
    
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
    
    # Replace any NaN with 180 degrees
    angles_deg = torch.where(torch.isnan(angles_deg), torch.tensor(180.0, device=angles_deg.device), angles_deg)
    
    return angles_deg

# --------------- 1. The Model Definition ---------------
class DinoV2RotationDelta6D(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=6):
        super().__init__()
        self.encoder = AutoModel.from_pretrained('facebook/dinov2-small')
        encoder_dim = self.encoder.config.hidden_size
        
        # Add dropout for regularization
        self.dropout = nn.Dropout(0.1)
        
        # 6D rotation delta embedder
        self.rot_delta_embedder = nn.Linear(rot_dim, encoder_dim)
        nn.init.xavier_uniform_(self.rot_delta_embedder.weight, gain=0.1)
        nn.init.zeros_(self.rot_delta_embedder.bias)
        
        # Add layer normalization
        self.layer_norm1 = nn.LayerNorm(encoder_dim)
        self.layer_norm2 = nn.LayerNorm(encoder_dim)
        
        # Project concatenated features (keeping 3 features like working version)
        self.projector = nn.Linear(encoder_dim * 3, embed_dim)
        nn.init.xavier_uniform_(self.projector.weight, gain=0.1)
        nn.init.zeros_(self.projector.bias)
        
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim),
            activation='gelu', batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=depth)
        
        # Output 6D representation
        self.regressor = nn.Linear(embed_dim, 6)
        nn.init.xavier_uniform_(self.regressor.weight, gain=0.01)
        nn.init.zeros_(self.regressor.bias)

    def forward(self, img1, img2, img3, delta_12_6d):
        # Get features and apply layer norm
        feat1 = self.layer_norm1(self.encoder(img1).last_hidden_state[:, 0])
        feat2 = self.layer_norm1(self.encoder(img2).last_hidden_state[:, 0])
        feat3 = self.layer_norm1(self.encoder(img3).last_hidden_state[:, 0])
        
        # Embed the known 6D rotation delta
        delta_emb = self.layer_norm2(self.rot_delta_embedder(delta_12_6d))
        
        # Fuse img2 feature with delta context (img2 is the bridge)
        fused_feat2 = feat2 + delta_emb
        
        # Apply dropout
        feat1 = self.dropout(feat1)
        fused_feat2 = self.dropout(fused_feat2)
        feat3 = self.dropout(feat3)
        
        # Combine features (same order as working version)
        combined = torch.cat([feat1, fused_feat2, feat3], dim=1)
        projected = self.projector(combined).unsqueeze(1)
        
        # Apply transformer
        transformer_out = self.transformer(projected).squeeze(1)
        
        # Output 6D representation (no special normalization needed!)
        output_6d = self.regressor(transformer_out)
        
        return output_6d

# --------------- 2. Dataset ---------------
class RotationDelta6DDataset(Dataset):
    def __init__(self, patient_dirs, pairs_per_patient=1000):
        self.image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-small')
        self.data_triplets = []
        
        print(f"Loading data from {len(patient_dirs)} patient directories...")
        for patient_dir in tqdm(patient_dirs, desc="Loading Data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')
            if not os.path.exists(info_path): 
                continue
            
            with open(info_path) as f: 
                points = json.load(f)['PointInfos']
            if len(points) < 3: 
                continue
                
            for _ in range(pairs_per_patient): 
                self.data_triplets.append((images_dir, *random.sample(points, 3)))
        
        print(f"Created {len(self.data_triplets)} triplets. Computing 6D delta statistics...")
        
        # Compute 6D delta statistics
        all_deltas_6d = []
        sample_size = min(len(self.data_triplets), 10000)  # Don't compute all for efficiency
        
        for i, (_, p1, p2, p3) in enumerate(random.sample(self.data_triplets, sample_size)):
            r1 = self.get_rotation_matrix(p1)
            r2 = self.get_rotation_matrix(p2)
            r3 = self.get_rotation_matrix(p3)
            
            # Compute relative rotations
            delta_12 = np.dot(r2, r1.T)  # R2 * R1^T
            delta_23 = np.dot(r3, r2.T)  # R3 * R2^T
            
            # Convert to 6D (first two columns)
            delta_12_6d = delta_12[:, :2].reshape(6)
            delta_23_6d = delta_23[:, :2].reshape(6)
            
            all_deltas_6d.extend([delta_12_6d, delta_23_6d])
        
        self.deltas_6d = np.array(all_deltas_6d)
        self.mean = self.deltas_6d.mean(axis=0)
        self.std = self.deltas_6d.std(axis=0) + 1e-8
        
        print(f"6D Delta statistics - Mean range: [{self.mean.min():.4f}, {self.mean.max():.4f}]")
        print(f"                   - Std range:  [{self.std.min():.4f}, {self.std.max():.4f}]")
        
    def get_quat(self, pt_info): 
        return np.array([pt_info['RotationQuaternion'][k] for k in 'xyzw'])
    
    def get_rotation_matrix(self, pt_info):
        quat = self.get_quat(pt_info)
        return R.from_quat(quat).as_matrix()
    
    def __len__(self): 
        return len(self.data_triplets)
    
    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]
        
        def load_img(pt): 
            img_path = os.path.join(images_dir, pt['FileName'])
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return self.image_processor(img_rgb, return_tensors="pt").pixel_values.squeeze(0)
        
        img1, img2, img3 = load_img(p1), load_img(p2), load_img(p3)
        
        # Get rotation matrices
        r1 = self.get_rotation_matrix(p1)
        r2 = self.get_rotation_matrix(p2)
        r3 = self.get_rotation_matrix(p3)
        
        # Compute deltas
        delta_12 = np.dot(r2, r1.T)
        delta_23 = np.dot(r3, r2.T)
        
        # Convert to 6D representation (first two columns flattened)
        delta_12_6d = delta_12[:, :2].reshape(6)
        delta_23_6d = delta_23[:, :2].reshape(6)
        
        # Normalize using dataset statistics
        delta_12_6d_norm = (delta_12_6d - self.mean) / self.std
        delta_23_6d_norm = (delta_23_6d - self.mean) / self.std
        
        return (
            img1, 
            img2, 
            img3, 
            torch.tensor(delta_12_6d_norm).float(), 
            torch.tensor(delta_23_6d_norm).float()
        )

# --------------- 3. Loss Function ---------------
class RotationDelta6DLoss(nn.Module):
    def __init__(self, angular_weight=0.1, l2_reg=1e-4):
        super().__init__()
        self.angular_weight = angular_weight
        self.l2_reg = l2_reg

    def forward(self, pred_6d, target_6d):
        # Primary MSE loss on 6D representation (this is the key advantage!)
        mse_loss = torch.mean((pred_6d - target_6d) ** 2)
        
        # Angular loss for rotation quality
        try:
            angular_errors = angular_distance_6d(pred_6d, target_6d)
            angular_loss = torch.mean(angular_errors)
        except:
            # Fallback if angular computation fails
            angular_loss = torch.tensor(0.0, device=pred_6d.device)
        
        # L2 regularization
        l2_loss = torch.mean(pred_6d ** 2)
        
        # Combined loss (MSE is primary since 6D is continuous)
        total_loss = mse_loss + self.angular_weight * angular_loss + self.l2_reg * l2_loss
        
        if torch.isnan(total_loss):
            logging.warning("NaN detected in loss computation. Using MSE only.")
            total_loss = mse_loss
        
        return total_loss

# --------------- 4. Evaluation ---------------
def eval_epoch(model, loader, criterion, writer, step, mean, std, output_dir):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    valid_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(loader, desc='Evaluating', leave=False):
            img1, img2, img3, delta_12_6d, delta_23_target_6d = [b.to(device) for b in batch]
            
            try:
                delta_23_pred_6d = model(img1, img2, img3, delta_12_6d)
                loss = criterion(delta_23_pred_6d, delta_23_target_6d)
                
                if not torch.isfinite(loss):
                    logging.warning(f"Non-finite loss in validation batch. Skipping.")
                    continue
                    
                total_loss += loss.item()
                all_preds.append(delta_23_pred_6d.cpu().numpy())
                all_labels.append(delta_23_target_6d.cpu().numpy())
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
        # Compute angular errors
        ang_errors = angular_distance_6d(torch.tensor(preds_np), torch.tensor(labels_np)).numpy()
        ang_errors = ang_errors[np.isfinite(ang_errors)]
        
        if len(ang_errors) > 0:
            mean_error = np.mean(ang_errors)
            median_error = np.median(ang_errors)
            logging.info(f"Validation @ Step {step}: Loss={avg_loss:.4f}, Mean Angular Err={mean_error:.2f}°, Median={median_error:.2f}°")
            
            # Log to tensorboard
            writer.add_scalar('Loss/val', avg_loss, step)
            writer.add_scalar('Metrics/mean_6d_delta_angular_error', mean_error, step)
            writer.add_scalar('Metrics/median_6d_delta_angular_error', median_error, step)
            writer.add_histogram('6D_Delta_Angular_Errors_val', ang_errors, step)
            
            # Generate plot
            pdf_path = os.path.join(output_dir, 'plots', f'6d_delta_results_step_{step}.pdf')
            os.makedirs(os.path.dirname(pdf_path), exist_ok=True)
            
            # Denormalize for plotting
            preds_denorm = preds_np * std + mean
            labels_denorm = labels_np * std + mean
            errors_denorm = angular_distance_6d(
                torch.tensor(preds_denorm), 
                torch.tensor(labels_denorm)
            ).numpy()
            errors_denorm = errors_denorm[np.isfinite(errors_denorm)]
            
            if len(errors_denorm) > 0:
                with PdfPages(pdf_path) as pdf:
                    plt.figure(figsize=(10, 6))
                    plt.hist(errors_denorm, bins=50, alpha=0.7, edgecolor='black')
                    plt.title(f'6D Delta Angular Error @ Step {step}\nMean: {np.mean(errors_denorm):.2f}°, Median: {np.median(errors_denorm):.2f}°')
                    plt.xlabel('Angular Error (degrees)')
                    plt.ylabel('Frequency')
                    plt.grid(True, alpha=0.3)
                    pdf.savefig(bbox_inches='tight')
                    plt.close()
        else:
            logging.warning("No valid angular errors computed in validation.")
            
    except Exception as e:
        logging.error(f"Error computing validation metrics: {e}")

# --------------- 5. Main Training Loop ---------------
def main(args):
    output_dir = f"./rotation_6d_delta_training_runs/{datetime.datetime.now().strftime('%Y%b%d_%H-%M-%S')}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                        handlers=[logging.FileHandler(os.path.join(output_dir, 'train.log')), 
                                 logging.StreamHandler()])
    
    logging.info(f"Starting 6D delta rotation training")
    logging.info(f"Output directory: {output_dir}")
    logging.info(f"Arguments: {args}")

    # Load data directories
    train_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]
    
    if args.single_patient:
        logging.warning("--- RUNNING IN SINGLE PATIENT MODE FOR DEBUGGING ---")
        train_dirs, val_dirs = train_dirs[:1], val_dirs[:1]

    logging.info(f"Found {len(train_dirs)} training patients, {len(val_dirs)} validation patients")

    # Create datasets
    train_dataset = RotationDelta6DDataset(train_dirs, args.pairs_per_patient)
    val_dataset = RotationDelta6DDataset(val_dirs, args.val_pairs)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logging.info(f"Training dataset: {len(train_dataset)} samples")
    logging.info(f"Validation dataset: {len(val_dataset)} samples")

    # Create model
    model = DinoV2RotationDelta6D().to(device)
    model.encoder.gradient_checkpointing_enable()

    # Setup optimizer (same as working version)
    param_groups = [
        {'params': model.encoder.parameters(), 'lr': args.lr_backbone, 'weight_decay': 1e-5},
        {'params': [p for n, p in model.named_parameters() if not n.startswith('encoder.')], 'lr': args.lr_head, 'weight_decay': 1e-4}
    ]
    optimizer = optim.AdamW(param_groups, eps=1e-8)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.total_steps, eta_min=1e-7)
    criterion = RotationDelta6DLoss()
    scaler = amp.GradScaler(init_scale=512.0)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'tensorboard'))

    # Run initial validation
    logging.info("Running initial validation...")
    eval_epoch(model, val_loader, criterion, writer, 0, val_dataset.mean, val_dataset.std, output_dir)
    torch.save(model.state_dict(), os.path.join(output_dir, f'checkpoint_step_0.pth'))

    # Training loop
    train_iter = iter(train_loader)
    pbar = tqdm(range(1, args.total_steps + 1), desc="Training 6D Delta Model")
    
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
            
        img1, img2, img3, delta_12_6d, delta_23_target_6d = [b.to(device) for b in batch]

        model.train()
        optimizer.zero_grad(set_to_none=True)
        
        try:
            with amp.autocast(device_type='cuda'):
                delta_23_pred_6d = model(img1, img2, img3, delta_12_6d)
                loss = criterion(delta_23_pred_6d, delta_23_target_6d)
            
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
                consecutive_nan_count = 0
            
            # Backward pass
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            
            if grad_norm > 1000.0:
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
            if step % 10 == 0:  # Log every 10 steps to reduce overhead
                writer.add_scalar('Loss/train', raw_loss, step)
                writer.add_scalar('Metrics/grad_norm', grad_norm, step)
                writer.add_scalar('LR/head', optimizer.param_groups[1]['lr'], step)
                writer.add_scalar('LR/backbone', optimizer.param_groups[0]['lr'], step)
                writer.add_scalar('Loss/train_ema', ema_loss, step)

            # Validation
            if step % args.val_freq == 0 or step == args.total_steps:
                logging.info(f"Running validation at step {step}...")
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
    logging.info("6D delta rotation training finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a DinoV2-based 6D Rotation Delta Transformer.")
    parser.add_argument('--data_root', type=str, default='../ct_data_rotation_only')
    parser.add_argument('--lr_head', type=float, default=1e-5, help='Peak learning rate for the new layers.')
    parser.add_argument('--lr_backbone', type=float, default=1e-6, help='Peak learning rate for the backbone.')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--total_steps', type=int, default=20000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pairs_per_patient', type=int, default=1000)
    parser.add_argument('--val_pairs', type=int, default=200)
    parser.add_argument('--single_patient', action='store_true', help='Debug with one patient.')
    args = parser.parse_args()
    main(args)