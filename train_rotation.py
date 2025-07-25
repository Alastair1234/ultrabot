import os
import json
import torch
import numpy as np
import cv2
import random
import datetime
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim, amp
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from torch.utils.tensorboard import SummaryWriter
import logging
from logging.handlers import RotatingFileHandler
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class DinoV2RotationTransformer(nn.Module):
    def __init__(self, embed_dim=384, num_heads=6, depth=3, mlp_ratio=4, rot_dim=9):
        super().__init__()
        self.encoder = timm.create_model('facebook/dinov2-base', pretrained=True, num_classes=0)
        encoder_dim = self.encoder.embed_dim
        
        # Embedder for rotations (project 9D to encoder_dim)
        self.rot_embedder = nn.Linear(rot_dim, encoder_dim)
        
        # Projector for concatenated fused features
        self.projector = nn.Linear(encoder_dim * 3, embed_dim)
        
        layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=int(mlp_ratio * embed_dim), activation='gelu')
        self.transformer = nn.TransformerEncoder(layer, num_layers=depth)
        self.regressor = nn.Linear(embed_dim, 9)  # Output 9D absolute for p3

    def forward(self, img1, img2, img3, input_abs):
        # Split input_abs into abs_p1 (first 9) and abs_p2 (next 9)
        abs_p1 = input_abs[:, :9]  # (B, 9)
        abs_p2 = input_abs[:, 9:]  # (B, 9)
        
        # Extract image features
        feat1 = self.encoder(img1)  # (B, encoder_dim)
        feat2 = self.encoder(img2)
        feat3 = self.encoder(img3)
        
        # Early fusion: Embed rotations and add to their image features
        embedded_p1 = self.rot_embedder(abs_p1)  # (B, encoder_dim)
        embedded_p2 = self.rot_embedder(abs_p2)
        fused_feat1 = feat1 + embedded_p1
        fused_feat2 = feat2 + embedded_p2
        # feat3 has no rotation (predicting it), so leave as-is
        
        # Concatenate fused features
        combined = torch.cat([fused_feat1, fused_feat2, feat3], dim=1)  # (B, 3*encoder_dim)
        
        projected = self.projector(combined).unsqueeze(0)  # (1, B, embed_dim)
        transformer_out = self.transformer(projected).squeeze(0)
        return self.regressor(transformer_out)

class RotationDataset(Dataset):
    def __init__(self, patient_dirs, pairs_per_patient=2000):
        self.data_triplets = []
        for patient_dir in tqdm(patient_dirs, desc="Loading data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')
            if not os.path.exists(info_path) or not os.path.isdir(images_dir):
                continue
            with open(info_path) as f:
                points = json.load(f)['PointInfos']
            if len(points) < 3:
                continue
            for _ in range(pairs_per_patient):
                p1, p2, p3 = random.sample(points, 3)
                self.data_triplets.append((images_dir, p1, p2, p3))

        # Compute absolute 9D for all
        self.absolutes = np.array([self.quat_to_9d(self.get_quat(p)) for _, p1, p2, p3 in self.data_triplets for p in [p1, p2, p3]]).reshape(-1, 3, 9)
        self.mean = self.absolutes.mean(axis=(0,1))
        self.std = self.absolutes.std(axis=(0,1)) + 1e-8
        
        # Log data stats
        logger = logging.getLogger('rotation_training')
        norms = np.linalg.norm(self.absolutes.reshape(-1, 9), axis=1)
        logger.info(f"Dataset stats (absolutes): Samples={len(self)}, Mean norm={np.mean(norms):.4f}, Std={np.std(norms):.4f}, Min/Max={np.min(norms):.4f}/{np.max(norms):.4f}")
        logger.info(f"Absolute mean={self.mean.tolist()}, std={self.std.tolist()}")

    def get_quat(self, pt):
        return np.array([pt['RotationQuaternion'][k] for k in 'xyzw'])

    def quat_to_9d(self, quat):
        return R.from_quat(quat).as_matrix().reshape(9)

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]

        def load_img(pt):
            img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            return torch.tensor(img).permute(2, 0, 1).float()

        img1, img2, img3 = load_img(p1), load_img(p2), load_img(p3)
        
        # Absolutes (normalized)
        abs_p1 = (self.quat_to_9d(self.get_quat(p1)) - self.mean) / self.std
        abs_p2 = (self.quat_to_9d(self.get_quat(p2)) - self.mean) / self.std
        abs_p3 = (self.quat_to_9d(self.get_quat(p3)) - self.mean) / self.std
        
        # Input: Concat abs_p1 and abs_p2 (18D)
        input_abs = np.concatenate([abs_p1, abs_p2])
        
        return img1, img2, img3, torch.tensor(input_abs).float(), torch.tensor(abs_p3).float()

def nine_d_to_rotmat(nine_d):
    if isinstance(nine_d, np.ndarray):
        nine_d = torch.from_numpy(nine_d).float()
    
    nine_d = nine_d.to(torch.float32)
    batch_size = nine_d.shape[0]
    mats = nine_d.reshape(batch_size, 3, 3).to(torch.float32)
    u, s, vt = torch.linalg.svd(mats)
    ortho_mats = torch.bmm(u, vt)
    dets = torch.det(ortho_mats)
    u_clone = u.clone()
    for i in range(batch_size):
        if dets[i] < 0:
            u_clone[i, :, -1] = -u_clone[i, :, -1]
    return torch.bmm(u_clone, vt)

def angular_distance(pred_9d, target_9d):
    pred_9d = pred_9d.to(torch.float32)
    target_9d = target_9d.to(torch.float32)
    
    pred_rot = nine_d_to_rotmat(pred_9d)
    target_rot = nine_d_to_rotmat(target_9d)
    relative = torch.bmm(pred_rot.transpose(1, 2), target_rot)
    traces = torch.einsum('bii->b', relative)
    return torch.acos(torch.clamp((traces - 1) / 2, -1 + 1e-7, 1 - 1e-7)) * (180 / np.pi)

class RotationLoss(nn.Module):
    def __init__(self, angular_weight=0.1):
        super().__init__()
        self.angular_weight = angular_weight

    def forward(self, pred, target):
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        
        pred_rot = nine_d_to_rotmat(pred)
        target_rot = nine_d_to_rotmat(target)
        chordal_loss = torch.mean(torch.norm(pred_rot - target_rot, p='fro', dim=(1,2)) ** 2)
        angular_loss = torch.mean(angular_distance(pred, target))
        return chordal_loss + self.angular_weight * angular_loss

def plot_results(preds, labels, step, save_dir, mean, std):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'results_step_{step}.pdf')
    preds_denorm = preds * std + mean
    labels_denorm = labels * std + mean
    angular_errors = angular_distance(torch.tensor(preds_denorm), torch.tensor(labels_denorm)).numpy()
    mean_angular_error = angular_errors.mean()

    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8, 8))
        plt.hist(angular_errors, bins=50, color='blue', alpha=0.7)
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Count')
        plt.title(f'Mean Angular Error: {mean_angular_error:.4f} degrees')
        pdf.savefig()
        plt.close()

def eval_epoch(model, loader, criterion, writer, global_step, dataset_mean, dataset_std):
    model.eval()
    total_loss = 0
    preds, labels_list, inputs_list = [], [], []  # Collect inputs for logging
    with torch.no_grad():
        for img1, img2, img3, input_abs, abs_p3 in tqdm(loader, desc='Evaluating'):
            img1, img2, img3, input_abs, abs_p3 = img1.to(device), img2.to(device), img3.to(device), input_abs.to(device), abs_p3.to(device)
            with amp.autocast(device_type='cuda'):
                output = model(img1, img2, img3, input_abs)
            loss = criterion(output, abs_p3)
            total_loss += loss.item()
            preds.append(output.cpu().numpy())
            labels_list.append(abs_p3.cpu().numpy())
            inputs_list.append(input_abs.cpu().numpy())

    preds = np.vstack(preds)
    labels_list = np.vstack(labels_list)
    inputs_list = np.vstack(inputs_list)
    
    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels_list)
    ang_errors = angular_distance(preds_t, labels_t).numpy()
    mean_ang = np.mean(ang_errors)
    median_ang = np.median(ang_errors)
    percent_under_5 = np.mean(ang_errors < 5) * 100
    
    logger = logging.getLogger('rotation_training')
    logger.info(f"Validation at step {global_step}: Loss={total_loss / len(loader):.4f}, Mean Angular Error={mean_ang:.4f} deg, Median={median_ang:.4f} deg, % <5 deg={percent_under_5:.2f}")
    logger.info(f"Angular Error Stats: Min={np.min(ang_errors):.4f}, Max={np.max(ang_errors):.4f}, Std={np.std(ang_errors):.4f}")
    
    writer.add_scalar('Loss/val', total_loss / len(loader), global_step)
    writer.add_scalar('Error/mean_angular_val', mean_ang, global_step)
    writer.add_scalar('Error/median_angular_val', median_ang, global_step)
    writer.add_scalar('Error/percent_under_5', percent_under_5, global_step)
    writer.add_histogram('Angular_Errors', ang_errors, global_step)
    
    num_samples = min(3, len(preds))
    sample_indices = random.sample(range(len(preds)), num_samples)
    for i, idx in enumerate(sample_indices):
        sample_input = inputs_list[idx][:6]  # First 6 for brevity (of 18D)
        sample_pred = preds[idx][:3]
        sample_label = labels_list[idx][:3]
        sample_ang_error = ang_errors[idx]
        
        sample_pred_t = torch.tensor(preds[idx]).unsqueeze(0)
        sample_label_t = torch.tensor(labels_list[idx]).unsqueeze(0)
        sample_chordal = torch.norm(nine_d_to_rotmat(sample_pred_t) - nine_d_to_rotmat(sample_label_t), p='fro', dim=(1,2)).item() ** 2
        
        logger.info(f"Val Sample {i+1} (idx {idx}): Input_Abs_p1_p2={sample_input.tolist()}, Pred={sample_pred.tolist()}, Label={sample_label.tolist()}, Angular Error={sample_ang_error:.4f} deg, Chordal Loss={sample_chordal:.4f}")
        
        text_str = f"Sample {i+1}: Input_Abs_p1_p2={sample_input}, Pred={sample_pred}, Label={sample_label}, Angular Error={sample_ang_error:.4f} deg, Chordal Loss={sample_chordal:.4f}"
        writer.add_text(f'Samples/Val_Sample_{i+1}', text_str, global_step)
        writer.add_scalar(f'Samples/Sample_{i+1}/Angular_Error', sample_ang_error, global_step)
        writer.add_scalar(f'Samples/Sample_{i+1}/Chordal_Loss', sample_chordal, global_step)

    plot_results(preds, labels_list, global_step, output_dir, dataset_mean, dataset_std)
    return total_loss / len(loader), preds, labels_list

def main(args):
    torch._dynamo.config.suppress_errors = True

    logger = logging.getLogger('rotation_training')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('train_rotation_detailed.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info(f"Starting training with args: {args}")
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}, Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    train_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]

    if args.single_patient:
        train_dirs = [train_dirs[0]]
        val_dirs = [val_dirs[0]]

    train_dataset = RotationDataset(train_dirs, args.pairs_per_patient)
    val_dataset = RotationDataset(val_dirs, args.val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = DinoV2RotationTransformer().to(device)
    model.encoder.gradient_checkpointing = True
    
    # Freeze encoder initially
    for param in model.encoder.parameters():
        param.requires_grad = False

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)  # Simple fixed LR
    criterion = RotationLoss(angular_weight=0.1)
    scaler = amp.GradScaler()

    global output_dir
    output_dir = f"./rotation_training_runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    train_iter = iter(train_loader)
    global_step = 0
    with tqdm(total=args.total_steps, desc="Training Steps") as pbar:
        while global_step < args.total_steps:
            try:
                img1, img2, img3, input_abs, abs_p3 = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                img1, img2, img3, input_abs, abs_p3 = next(train_iter)

            img1, img2, img3, input_abs, abs_p3 = img1.to(device), img2.to(device), img3.to(device), input_abs.to(device), abs_p3.to(device)

            model.train()
            optimizer.zero_grad()
            
            start_time = time.time()
            with amp.autocast(device_type='cuda'):
                outputs = model(img1, img2, img3, input_abs)
            loss = criterion(outputs, abs_p3)
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {global_step}!")
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            batch_time = time.time() - start_time

            writer.add_scalar('Loss/train', loss.item(), global_step)
            
            logger.info(f"Step {global_step}, Train Loss: {loss.item():.4f}")
            
            if global_step % 100 == 0:
                grad_norm = torch.norm(torch.cat([p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]))
                logger.info(f"Step {global_step}, Grad Norm: {grad_norm:.4f}, Learning Rate: {args.lr:.6f}, Batch Time: {batch_time:.2f} sec")  # Fixed LR logged
                writer.add_scalar('Grad/Norm', grad_norm, global_step)
            
            if global_step % 500 == 0 and global_step > 0:
                sample_idx = random.randint(0, len(outputs) - 1)
                sample_input = input_abs[sample_idx].cpu().numpy()[:6]
                sample_pred = outputs[sample_idx].cpu().detach().numpy()[:3]
                sample_label = abs_p3[sample_idx].cpu().numpy()[:3]
                sample_ang_error = angular_distance(outputs[sample_idx].unsqueeze(0), abs_p3[sample_idx].unsqueeze(0)).item()
                logger.info(f"Train Sample (step {global_step}, idx {sample_idx}): Input_Abs_p1_p2={sample_input.tolist()}, Pred={sample_pred.tolist()}, Label={sample_label.tolist()}, Angular Error={sample_ang_error:.4f} deg")
                text_str = f"Train Sample: Input_Abs_p1_p2={sample_input}, Pred={sample_pred}, Label={sample_label}, Angular Error={sample_ang_error:.4f} deg"
                writer.add_text('Samples/Train_Sample', text_str, global_step)

            # Unfreeze encoder after 2000 steps
            if global_step == 2000:
                for param in model.encoder.parameters():
                    param.requires_grad = True
                logger.info("Unfreezing encoder at step 2000")

            global_step += 1
            pbar.update(1)

            if global_step % args.val_freq == 0 or global_step == args.total_steps:
                val_loss, preds, val_labels = eval_epoch(model, val_loader, criterion, writer, global_step, val_dataset.mean, val_dataset.std)
                writer.add_scalar('Loss/val', val_loss, global_step)
                logger.info(f"Step {global_step}, Val Loss: {val_loss:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../ct_data_rotation_only')
    parser.add_argument('--lr', type=float, default=1e-5, help='Fixed learning rate')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--total_steps', type=int, default=100000)
    parser.add_argument('--val_freq', type=int, default=1000)
    parser.add_argument('--pairs_per_patient', type=int, default=2000)
    parser.add_argument('--val_pairs', type=int, default=500)
    parser.add_argument('--single_patient', action='store_true')
    args = parser.parse_args()
    main(args)