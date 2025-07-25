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
from model_rotation_only import DinoV2RotationTransformer  # Import from the model file
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
from logging.handlers import RotatingFileHandler
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.labels = np.array([self.calc_label(p2, p3) for _, _, p2, p3 in self.data_triplets])
        self.mean = self.labels.mean(axis=0)
        self.std = self.labels.std(axis=0) + 1e-8
        
        # Log data stats (using global logger if available, else print)
        try:
            logger = logging.getLogger('rotation_training')
            angles = [np.linalg.norm(label) for label in self.labels]  # Rough magnitude
            logger.info(f"Dataset stats: Samples={len(self)}, Mean label norm={np.mean(angles):.4f}, Std={np.std(angles):.4f}, Min/Max={np.min(angles):.4f}/{np.max(angles):.4f}")
            logger.info(f"Label mean={self.mean.tolist()}, std={self.std.tolist()}")
        except:
            print("Logger not initialized yet for dataset stats.")

    def quat_to_9d(self, quat):
        return R.from_quat(quat).as_matrix().reshape(9)

    def calc_label(self, p1, p2):
        q1 = np.array([p1['RotationQuaternion'][k] for k in 'xyzw'])
        q2 = np.array([p2['RotationQuaternion'][k] for k in 'xyzw'])
        quat_diff = (R.from_quat(q2) * R.from_quat(q1).inv()).as_quat()
        return self.quat_to_9d(quat_diff)

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]

        def load_img(pt):
            img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0
            return torch.tensor(img).permute(2, 0, 1).float()

        img1, img2, img3 = load_img(p1), load_img(p2), load_img(p3)
        delta_p1_p2 = (self.calc_label(p1, p2) - self.mean) / self.std
        label_p2_p3 = (self.calc_label(p2, p3) - self.mean) / self.std
        return img1, img2, img3, torch.tensor(delta_p1_p2).float(), torch.tensor(label_p2_p3).float()

def nine_d_to_rotmat(nine_d):
    if isinstance(nine_d, np.ndarray):
        nine_d = torch.from_numpy(nine_d).float()
    
    # Explicitly cast to fp32 to avoid fp16 errors in SVD/det
    nine_d = nine_d.to(torch.float32)
    
    batch_size = nine_d.shape[0]
    mats = nine_d.reshape(batch_size, 3, 3).to(torch.float32)  # Ensure reshape is in fp32
    u, s, vt = torch.linalg.svd(mats)
    ortho_mats = torch.bmm(u, vt)
    dets = torch.det(ortho_mats)
    u_clone = u.clone()
    for i in range(batch_size):
        if dets[i] < 0:
            u_clone[i, :, -1] = -u_clone[i, :, -1]
    return torch.bmm(u_clone, vt)

def angular_distance(pred_9d, target_9d):
    # Explicitly cast to fp32 to avoid fp16 errors
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
        # Cast inputs to fp32 here as well for safety
        pred = pred.to(torch.float32)
        target = target.to(torch.float32)
        
        pred_rot = nine_d_to_rotmat(pred)
        target_rot = nine_d_to_rotmat(target)  # Target is already orthogonal
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
    preds, labels_list, deltas_list = [], [], []  # Collect deltas for logging
    with torch.no_grad():
        for img1, img2, img3, delta_p1_p2, labels in tqdm(loader, desc='Evaluating'):
            img1, img2, img3, delta_p1_p2, labels = img1.to(device), img2.to(device), img3.to(device), delta_p1_p2.to(device), labels.to(device)
            with amp.autocast(device_type='cuda'):  # Keep for model forward (efficient)
                output = model(img1, img2, img3, delta_p1_p2)
            loss = criterion(output, labels)  # Compute loss OUTSIDE autocast for fp32 safety
            total_loss += loss.item()
            preds.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            deltas_list.append(delta_p1_p2.cpu().numpy())  # Collect deltas

    # Stack collected data
    preds = np.vstack(preds)
    labels_list = np.vstack(labels_list)
    deltas_list = np.vstack(deltas_list)
    
    # Calculate detailed metrics
    preds_t = torch.tensor(preds)
    labels_t = torch.tensor(labels_list)
    ang_errors = angular_distance(preds_t, labels_t).numpy()
    mean_ang = np.mean(ang_errors)
    median_ang = np.median(ang_errors)
    percent_under_5 = np.mean(ang_errors < 5) * 100
    
    # Log to file
    logger = logging.getLogger('rotation_training')
    logger.info(f"Validation at step {global_step}: Loss={total_loss / len(loader):.4f}, Mean Angular Error={mean_ang:.4f} deg, Median={median_ang:.4f} deg, % <5 deg={percent_under_5:.2f}")
    logger.info(f"Angular Error Stats: Min={np.min(ang_errors):.4f}, Max={np.max(ang_errors):.4f}, Std={np.std(ang_errors):.4f}")
    
    # Log to TensorBoard
    writer.add_scalar('Loss/val', total_loss / len(loader), global_step)
    writer.add_scalar('Error/mean_angular_val', mean_ang, global_step)
    writer.add_scalar('Error/median_angular_val', median_ang, global_step)
    writer.add_scalar('Error/percent_under_5', percent_under_5, global_step)
    writer.add_histogram('Angular_Errors', ang_errors, global_step)
    
    # Log 3 individual examples for debugging (random samples)
    num_samples = min(3, len(preds))  # Up to 3
    sample_indices = random.sample(range(len(preds)), num_samples)
    for i, idx in enumerate(sample_indices):
        sample_delta = deltas_list[idx][:3]  # First 3 elements for brevity
        sample_pred = preds[idx][:3]
        sample_label = labels_list[idx][:3]
        sample_ang_error = ang_errors[idx]
        
        # Per-sample chordal loss (for debug)
        sample_pred_t = torch.tensor(preds[idx]).unsqueeze(0)
        sample_label_t = torch.tensor(labels_list[idx]).unsqueeze(0)
        sample_chordal = torch.norm(nine_d_to_rotmat(sample_pred_t) - nine_d_to_rotmat(sample_label_t), p='fro', dim=(1,2)).item() ** 2
        
        # Log to file
        logger.info(f"Val Sample {i+1} (idx {idx}): Delta_p1_p2={sample_delta.tolist()}, Pred={sample_pred.tolist()}, Label={sample_label.tolist()}, Angular Error={sample_ang_error:.4f} deg, Chordal Loss={sample_chordal:.4f}")
        
        # Log to TensorBoard (text for details, scalars for metrics)
        text_str = f"Sample {i+1}: Delta_p1_p2={sample_delta}, Pred={sample_pred}, Label={sample_label}, Angular Error={sample_ang_error:.4f} deg, Chordal Loss={sample_chordal:.4f}"
        writer.add_text(f'Samples/Val_Sample_{i+1}', text_str, global_step)
        writer.add_scalar(f'Samples/Sample_{i+1}/Angular_Error', sample_ang_error, global_step)
        writer.add_scalar(f'Samples/Sample_{i+1}/Chordal_Loss', sample_chordal, global_step)

    plot_results(preds, labels_list, global_step, output_dir, dataset_mean, dataset_std)  # Assuming output_dir is global or passed
    return total_loss / len(loader), preds, labels_list

def main(args):
    torch._dynamo.config.suppress_errors = True

    # Setup logging
    global logger  # For access in other functions if needed
    logger = logging.getLogger('rotation_training')
    logger.setLevel(logging.INFO)
    handler = RotatingFileHandler('train_rotation_detailed.log', maxBytes=10*1024*1024, backupCount=5)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    # Log setup info
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
    model.encoder.gradient_checkpointing = True  # Save memory
    
    # Disabled due to compilation errors (e.g., dtype mismatches); re-enable if fixed
    # model = torch.compile(model)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = RotationLoss(angular_weight=0.1)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500)
    scaler = amp.GradScaler()

    global output_dir  # For plot_results access
    output_dir = f"./rotation_training_runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(output_dir, 'logs'))

    train_iter = iter(train_loader)
    global_step = 0
    with tqdm(total=args.total_steps, desc="Training Steps") as pbar:
        while global_step < args.total_steps:
            try:
                img1, img2, img3, delta_p1_p2, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                img1, img2, img3, delta_p1_p2, labels = next(train_iter)

            img1, img2, img3, delta_p1_p2, labels = img1.to(device), img2.to(device), img3.to(device), delta_p1_p2.to(device), labels.to(device)

            model.train()
            optimizer.zero_grad()
            
            start_time = time.time()
            with amp.autocast(device_type='cuda'):  # Keep for model forward (efficient)
                outputs = model(img1, img2, img3, delta_p1_p2)
            loss = criterion(outputs, labels)  # Compute loss OUTSIDE autocast for fp32 safety
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at step {global_step}!")
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            batch_time = time.time() - start_time

            writer.add_scalar('Loss/train', loss.item(), global_step)
            
            # Log per-step
            logger.info(f"Step {global_step}, Train Loss: {loss.item():.4f}")
            
            # Every 100 steps: Grad norm, LR, timing
            if global_step % 100 == 0:
                grad_norm = torch.norm(torch.cat([p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]))
                logger.info(f"Step {global_step}, Grad Norm: {grad_norm:.4f}, Learning Rate: {optimizer.param_groups[0]['lr']:.6f}, Batch Time: {batch_time:.2f} sec")
                writer.add_scalar('Grad/Norm', grad_norm, global_step)
            
            # Every 500 steps: Log 1 random train sample for debug
            if global_step % 500 == 0 and global_step > 0:
                sample_idx = random.randint(0, len(outputs) - 1)
                sample_delta = delta_p1_p2[sample_idx].cpu().numpy()[:3]
                sample_pred = outputs[sample_idx].cpu().detach().numpy()[:3]
                sample_label = labels[sample_idx].cpu().numpy()[:3]
                sample_ang_error = angular_distance(outputs[sample_idx].unsqueeze(0), labels[sample_idx].unsqueeze(0)).item()
                logger.info(f"Train Sample (step {global_step}, idx {sample_idx}): Delta_p1_p2={sample_delta.tolist()}, Pred={sample_pred.tolist()}, Label={sample_label.tolist()}, Angular Error={sample_ang_error:.4f} deg")
                text_str = f"Train Sample: Delta_p1_p2={sample_delta}, Pred={sample_pred}, Label={sample_label}, Angular Error={sample_ang_error:.4f} deg"
                writer.add_text('Samples/Train_Sample', text_str, global_step)

            global_step += 1
            pbar.update(1)

            if global_step % args.val_freq == 0 or global_step == args.total_steps:
                val_loss, preds, val_labels = eval_epoch(model, val_loader, criterion, writer, global_step, val_dataset.mean, val_dataset.std)
                writer.add_scalar('Loss/val', val_loss, global_step)  # Already in eval_epoch, but kept for completeness
                logger.info(f"Step {global_step}, Val Loss: {val_loss:.4f}")
                scheduler.step(val_loss)
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            if global_step % args.checkpoint_freq == 0 or global_step == args.total_steps:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_step_{global_step}.pth"))

    writer.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./ct_data_rotation_only')
    parser.add_argument('--total_steps', default=500000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--pairs_per_patient', default=2000, type=int)
    parser.add_argument('--val_pairs', default=500, type=int)
    parser.add_argument('--checkpoint_freq', default=10000, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--single_patient', action='store_true')
    args = parser.parse_args()
    main(args)