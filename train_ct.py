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
from model_ct import DinoV2PairTransformer  # Import from model_ct.py
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from torch.utils.tensorboard import SummaryWriter  # For TensorBoard logging
from torch.optim.lr_scheduler import ReduceLROnPlateau  # Added for LR scheduling

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CTDataset(Dataset):
    def __init__(self, patient_dirs, pairs_per_patient=500):
        self.data_triplets = []
        for patient_dir in tqdm(patient_dirs, desc="Loading patient data"):
            info_path = os.path.join(patient_dir, 'info.json')
            images_dir = os.path.join(patient_dir, 'images')

            if not os.path.exists(info_path) or not os.path.isdir(images_dir):
                print(f"Skipping {patient_dir}, missing data.")
                continue

            with open(info_path) as f:
                points = json.load(f)['PointInfos']

            if len(points) < 3:
                continue

            for _ in range(pairs_per_patient):
                p1, p2, p3 = random.sample(points, 3)
                self.data_triplets.append((images_dir, p1, p2, p3))

        assert len(self.data_triplets) > 0, "No data triplets created."

        self.labels = np.array([self.calc_label(p2, p3) for _, _, p2, p3 in self.data_triplets])
        self.mean = self.labels.mean(axis=0)
        self.std = self.labels.std(axis=0) + 1e-8

        print(f"Dataset normalization mean: {self.mean}, std: {self.std}")

    def quat_to_9d(self, quat):
        """Convert quaternion [x,y,z,w] to 9D flattened rotation matrix."""
        rot = R.from_quat(quat).as_matrix()  # 3x3 rotation matrix
        nine_d = rot.reshape(9)  # Flatten to [r00, r01, r02, r10, ...]
        return nine_d

    def calc_label(self, p1, p2):
        q1 = np.array([
            p1['RotationQuaternion']['x'], p1['RotationQuaternion']['y'],
            p1['RotationQuaternion']['z'], p1['RotationQuaternion']['w']
        ])
        q2 = np.array([
            p2['RotationQuaternion']['x'], p2['RotationQuaternion']['y'],
            p2['RotationQuaternion']['z'], p2['RotationQuaternion']['w']
        ])

        quat_diff = (R.from_quat(q2) * R.from_quat(q1).inv()).as_quat()
        return self.quat_to_9d(quat_diff)  # Convert to 9D (rotation only)

    def __len__(self):
        return len(self.data_triplets)

    def __getitem__(self, idx):
        images_dir, p1, p2, p3 = self.data_triplets[idx]

        def load_img(pt):
            img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (224, 224)) / 255.0  # Resize to 224x224 (if not already)
            return torch.tensor(img).permute(2, 0, 1).float()

        img1 = load_img(p1)
        img2 = load_img(p2)
        img3 = load_img(p3)

        delta_p1_p2 = (self.calc_label(p1, p2) - self.mean) / self.std
        label_p2_p3 = (self.calc_label(p2, p3) - self.mean) / self.std

        return img1, img2, img3, torch.tensor(delta_p1_p2).float(), torch.tensor(label_p2_p3).float()

def nine_d_to_rotmat(nine_d):
    """Convert batched 9D vectors [batch, 9] to orthogonal 3x3 rotation matrices [batch, 3, 3] using SVD."""
    if isinstance(nine_d, np.ndarray):
        nine_d = torch.from_numpy(nine_d).float()
    
    batch_size = nine_d.shape[0]
    mats = nine_d.reshape(batch_size, 3, 3)  # [batch, 3, 3]
    
    # SVD for each matrix in the batch
    u, s, vt = torch.linalg.svd(mats)  # u [batch, 3, 3], s [batch, 3], vt [batch, 3, 3]
    ortho_mats = torch.bmm(u, vt)  # u @ vt for each batch
    
    # Ensure det=1 (proper rotation) for each
    dets = torch.det(ortho_mats)
    signs = torch.sign(dets).view(batch_size, 1, 1)
    ortho_mats = ortho_mats * signs  # Flip if needed (simple approximation; for exact, adjust u last col)
    
    return ortho_mats

def angular_distance(n1, n2):
    """Angular distance between two 9D vectors (convert to rotmats first). Assumes single pair; vectorize if needed."""
    r1 = R.from_matrix(nine_d_to_rotmat(n1[None, :])[0].numpy())  # Add batch dim and squeeze
    r2 = R.from_matrix(nine_d_to_rotmat(n2[None, :])[0].numpy())
    relative_rotation = r1.inv() * r2
    return relative_rotation.magnitude() * (180 / np.pi)

def plot_results(preds, labels, step, save_dir, mean, std):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'results_step_{step}.pdf')

    preds_denorm = preds * std + mean
    labels_denorm = labels * std + mean

    angular_errors = np.array([angular_distance(t, p) for t, p in zip(labels_denorm, preds_denorm)])
    mean_angular_error = angular_errors.mean()

    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8, 8))
        plt.hist(angular_errors, bins=50, color='green', alpha=0.7)
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Count')
        plt.title(f'Mean Angular Error: {mean_angular_error:.4f} degrees')
        plt.grid(True)
        pdf.savefig()
        plt.close()

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    preds, labels_list = [], []
    with torch.no_grad():
        for img1, img2, img3, delta_p1_p2, labels in tqdm(loader, desc='Evaluating'):
            img1, img2, img3, delta_p1_p2, labels = (
                img1.to(device), img2.to(device), img3.to(device),
                delta_p1_p2.to(device), labels.to(device)
            )
            with amp.autocast(device_type='cuda'):
                output = model(img1, img2, img3, delta_p1_p2)
                loss = criterion(output, labels)
            total_loss += loss.item()
            preds.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
    return total_loss / len(loader), np.vstack(preds), np.vstack(labels_list)

class CustomPoseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        # Simple MSE on raw 9D vectors (position-like, Euclidean)
        mse_loss = nn.functional.mse_loss(pred, target)
        
        # Optional: Add MSE on orthogonalized rotmats for stability (batched)
        pred_rot = nine_d_to_rotmat(pred)  # [batch, 3, 3]
        target_rot = nine_d_to_rotmat(target)  # [batch, 3, 3]
        rot_loss = nn.functional.mse_loss(pred_rot, target_rot)  # Frobenius-like
        
        return mse_loss + 0.5 * rot_loss  # Weighted sum (adjust weight or set to 0 for pure MSE)

def main(args):
    # Suppress Torch Dynamo errors (fallback to eager if compile fails)
    torch._dynamo.config.suppress_errors = True

    train_dirs = [os.path.join(args.data_root, 'train', d) for d in os.listdir(os.path.join(args.data_root, 'train'))]
    val_dirs = [os.path.join(args.data_root, 'val', d) for d in os.listdir(os.path.join(args.data_root, 'val'))]

    if args.single_patient:
        train_dirs = [train_dirs[0]]
        val_dirs = [val_dirs[0]]
        print(f"Single patient mode: train={train_dirs[0]}, val={val_dirs[0]}")

    train_dataset = CTDataset(train_dirs, args.pairs_per_patient)
    val_dataset = CTDataset(val_dirs, args.val_pairs)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    model = DinoV2PairTransformer(
        vision_model='facebook/webssl-dino300m-full2b-224'  # Aligned to match
    ).to(device)

    # Enable gradient checkpointing to mitigate OOM with larger inputs/model
    model.encoder.gradient_checkpointing = True

    model = torch.compile(model)  # Optional; comment out if issues persist

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CustomPoseLoss()  # Updated for 9D
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)  # New LR scheduler
    scaler = amp.GradScaler()

    output_dir = f"./training_runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, 'logs')
    writer = SummaryWriter(log_dir=log_dir)  # TensorBoard writer

    # Infinite train iterator
    train_iter = iter(train_loader)

    global_step = 0
    with tqdm(total=args.total_steps, desc="Training Steps") as pbar:
        while global_step < args.total_steps:
            try:
                img1, img2, img3, delta_p1_p2, labels = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)  # Reset iterator for continuous training
                img1, img2, img3, delta_p1_p2, labels = next(train_iter)

            img1, img2, img3, delta_p1_p2, labels = (
                img1.to(device), img2.to(device), img3.to(device),
                delta_p1_p2.to(device), labels.to(device)
            )

            model.train()
            optimizer.zero_grad()
            with amp.autocast(device_type='cuda'):
                outputs = model(img1, img2, img3, delta_p1_p2)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping for stability
            scaler.step(optimizer)
            scaler.update()

            # Log train loss to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            # Optional: Log rotation loss (it's the only component)
            # writer.add_scalar('Loss/train_rot', loss.item(), global_step)

            global_step += 1
            pbar.update(1)

            # Validation every val_freq steps
            if global_step % args.val_freq == 0 or global_step == args.total_steps:
                val_loss, preds, val_labels = eval_epoch(model, val_loader, criterion)
                writer.add_scalar('Loss/val', val_loss, global_step)
                print(f"Step {global_step}/{args.total_steps}, Val Loss: {val_loss:.4f}")
                plot_results(preds, val_labels, global_step, output_dir, val_dataset.mean, val_dataset.std)
                
                # Step the scheduler with val_loss
                scheduler.step(val_loss)
                # Log current learning rate
                writer.add_scalar('LR', optimizer.param_groups[0]['lr'], global_step)

            # Checkpoint every checkpoint_freq steps
            if global_step % args.checkpoint_freq == 0 or global_step == args.total_steps:
                torch.save(model.state_dict(), os.path.join(output_dir, f"model_step_{global_step}.pth"))

    writer.close()  # Close TensorBoard writer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../ct_data_random_angle')
    parser.add_argument('--total_steps', default=500000, type=int)  # New: total training steps
    parser.add_argument('--batch_size', default=16, type=int)  # Reduced default for safety
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--pairs_per_patient', default=2000, type=int)
    parser.add_argument('--val_pairs', default=500, type=int)
    parser.add_argument('--checkpoint_freq', default=10000, type=int)  # Save every N steps
    parser.add_argument('--val_freq', default=1000, type=int)  # Validate every N steps
    parser.add_argument('--single_patient', action='store_true')
    args = parser.parse_args()

    main(args)
