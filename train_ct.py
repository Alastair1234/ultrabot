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
from model_ct import DinoV2PairTransformer  # Import from your separate model_ct.py
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torchvision.utils as vutils  # For image grids in logging

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

    def calc_label(self, p1, p2):
        pos1 = np.array([p1['Position']['x'], p1['Position']['y'], p1['Position']['z']])
        pos2 = np.array([p2['Position']['x'], p2['Position']['y'], p2['Position']['z']])
        q1 = np.array([
            p1['RotationQuaternion']['x'], p1['RotationQuaternion']['y'],
            p1['RotationQuaternion']['z'], p1['RotationQuaternion']['w']
        ])
        q2 = np.array([
            p2['RotationQuaternion']['x'], p2['RotationQuaternion']['y'],
            p2['RotationQuaternion']['z'], p2['RotationQuaternion']['w']
        ])

        pos_diff = pos2 - pos1
        quat_diff = (R.from_quat(q2) * R.from_quat(q1).inv()).as_quat()
        return np.concatenate([pos_diff, quat_diff])

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

def angular_distance(q1, q2):
    r1, r2 = R.from_quat(q1), R.from_quat(q2)
    relative_rotation = r1.inv() * r2
    return relative_rotation.magnitude() * (180 / np.pi)

def plot_results(preds, labels, step, save_dir, mean, std):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'results_step_{step}.pdf')

    preds_denorm = preds * std + mean
    labels_denorm = labels * std + mean

    pred_pos, true_pos = preds_denorm[:, :3], labels_denorm[:, :3]
    pred_quat, true_quat = preds_denorm[:, 3:], labels_denorm[:, 3:]

    pos_rmse = np.sqrt(np.mean((true_pos - pred_pos)**2, axis=0))
    angular_errors = np.array([angular_distance(t, p) for t, p in zip(true_quat, pred_quat)])
    mean_angular_error = angular_errors.mean()

    with PdfPages(pdf_path) as pdf:
        plt.figure(figsize=(8, 8))
        plt.scatter(true_pos[:, 0], pred_pos[:, 0], alpha=0.3)
        plt.plot([true_pos.min(), true_pos.max()], [true_pos.min(), true_pos.max()], 'r--')
        plt.xlabel('True Position X')
        plt.ylabel('Predicted Position X')
        plt.title(f'Positional RMSE X: {pos_rmse[0]:.4f} m')
        plt.grid(True)
        pdf.savefig()
        plt.close()

        plt.figure(figsize=(8, 8))
        plt.hist(angular_errors, bins=50, color='green', alpha=0.7)
        plt.xlabel('Angular Error (degrees)')
        plt.ylabel('Count')
        plt.title(f'Mean Angular Error: {mean_angular_error:.4f} degrees')
        plt.grid(True)
        pdf.savefig()
        plt.close()

class CustomPoseLoss(nn.Module):
    def __init__(self, pos_weight=1.0, rot_weight=20.0):
        super().__init__()
        self.mse = nn.MSELoss()
        self.pos_weight = pos_weight
        self.rot_weight = rot_weight

    def forward(self, pred, target):
        pred_pos, pred_quat = pred[:, :3], pred[:, 3:]
        target_pos, target_quat = target[:, :3], target[:, 3:]

        # Normalize quats (in case of drift)
        pred_quat = nn.functional.normalize(pred_quat, p=2, dim=1)
        target_quat = nn.functional.normalize(target_quat, p=2, dim=1)

        # Position: MSE
        pos_loss = self.mse(pred_pos, target_pos)

        # Rotation: Geodesic with antipodal handling (min dist to q or -q)
        dot = torch.sum(pred_quat * target_quat, dim=1)
        dot_neg = torch.sum(pred_quat * (-target_quat), dim=1)
        dot = torch.max(dot.abs(), dot_neg.abs())  # Abs and max for antipodal equivalence
        dot = torch.clamp(dot, -1.0, 1.0)  # Numerical stability
        angle = 2 * torch.acos(dot)
        rot_loss = angle.mean()  # Mean angular error in radians

        return self.pos_weight * pos_loss + self.rot_weight * rot_loss

def eval_epoch(model, loader, criterion, writer=None, step=0, dataset=None, num_examples=5):
    model.eval()
    total_loss = 0
    preds, labels_list = [], []
    example_data = []  # For high-error examples

    with torch.no_grad():
        for batch_idx, (img1, img2, img3, delta_p1_p2, labels) in enumerate(tqdm(loader, desc='Evaluating')):
            img1, img2, img3, delta_p1_p2, labels = (
                img1.to(device), img2.to(device), img3.to(device),
                delta_p1_p2.to(device), labels.to(device)
            )
            with amp.autocast(device_type='cuda'):
                output = model(img1, img2, img3, delta_p1_p2)
                batch_loss = criterion(output, labels)

                # Compute per-sample losses for sorting
                per_sample_losses = torch.stack([
                    criterion(output[i].unsqueeze(0), labels[i].unsqueeze(0)) for i in range(output.size(0))
                ]).cpu().numpy()

            total_loss += batch_loss.item()
            preds.append(output.cpu().numpy())
            labels_list.append(labels.cpu().numpy())

            # Collect example data
            for i in range(output.size(0)):
                example_data.append({
                    'img1': img1[i].cpu(), 'img2': img2[i].cpu(), 'img3': img3[i].cpu(),
                    'delta_p1_p2': delta_p1_p2[i].cpu().numpy(),
                    'pred': output[i].cpu().numpy(),
                    'label': labels[i].cpu().numpy(),
                    'loss': per_sample_losses[i]
                })

    avg_loss = total_loss / len(loader)
    preds_np = np.vstack(preds)
    labels_np = np.vstack(labels_list)

    # Compute additional metrics
    preds_denorm = preds_np * dataset.std + dataset.mean
    labels_denorm = labels_np * dataset.std + dataset.mean
    pos_rmse = np.sqrt(np.mean((labels_denorm[:, :3] - preds_denorm[:, :3])**2))
    angular_errors = np.array([angular_distance(labels_denorm[i, 3:], preds_denorm[i, 3:]) for i in range(len(labels_denorm))])
    mean_angular = np.mean(angular_errors)

    # Sort and log top N high-error examples to TensorBoard
    if writer:
        example_data.sort(key=lambda x: x['loss'], reverse=True)
        top_examples = example_data[:num_examples]
        for idx, ex in enumerate(top_examples):
            # Denormalize
            pred_denorm = ex['pred'] * dataset.std + dataset.mean
            label_denorm = ex['label'] * dataset.std + dataset.mean
            delta_denorm = ex['delta_p1_p2'] * dataset.std + dataset.mean

            # Per-example metrics
            ex_pos_rmse = np.sqrt(np.mean((label_denorm[:3] - pred_denorm[:3])**2))
            ex_angular_err = angular_distance(label_denorm[3:], pred_denorm[3:])

            # Text log
            text = (
                f"Example {idx+1} (Loss: {ex['loss']:.4f})\n"
                f"Delta P1-P2: {delta_denorm}\n"
                f"True Label (P2-P3): Pos {label_denorm[:3]}, Quat {label_denorm[3:]}\n"
                f"Pred: Pos {pred_denorm[:3]}, Quat {pred_denorm[3:]}\n"
                f"Pos RMSE: {ex_pos_rmse:.4f}, Angular Err: {ex_angular_err:.4f}°"
            )
            writer.add_text(f"Val_Examples/Text_{idx+1}", text, step)

            # Image grid log (img1 | img2 | img3)
            img_grid = vutils.make_grid([ex['img1'], ex['img2'], ex['img3']], nrow=3, normalize=True)
            writer.add_image(f"Val_Examples/Images_{idx+1}", img_grid, step)

    return avg_loss, preds_np, labels_np, pos_rmse, mean_angular

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
    model.encoder.gradient_checkpointing = True  # Assuming encoder is part of model
    
    model = torch.compile(model)  # Optional; comment out if issues persist

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = CustomPoseLoss(pos_weight=1.0, rot_weight=args.rot_weight)  # Updated with arg
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=500, verbose=True)
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

            # Log train loss and separate pos/rot to TensorBoard
            writer.add_scalar('Loss/train', loss.item(), global_step)
            pos_loss = criterion.mse(outputs[:, :3], labels[:, :3])
            pred_quat = nn.functional.normalize(outputs[:, 3:], dim=1)
            target_quat = nn.functional.normalize(labels[:, 3:], dim=1)
            dot = torch.max(
                torch.sum(pred_quat * target_quat, dim=1).abs(),
                torch.sum(pred_quat * (-target_quat), dim=1).abs()
            )
            dot = torch.clamp(dot, -1.0, 1.0)
            rot_loss = (2 * torch.acos(dot)).mean()
            writer.add_scalar('Loss/train_pos', pos_loss.item(), global_step)
            writer.add_scalar('Loss/train_rot', rot_loss.item(), global_step)

            global_step += 1
            pbar.update(1)

            # Validation every val_freq steps
            if global_step % args.val_freq == 0 or global_step == args.total_steps:
                val_loss, preds, val_labels, pos_rmse, mean_angular = eval_epoch(
                    model, val_loader, criterion, writer=writer, step=global_step, dataset=val_dataset
                )
                writer.add_scalar('Loss/val', val_loss, global_step)
                writer.add_scalar('Metrics/pos_rmse', pos_rmse, global_step)
                writer.add_scalar('Metrics/mean_angular_error', mean_angular, global_step)
                print(f"Step {global_step}/{args.total_steps}, Val Loss: {val_loss:.4f}, Pos RMSE: {pos_rmse:.4f}, Angular Error: {mean_angular:.4f}")
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
    parser.add_argument('--total_steps', default=500000, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--pairs_per_patient', default=2000, type=int)
    parser.add_argument('--val_pairs', default=500, type=int)
    parser.add_argument('--checkpoint_freq', default=10000, type=int)
    parser.add_argument('--val_freq', default=1000, type=int)
    parser.add_argument('--single_patient', action='store_true')
    parser.add_argument('--rot_weight', default=20.0, type=float)  # New arg for rot emphasis
    args = parser.parse_args()

    main(args)
