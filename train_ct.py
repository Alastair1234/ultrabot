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
from model_ct import DinoV2PairTransformer
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse

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
            img = cv2.cvtColor(cv2.imread(os.path.join(images_dir, pt['FileName'])), cv2.COLOR_BGR2RGB) / 255.0
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

def plot_results(preds, labels, epoch, save_dir, mean, std):
    os.makedirs(save_dir, exist_ok=True)
    pdf_path = os.path.join(save_dir, f'results_epoch_{epoch}.pdf')

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

def train_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    total_loss = 0
    for img1, img2, img3, delta_p1_p2, labels in tqdm(loader, desc='Training'):
        img1, img2, img3, delta_p1_p2, labels = (
            img1.to(device), img2.to(device), img3.to(device),
            delta_p1_p2.to(device), labels.to(device)
        )
        optimizer.zero_grad()
        with amp.autocast(device_type='cuda'):
            outputs = model(img1, img2, img3, delta_p1_p2)
            loss = criterion(outputs, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
    return total_loss / len(loader)


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


def main(args):
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

    model = DinoV2PairTransformer().to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    scaler = amp.GradScaler()

    output_dir = f"./training_runs/{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    os.makedirs(output_dir, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, scaler)
        val_loss, preds, labels = eval_epoch(model, val_loader, criterion)
        print(f"Epoch {epoch}/{args.epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if epoch % args.checkpoint_freq == 0 or epoch == args.epochs:
            torch.save(model.state_dict(), os.path.join(output_dir, f"model_epoch_{epoch}.pth"))
            plot_results(preds, labels, epoch, output_dir, val_dataset.mean, val_dataset.std)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='./ct_data_random_angle')
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--lr', default=1e-5, type=float)
    parser.add_argument('--pairs_per_patient', default=2000, type=int)
    parser.add_argument('--val_pairs', default=500, type=int)
    parser.add_argument('--checkpoint_freq', default=10, type=int)
    parser.add_argument('--single_patient', action='store_true')
    args = parser.parse_args()

    main(args)
