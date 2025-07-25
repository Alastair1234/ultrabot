import os
import torch
import numpy as np
import cv2
import json
import pydicom
import scipy.ndimage as ndi
from scipy.spatial.transform import Rotation as R
from sklearn.model_selection import train_test_split
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def find_dicom_dirs(root_dir):
    return [
        dirpath for dirpath, _, files in os.walk(root_dir)
        if any(f.lower().endswith('.dcm') for f in files)
    ]

def robust_load_ct_volume(dicom_dir):
    files = [f for f in sorted(os.listdir(dicom_dir)) if f.lower().endswith('.dcm')]
    slices = [pydicom.dcmread(os.path.join(dicom_dir, f)) for f in files]
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    thickness = float(getattr(slices[0], 'SliceThickness', 1.0) or 1.0)
    spacing = getattr(slices[0], 'PixelSpacing', [1.0, 1.0]) or [1.0, 1.0]
    spacing_arr = np.array([thickness, *map(float, spacing)], dtype=np.float32)

    volume_pixels = np.stack([s.pixel_array for s in slices], axis=0)
    slope = float(getattr(slices[0], 'RescaleSlope', 1.0))
    intercept = float(getattr(slices[0], 'RescaleIntercept', -1024.0))
    volume_hu = volume_pixels.astype(np.float32) * slope + intercept
    
    return volume_hu, spacing_arr

def find_body_center(volume_hu, threshold_hu=-500):
    body_mask = volume_hu > threshold_hu
    center_of_mass_voxels = ndi.center_of_mass(body_mask)
    return np.array(center_of_mass_voxels)

def dicom_to_unity_coords(pos_mm):
    dicom_x, dicom_y, dicom_z = pos_mm[2], pos_mm[1], pos_mm[0]
    unity_x = -dicom_x
    unity_y = dicom_z
    unity_z = dicom_y
    return np.array([unity_x, unity_y, unity_z]) / 1000.0

def generate_slice(volume, spacing, body_center_voxels, size=(224, 224), 
                   physical_size_multiplier=3.0):  # Increased for more FOV variety
    # FIXED CENTER: No offset, all at body center
    center_mm = body_center_voxels * spacing

    # Fully random rotation
    rotation = R.random()
    rot_matrix = rotation.as_matrix()
    right = rot_matrix[:, 0]
    up = rot_matrix[:, 1]

    # Sampling grid
    physical_width = size[0] * physical_size_multiplier
    physical_height = size[1] * physical_size_multiplier
    grid_x, grid_y = np.meshgrid(
        np.linspace(-physical_width / 2, physical_width / 2, size[0]),
        np.linspace(-physical_height / 2, physical_height / 2, size[1]),
        indexing='ij'
    )

    points_mm = center_mm[:, None, None] + grid_x * right[:, None, None] + grid_y * up[:, None, None]
    points_voxels = points_mm / spacing[:, None, None]

    slice_img = ndi.map_coordinates(
        volume, points_voxels, order=1, mode='constant', cval=volume.min()
    ).reshape(size)

    # Normalize to RGB (soft tissue window)
    window_min, window_max = -150, 250
    slice_img_clipped = np.clip(slice_img, window_min, window_max)
    slice_img_norm = (slice_img_clipped - window_min) / (window_max - window_min)
    img_rgb = cv2.cvtColor((slice_img_norm * 255).astype(np.uint8), cv2.COLOR_GRAY2RGB)

    pos_unity = dicom_to_unity_coords(center_mm)
    quat_unity = rotation.as_quat()  # [x, y, z, w]

    return img_rgb, pos_unity.tolist(), quat_unity.tolist()

def create_patient_slices(root_dir, output_root, slices_per_patient=5000, test_size=0.3, 
                          test_mode=False, physical_size_multiplier=3.0):
    dicom_dirs = find_dicom_dirs(root_dir)
    
    if test_mode:
        dicom_dirs = dicom_dirs[:1]
        test_dir = os.path.join(output_root, 'test_output')
        os.makedirs(test_dir, exist_ok=True)
        patient_name = os.path.basename(dicom_dirs[0])
        patient_dir = os.path.join(test_dir, patient_name)
        images_dir = os.path.join(patient_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)

        volume, spacing = robust_load_ct_volume(dicom_dirs[0])
        body_center = find_body_center(volume)

        info = {"PointInfos": []}
        for i in tqdm(range(slices_per_patient), desc=f'Generating slices for {patient_name}'):
            img, pos, quat = generate_slice(volume, spacing, body_center, physical_size_multiplier=physical_size_multiplier)
            img_filename = f'capture_{i}.jpg'
            cv2.imwrite(os.path.join(images_dir, img_filename), img)
            info["PointInfos"].append({
                "FileName": img_filename,
                "Position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                "RotationQuaternion": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]}
            })
        with open(os.path.join(patient_dir, 'info.json'), 'w') as f:
            json.dump(info, f, indent=2)
        return

    train_dirs, temp_dirs = train_test_split(dicom_dirs, test_size=test_size, random_state=42)
    val_dirs, test_dirs = train_test_split(temp_dirs, test_size=0.5, random_state=42)
    splits = {'train': train_dirs, 'val': val_dirs, 'test': test_dirs}

    for split_name, dirs in splits.items():
        split_dir = os.path.join(output_root, split_name)
        os.makedirs(split_dir, exist_ok=True)
        for dicom_dir in tqdm(dirs, desc=f'Processing {split_name}'):
            patient_name = os.path.basename(dicom_dir)
            patient_dir = os.path.join(split_dir, patient_name)
            images_dir = os.path.join(patient_dir, 'images')
            os.makedirs(images_dir, exist_ok=True)

            volume, spacing = robust_load_ct_volume(dicom_dir)
            body_center = find_body_center(volume)
            
            info = {"PointInfos": []}
            for i in range(slices_per_patient):
                img, pos, quat = generate_slice(volume, spacing, body_center, physical_size_multiplier=physical_size_multiplier)
                img_filename = f'capture_{i}.jpg'
                cv2.imwrite(os.path.join(images_dir, img_filename), img)
                info["PointInfos"].append({
                    "FileName": img_filename,
                    "Position": {"x": pos[0], "y": pos[1], "z": pos[2]},
                    "RotationQuaternion": {"x": quat[0], "y": quat[1], "z": quat[2], "w": quat[3]}
                })
            with open(os.path.join(patient_dir, 'info.json'), 'w') as f:
                json.dump(info, f, indent=2)

if __name__ == "__main__":
    dicom_root = "C:/Users/alast/Desktop/manifest-1599750808610/Pancreas-CT"  # Adjust to your path
    output_dir = "./ct_data_rotation_only"
    create_patient_slices(dicom_root, output_dir, slices_per_patient=50)  # Run to generate dataset