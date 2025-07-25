# ==============================================================================
# File: calculate_normalization_stats.py
#
# Description:
# Calculate normalization statistics from local ct_data_random_angle dataset
# ==============================================================================

import os
import json
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

def get_quat(pt_info): 
    return np.array([pt_info['RotationQuaternion'][k] for k in 'xyzw'])

def get_position(pt_info):
    return np.array([pt_info['Position'][k] for k in 'xyz'])

def quat_to_9d(quat): 
    return R.from_quat(quat).as_matrix().reshape(9)

def calculate_stats_from_dataset(data_root, subset='val', max_patients=None):
    """Calculate normalization statistics from the dataset"""
    
    subset_dir = os.path.join(data_root, subset)
    if not os.path.exists(subset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {subset_dir}")
    
    patient_dirs = [os.path.join(subset_dir, d) for d in os.listdir(subset_dir)]
    if max_patients:
        patient_dirs = patient_dirs[:max_patients]
    
    print(f"Processing {len(patient_dirs)} patients from {subset} set...")
    
    all_rotations = []
    all_positions = []
    
    for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
        info_path = os.path.join(patient_dir, 'info.json')
        
        if not os.path.exists(info_path):
            print(f"Skipping {patient_dir}, no info.json found")
            continue
        
        try:
            with open(info_path, 'r') as f:
                data = json.load(f)
            
            points = data['PointInfos']
            
            for point in points:
                # Extract rotation and position
                quat = get_quat(point)
                pos = get_position(point)
                
                # Convert rotation to 9D
                rot_9d = quat_to_9d(quat)
                
                all_rotations.append(rot_9d)
                all_positions.append(pos)
                
        except Exception as e:
            print(f"Error processing {patient_dir}: {e}")
            continue
    
    if not all_rotations:
        raise ValueError("No valid data found!")
    
    # Convert to numpy arrays
    all_rotations = np.array(all_rotations)
    all_positions = np.array(all_positions)
    
    # Calculate statistics
    rot_mean = all_rotations.mean(axis=0)
    rot_std = all_rotations.std(axis=0) + 1e-8
    pos_mean = all_positions.mean(axis=0) 
    pos_std = all_positions.std(axis=0) + 1e-8
    
    print(f"\nProcessed {len(all_rotations)} data points")
    print(f"Rotation stats - Mean: {rot_mean[:3]}, Std: {rot_std[:3]}")
    print(f"Position stats - Mean: {pos_mean}, Std: {pos_std}")
    
    return rot_mean, rot_std, pos_mean, pos_std

def main():
    # Update this path to your data directory
    data_root = r"C:\Users\alast\Documents\robotic_ultrasound\ct_data_random_angle"
    
    print(f"Looking for data in: {data_root}")
    
    if not os.path.exists(data_root):
        print(f"Data directory not found: {data_root}")
        print("Please update the data_root path in the script")
        return
    
    try:
        # Calculate stats using validation set (smaller, faster)
        rot_mean, rot_std, pos_mean, pos_std = calculate_stats_from_dataset(
            data_root, subset='val', max_patients=5  # Use first 5 patients for speed
        )
        
        # Save the statistics
        np.save('rot_mean.npy', rot_mean)
        np.save('rot_std.npy', rot_std)
        np.save('pos_mean.npy', pos_mean)
        np.save('pos_std.npy', pos_std)
        
        print("\nSaved normalization statistics:")
        print("- rot_mean.npy")
        print("- rot_std.npy") 
        print("- pos_mean.npy")
        print("- pos_std.npy")
        
        print("\nFiles created successfully! You can now run the GUI.")
        
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()