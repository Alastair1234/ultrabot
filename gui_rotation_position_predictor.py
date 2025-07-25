# ==============================================================================
# File: gui_rotation_position_predictor.py
#
# Description:
# Streamlit GUI for the rotation+position prediction model with random test selection
# from train/val/test splits.
# ==============================================================================

import torch
import numpy as np
import cv2
import streamlit as st
import json
import os
import pyperclip
import random
from scipy.spatial.transform import Rotation as R
from transformers import AutoImageProcessor
import logging

# Import your model
from model_ct import DinoV2RotationPositionTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

@st.cache_resource
def load_model_and_stats():
    """Load the model and normalization statistics"""
    # Load normalization stats
    try:
        rot_mean = np.load('rot_mean.npy')
        rot_std = np.load('rot_std.npy') 
        pos_mean = np.load('pos_mean.npy')
        pos_std = np.load('pos_std.npy')
        st.success("✅ Loaded normalization statistics")
    except FileNotFoundError as e:
        st.error(f"❌ Normalization files not found: {e}")
        return None, None, None, None, None, None
    
    # Load model
    model = DinoV2RotationPositionTransformer().to(device)
    try:
        model.load_state_dict(torch.load('checkpoint_step_20000.pth', map_location=device))
        model.eval()
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Failed to load model: {e}")
        return None, None, None, None, None, None
    
    # Load image processor
    image_processor = AutoImageProcessor.from_pretrained('facebook/dinov2-base')
    
    return model, image_processor, rot_mean, rot_std, pos_mean, pos_std

@st.cache_data
def load_dataset(dataset_root, split):
    """Load and cache dataset information for a specific split (train/val/test)"""
    dataset_dir = os.path.join(dataset_root, split)
    if not os.path.exists(dataset_dir):
        st.error(f"❌ Dataset directory not found: {dataset_dir}")
        return None, None
    
    # Find all patient directories
    patient_dirs = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    all_images = []
    all_metadata = {}
    
    for patient_dir in patient_dirs:
        patient_path = os.path.join(dataset_dir, patient_dir)
        info_path = os.path.join(patient_path, 'info.json')
        images_dir = os.path.join(patient_path, 'images')
        
        if os.path.exists(info_path) and os.path.exists(images_dir):
            try:
                with open(info_path, 'r') as f:
                    json_data = json.load(f)
                
                for point_info in json_data['PointInfos']:
                    filename = point_info['FileName']
                    image_path = os.path.join(images_dir, filename)
                    
                    if os.path.exists(image_path):
                        all_images.append({
                            'path': image_path,
                            'filename': filename,
                            'patient': patient_dir
                        })
                        all_metadata[filename] = point_info
                        
            except Exception as e:
                st.warning(f"⚠️ Error loading {patient_dir}: {e}")
    
    return all_images, all_metadata

def preprocess_image(img, image_processor):
    """Preprocess image for the model"""
    if len(img.shape) == 3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img_rgb = img
    
    processed = image_processor(img_rgb, return_tensors="pt")
    return processed.pixel_values.squeeze(0).to(device)

def get_quat_from_point(point_data):
    """Extract quaternion from point data"""
    return np.array([
        point_data['RotationQuaternion']['x'],
        point_data['RotationQuaternion']['y'], 
        point_data['RotationQuaternion']['z'],
        point_data['RotationQuaternion']['w']
    ])

def get_position_from_point(point_data):
    """Extract position from point data"""
    return np.array([
        point_data['Position']['x'],
        point_data['Position']['y'],
        point_data['Position']['z']
    ])

def quat_to_9d(quat):
    """Convert quaternion to 9D rotation matrix representation"""
    return R.from_quat(quat).as_matrix().reshape(9)

def nine_d_to_quat(nine_d):
    """Convert 9D rotation matrix back to quaternion"""
    mat = nine_d.reshape(3, 3)
    u, s, vt = np.linalg.svd(mat)
    rot_mat = u @ vt
    
    if np.linalg.det(rot_mat) < 0:
        u[:, -1] = -u[:, -1]
        rot_mat = u @ vt
    
    return R.from_matrix(rot_mat).as_quat()

def predict_third_point(model, image_processor, img1, img2, img3, point1_data, point2_data, 
                        rot_mean, rot_std, pos_mean, pos_std):
    """Predict the pose of the third point given two reference points"""
    
    img1_tensor = preprocess_image(img1, image_processor).unsqueeze(0)
    img2_tensor = preprocess_image(img2, image_processor).unsqueeze(0) 
    img3_tensor = preprocess_image(img3, image_processor).unsqueeze(0)
    
    rot1 = quat_to_9d(get_quat_from_point(point1_data))
    pos1 = get_position_from_point(point1_data)
    rot2 = quat_to_9d(get_quat_from_point(point2_data))
    pos2 = get_position_from_point(point2_data)
    
    rot1_norm = (rot1 - rot_mean) / rot_std
    pos1_norm = (pos1 - pos_mean) / pos_std
    rot2_norm = (rot2 - rot_mean) / rot_std  
    pos2_norm = (pos2 - pos_mean) / pos_std
    
    input_abs = np.concatenate([rot1_norm, pos1_norm, rot2_norm, pos2_norm])
    input_tensor = torch.tensor(input_abs).float().unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(img1_tensor, img2_tensor, img3_tensor, input_tensor)
    
    pred_rot_norm, pred_pos_norm = output[0, :9].cpu().numpy(), output[0, 9:].cpu().numpy()
    pred_rot = pred_rot_norm * rot_std + rot_mean
    pred_pos = pred_pos_norm * pos_std + pos_mean
    
    pred_quat = nine_d_to_quat(pred_rot)
    
    return pred_pos, pred_quat

def display_results_panel(pred_pos, pred_quat, real_pos, real_quat, img_info, split):
    """Display results in a nice panel format"""
    
    # Calculate errors
    pos_error = np.linalg.norm(pred_pos - real_pos)
    r_pred = R.from_quat(pred_quat)
    r_real = R.from_quat(real_quat)
    angular_error = (r_real.inv() * r_pred).magnitude() * (180 / np.pi)
    
    # Create columns for side-by-side comparison
    col1, col2, col3 = st.columns([1, 1, 1])
    
    with col1:
        st.markdown("### 🎯 **PREDICTED POSE**")
        st.markdown(f"**Position:**")
        st.markdown(f"- X: `{pred_pos[0]:.3f}` mm")
        st.markdown(f"- Y: `{pred_pos[1]:.3f}` mm") 
        st.markdown(f"- Z: `{pred_pos[2]:.3f}` mm")
        st.markdown(f"**Rotation (Quaternion):**")
        st.markdown(f"- X: `{pred_quat[0]:.3f}`")
        st.markdown(f"- Y: `{pred_quat[1]:.3f}`")
        st.markdown(f"- Z: `{pred_quat[2]:.3f}`")
        st.markdown(f"- W: `{pred_quat[3]:.3f}`")
    
    with col2:
        st.markdown("### ✅ **GROUND TRUTH**")
        st.markdown(f"**Position:**")
        st.markdown(f"- X: `{real_pos[0]:.3f}` mm")
        st.markdown(f"- Y: `{real_pos[1]:.3f}` mm")
        st.markdown(f"- Z: `{real_pos[2]:.3f}` mm")
        st.markdown(f"**Rotation (Quaternion):**")
        st.markdown(f"- X: `{real_quat[0]:.3f}`")
        st.markdown(f"- Y: `{real_quat[1]:.3f}`")
        st.markdown(f"- Z: `{real_quat[2]:.3f}`")
        st.markdown(f"- W: `{real_quat[3]:.3f}`")
    
    with col3:
        st.markdown("### 📊 **ERROR ANALYSIS**")
        
        # Color-code errors
        pos_color = "🟢" if pos_error < 1.0 else "🟡" if pos_error < 5.0 else "🔴"
        ang_color = "🟢" if angular_error < 5.0 else "🟡" if angular_error < 15.0 else "🔴"
        
        st.markdown(f"**Position Error:** {pos_color}")
        st.markdown(f"`{pos_error:.3f}` mm")
        
        st.markdown(f"**Angular Error:** {ang_color}")
        st.markdown(f"`{angular_error:.2f}` degrees")
        
        st.markdown("**Image Info:**")
        st.markdown(f"- Split: `{split.upper()}`")
        st.markdown(f"- Patient: `{img_info['patient']}`")
        st.markdown(f"- File: `{img_info['filename']}`")
    
    # Overall assessment
    if pos_error < 1.0 and angular_error < 5.0:
        st.success("🎉 **EXCELLENT PREDICTION!** Both position and rotation errors are very low.")
    elif pos_error < 5.0 and angular_error < 15.0:
        st.info("👍 **GOOD PREDICTION!** Errors are within acceptable range.")
    else:
        st.warning("⚠️ **MODERATE PREDICTION** - Consider the model's limitations.")
    
    return pos_error, angular_error

def main():
    st.set_page_config(page_title="DinoV2 Pose Predictor", page_icon="🤖", layout="wide")
    
    st.title("🤖 DinoV2 Rotation+Position Predictor")
    st.markdown("Predict absolute pose of third point given two reference points. Supports testing on train/val/test splits.")
    
    # Load model and normalization stats
    model, image_processor, rot_mean, rot_std, pos_mean, pos_std = load_model_and_stats()
    
    if model is None:
        st.error("❌ Failed to load model. Please check your files.")
        return
    
    # Dataset root input
    dataset_root = st.text_input("📁 Dataset Root Directory:", 
                                 value=r"C:\Users\alast\Documents\robotic_ultrasound\ct_data_random_angle",
                                 help="Path to your dataset root with train/val/test subdirectories")
    
    if dataset_root and os.path.exists(dataset_root):
        # Split selection
        split = st.selectbox("📂 Select Dataset Split:", ["test", "train", "val"], index=0)
        
        # Load dataset for the selected split
        all_images, all_metadata = load_dataset(dataset_root, split)
        
        if all_images and len(all_images) >= 3:
            st.success(f"✅ Found {len(all_images)} images in {split} split across {len(set(img['patient'] for img in all_images))} patients")
            
            # Random test button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🎲 **Random Test**", type="primary", help=f"Randomly select 3 images from {split} split and run prediction"):
                    # Store random selection in session state
                    selected_images = random.sample(all_images, 3)
                    st.session_state.selected_images = selected_images
                    st.session_state.selected_split = split
            
            with col2:
                if st.button("🔄 **Clear Results**", help="Clear current results"):
                    if 'selected_images' in st.session_state:
                        del st.session_state.selected_images
                    if 'selected_split' in st.session_state:
                        del st.session_state.selected_split
            
            # Display results if images are selected
            if 'selected_images' in st.session_state:
                selected_images = st.session_state.selected_images
                selected_split = st.session_state.selected_split
                
                st.markdown("---")
                st.markdown(f"## 🖼️ **Selected {selected_split.upper()} Images**")
                
                # Display the three selected images
                col1, col2, col3 = st.columns(3)
                
                images = []
                metadatas = []
                
                for i, img_info in enumerate(selected_images):
                    img = cv2.imread(img_info['path'])
                    images.append(img)
                    metadatas.append(all_metadata[img_info['filename']])
                    
                    with [col1, col2, col3][i]:
                        st.markdown(f"### {'Reference 1' if i == 0 else 'Reference 2' if i == 1 else 'Target'}")
                        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), use_column_width=True)
                        st.markdown(f"**Patient:** `{img_info['patient']}`")
                        st.markdown(f"**File:** `{img_info['filename']}`")
                
                # Run prediction
                st.markdown("---")
                st.markdown("## 🔮 **Prediction Results**")
                
                try:
                    # Make prediction
                    pred_pos, pred_quat = predict_third_point(
                        model, image_processor, images[0], images[1], images[2],
                        metadatas[0], metadatas[1],
                        rot_mean, rot_std, pos_mean, pos_std
                    )
                    
                    # Get ground truth
                    real_pos = get_position_from_point(metadatas[2])
                    real_quat = get_quat_from_point(metadatas[2])
                    
                    # Display results
                    pos_error, angular_error = display_results_panel(
                        pred_pos, pred_quat, real_pos, real_quat, selected_images[2], selected_split
                    )
                    
                    # JSON outputs for copying
                    st.markdown("---")
                    st.markdown("## 📋 **JSON Outputs**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.markdown("### Predicted Pose JSON")
                        pred_dict = {
                            "Position": {"X": float(pred_pos[0]), "Y": float(pred_pos[1]), "Z": float(pred_pos[2])},
                            "RotationQuaternion": {"X": float(pred_quat[0]), "Y": float(pred_quat[1]), "Z": float(pred_quat[2]), "W": float(pred_quat[3])}
                        }
                        st.json(pred_dict)
                        if st.button("📋 Copy Predicted JSON"):
                            pyperclip.copy(json.dumps(pred_dict, indent=2))
                            st.success("Copied to clipboard!")
                    
                    with col2:
                        st.markdown("### Ground Truth JSON")
                        real_dict = {
                            "Position": {"X": float(real_pos[0]), "Y": float(real_pos[1]), "Z": float(real_pos[2])},
                            "RotationQuaternion": {"X": float(real_quat[0]), "Y": float(real_quat[1]), "Z": float(real_quat[2]), "W": float(real_quat[3])}
                        }
                        st.json(real_dict)
                        if st.button("📋 Copy Ground Truth JSON"):
                            pyperclip.copy(json.dumps(real_dict, indent=2))
                            st.success("Copied to clipboard!")
                    
                    # Summary statistics
                    st.markdown("---")
                    st.markdown("## 📈 **Performance Summary**")
                    
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Position Error", f"{pos_error:.3f} mm")
                    with col2:
                        st.metric("Angular Error", f"{angular_error:.2f}°")
                    with col3:
                        st.metric("Device", "🔥 GPU" if torch.cuda.is_available() else "💻 CPU")
                    with col4:
                        st.metric("Model", "DinoV2-Base")
                        
                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
                    st.exception(e)
        else:
            st.error(f"❌ Need at least 3 images in {split} split")
    else:
        st.warning("⚠️ Please enter a valid dataset root directory")
    
    # Manual upload option
    st.markdown("---")
    st.markdown("## 📤 **Manual Upload Option**")
    
    with st.expander("Upload your own images"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            uploaded_file1 = st.file_uploader("Reference Point 1 Image", ["png", "jpg", "jpeg"], key="manual1")
        with col2:
            uploaded_file2 = st.file_uploader("Reference Point 2 Image", ["png", "jpg", "jpeg"], key="manual2")
        with col3:
            uploaded_file3 = st.file_uploader("Target Point Image", ["png", "jpg", "jpeg"], key="manual3")
        
        # Inputs for metadata (since manual upload doesn't have JSON)
        st.markdown("### Reference 1 Metadata")
        ref1_pos_x = st.number_input("Ref1 Pos X", value=0.0, key="ref1_x")
        ref1_pos_y = st.number_input("Ref1 Pos Y", value=0.0, key="ref1_y")
        ref1_pos_z = st.number_input("Ref1 Pos Z", value=0.0, key="ref1_z")
        ref1_quat_x = st.number_input("Ref1 Quat X", value=0.0, key="ref1_qx")
        ref1_quat_y = st.number_input("Ref1 Quat Y", value=0.0, key="ref1_qy")
        ref1_quat_z = st.number_input("Ref1 Quat Z", value=0.0, key="ref1_qz")
        ref1_quat_w = st.number_input("Ref1 Quat W", value=1.0, key="ref1_qw")
        
        st.markdown("### Reference 2 Metadata")
        ref2_pos_x = st.number_input("Ref2 Pos X", value=0.0, key="ref2_x")
        ref2_pos_y = st.number_input("Ref2 Pos Y", value=0.0, key="ref2_y")
        ref2_pos_z = st.number_input("Ref2 Pos Z", value=0.0, key="ref2_z")
        ref2_quat_x = st.number_input("Ref2 Quat X", value=0.0, key="ref2_qx")
        ref2_quat_y = st.number_input("Ref2 Quat Y", value=0.0, key="ref2_qy")
        ref2_quat_z = st.number_input("Ref2 Quat Z", value=0.0, key="ref2_qz")
        ref2_quat_w = st.number_input("Ref2 Quat W", value=1.0, key="ref2_qw")
        
        st.markdown("### Target Metadata (for Error Calculation)")
        target_pos_x = st.number_input("Target Pos X", value=0.0, key="target_x")
        target_pos_y = st.number_input("Target Pos Y", value=0.0, key="target_y")
        target_pos_z = st.number_input("Target Pos Z", value=0.0, key="target_z")
        target_quat_x = st.number_input("Target Quat X", value=0.0, key="target_qx")
        target_quat_y = st.number_input("Target Quat Y", value=0.0, key="target_qy")
        target_quat_z = st.number_input("Target Quat Z", value=0.0, key="target_qz")
        target_quat_w = st.number_input("Target Quat W", value=1.0, key="target_qw")
        
        if uploaded_file1 and uploaded_file2 and uploaded_file3:
            try:
                img1 = cv2.imdecode(np.frombuffer(uploaded_file1.read(), np.uint8), cv2.IMREAD_COLOR)
                img2 = cv2.imdecode(np.frombuffer(uploaded_file2.read(), np.uint8), cv2.IMREAD_COLOR)
                img3 = cv2.imdecode(np.frombuffer(uploaded_file3.read(), np.uint8), cv2.IMREAD_COLOR)
                
                point1_data = {
                    'Position': {'x': ref1_pos_x, 'y': ref1_pos_y, 'z': ref1_pos_z},
                    'RotationQuaternion': {'x': ref1_quat_x, 'y': ref1_quat_y, 'z': ref1_quat_z, 'w': ref1_quat_w}
                }
                point2_data = {
                    'Position': {'x': ref2_pos_x, 'y': ref2_pos_y, 'z': ref2_pos_z},
                    'RotationQuaternion': {'x': ref2_quat_x, 'y': ref2_quat_y, 'z': ref2_quat_z, 'w': ref2_quat_w}
                }
                point3_data = {
                    'Position': {'x': target_pos_x, 'y': target_pos_y, 'z': target_pos_z},
                    'RotationQuaternion': {'x': target_quat_x, 'y': target_quat_y, 'z': target_quat_z, 'w': target_quat_w}
                }
                
                pred_pos, pred_quat = predict_third_point(
                    model, image_processor, img1, img2, img3,
                    point1_data, point2_data,
                    rot_mean, rot_std, pos_mean, pos_std
                )
                
                real_pos = get_position_from_point(point3_data)
                real_quat = get_quat_from_point(point3_data)
                
                # Display uploaded images
                st.markdown("### Uploaded Images")
                colu1, colu2, colu3 = st.columns(3)
                with colu1:
                    st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), "Reference 1")
                with colu2:
                    st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), "Reference 2")
                with colu3:
                    st.image(cv2.cvtColor(img3, cv2.COLOR_BGR2RGB), "Target")
                
                # Display results
                display_results_panel(
                    pred_pos, pred_quat, real_pos, real_quat, 
                    {'patient': 'Manual', 'filename': 'uploaded'}, 'manual'
                )
                
            except Exception as e:
                st.error(f"❌ Manual prediction failed: {e}")
        else:
            st.info("📝 Upload all three images and provide metadata to run prediction.")

if __name__ == '__main__':
    main()