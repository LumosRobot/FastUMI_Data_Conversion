#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Universal Data Converter (Strict Merged Trajectory Version)
1. Recursively finds 'session_*' folders.
2. Auto-detects Dual-Arm vs Single-Arm layout.
3. Aligns data to target frequency using Merged_Trajectory only.
'''
import h5py
import pandas as pd
import os
import cv2
import glob
from tqdm import tqdm
import numpy as np
import subprocess
from concurrent.futures import ProcessPoolExecutor
import argparse

SOURCE_CAMERA_FPS = 60


def clamp_txt_to_csv(txt_path, csv_path):
    try:
        df = pd.read_csv(txt_path, sep=r'\s+', header=None)
        df.columns = ['timestamp', 'clamp']
        df.to_csv(csv_path, index=False)
        return True
    except Exception:
        return False

def ensure_clamp_csv_for_path(data_path):
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    if not os.path.exists(clamp_dir):
        return
    
    clamp_txt_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
    clamp_csv_path = os.path.join(clamp_dir, "clamp.csv")

    if os.path.exists(clamp_csv_path):
        return
    if os.path.exists(clamp_txt_path):
        clamp_txt_to_csv(clamp_txt_path, clamp_csv_path)

def detect_layout(session_path):
    if not os.path.isdir(session_path):
        return 'invalid', {}

    subdirs = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
    left_dir_name = next((d for d in subdirs if d.startswith('left_hand')), None)
    right_dir_name = next((d for d in subdirs if d.startswith('right_hand')), None)

    if left_dir_name and right_dir_name:
        return 'dual', {
            'left': os.path.join(session_path, left_dir_name),
            'right': os.path.join(session_path, right_dir_name)
        }
    
    if os.path.exists(os.path.join(session_path, "RGB_Images")):
        return 'single', {'single': session_path}
    
    return 'invalid', {}

def get_video_state(video_path):
    if not os.path.isfile(video_path):
        return False
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and float(result.stdout.strip()) > 0
    except:
        return False

def read_trj_txt(txt_path):
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"trajectory txt not found: {txt_path}")
    df = pd.read_csv(txt_path, sep=r'\s+', header=None)
    if df.shape[1] < 8:
        raise ValueError(f"trajectory txt expects >=8 columns")
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
    return df

def load_merged_trajectory(session_path):
    trj_dir = os.path.join(session_path, "Merged_Trajectory")
    trj_csv = os.path.join(trj_dir, "merged_trajectory.csv")
    trj_txt = os.path.join(trj_dir, "merged_trajectory.txt")
    
    if os.path.exists(trj_csv):
        df = pd.read_csv(trj_csv)
        expected_cols = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
        if all(c in df.columns for c in expected_cols):
            return df[expected_cols]
    
    if os.path.exists(trj_txt):
        return read_trj_txt(trj_txt)
    
    raise FileNotFoundError(f"Missing required Merged_Trajectory in {session_path}")

def load_arm_data(data_path):
    if not os.path.isdir(data_path):
        return None

    image_dir = os.path.join(data_path, "RGB_Images")
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    video_path = os.path.join(image_dir, "video.mp4")
    timestamps_path = os.path.join(image_dir, "timestamps.csv")
    clamp_path = os.path.join(clamp_dir, "clamp.csv")

    ensure_clamp_csv_for_path(data_path)

    if not (os.path.exists(video_path) and os.path.exists(timestamps_path) and os.path.exists(clamp_path)):
        return None
    if not get_video_state(video_path):
        return None

    try:
        traj = load_merged_trajectory(data_path)
        clamp = pd.read_csv(clamp_path)
        timestamps = pd.read_csv(timestamps_path)
    except Exception as e:
        return None

    if 'aligned_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['aligned_stamp']
    elif 'header_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['header_stamp']
    elif 'timestamp' not in timestamps.columns:
        timestamps['timestamp'] = np.arange(len(timestamps), dtype=float)
    
    if 'frame_index' not in timestamps.columns:
        timestamps['frame_index'] = np.arange(len(timestamps), dtype=int)

    return {
        "traj": traj,
        "clamp": clamp,
        "timestamps": timestamps,
        "video_path": video_path
    }


def write_hdf5_dual(output_path, records, camera_names):
    with h5py.File(output_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        arms_config = [("robot_0", records["robot_0"]), ("robot_1", records["robot_1"])]
        for arm_name, arm_data in arms_config:
            arm_grp = root.create_group(arm_name)
            obs_grp = arm_grp.create_group("observations")
            obs_grp.create_dataset("qpos", data=np.array(arm_data["qpos"], dtype=np.float32))
            img_grp = obs_grp.create_group("images")
            imgs_np = np.array(arm_data["images"], dtype=np.uint8)
            for cam_name in camera_names:
                img_grp.create_dataset(cam_name, data=imgs_np, compression='gzip', compression_opts=4)
            arm_grp.create_dataset("action", data=np.array(arm_data["action"], dtype=np.float32))

def write_hdf5_single(output_path, records, camera_names):
    with h5py.File(output_path, 'w', rdcc_nbytes=2 * 1024 ** 2) as root:
        root.attrs['sim'] = False
        obs_grp = root.create_group("observations")
        obs_grp.create_dataset("qpos", data=np.array(records["single"]["qpos"], dtype=np.float32))
        img_grp = obs_grp.create_group("images")
        imgs_np = np.array(records["single"]["images"], dtype=np.uint8)
        for cam_name in camera_names:
            img_grp.create_dataset(cam_name, data=imgs_np, compression='gzip', compression_opts=4)
        root.create_dataset("action", data=np.array(records["single"]["action"], dtype=np.float32))


def process_session_auto(session_path, output_root, episode_idx, camera_names, target_fps):
    session_name = os.path.basename(session_path)
    mode, paths = detect_layout(session_path)

    if mode == 'invalid':
        return

    step = max(1, int(SOURCE_CAMERA_FPS / target_fps))

    if mode == 'dual':
        left_data = load_arm_data(paths['left'])
        right_data = load_arm_data(paths['right'])
        if left_data is None or right_data is None: return

        master_timestamps = left_data["timestamps"].iloc[::step].reset_index(drop=True)
        if master_timestamps.empty: return

        records = {
            "robot_0": {"qpos": [], "action": [], "images": []},
            "robot_1": {"qpos": [], "action": [], "images": []}
        }
        
        cap_l = cv2.VideoCapture(left_data["video_path"])
        cap_r = cv2.VideoCapture(right_data["video_path"])
        
        l_traj_ts, l_clamp_ts = left_data["traj"]['timestamp'].to_numpy(), left_data["clamp"]['timestamp'].to_numpy()
        r_traj_ts, r_clamp_ts = right_data["traj"]['timestamp'].to_numpy(), right_data["clamp"]['timestamp'].to_numpy()
        r_cam_ts = right_data["timestamps"]['timestamp'].to_numpy()
        r_cam_fidx = right_data["timestamps"]['frame_index'].to_numpy()

        valid_frames = 0
        for _, row in master_timestamps.iterrows():
            t_master = row['timestamp']
            
            # Robot 0
            l_idx_t = int(np.argmin(np.abs(l_traj_ts - t_master)))
            l_idx_c = int(np.argmin(np.abs(l_clamp_ts - t_master)))
            l_pos = [
                left_data["traj"].iloc[l_idx_t]['Pos X'], left_data["traj"].iloc[l_idx_t]['Pos Y'], left_data["traj"].iloc[l_idx_t]['Pos Z'],
                left_data["traj"].iloc[l_idx_t]['Q_X'], left_data["traj"].iloc[l_idx_t]['Q_Y'], left_data["traj"].iloc[l_idx_t]['Q_Z'], left_data["traj"].iloc[l_idx_t]['Q_W'],
                left_data["clamp"].iloc[l_idx_c]['clamp']
            ]
            
            cap_l.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_index']))
            ret_l, frame_l = cap_l.read()

            # Robot 1
            r_cam_idx = int(np.argmin(np.abs(r_cam_ts - t_master)))
            t_right = r_cam_ts[r_cam_idx]
            r_idx_t = int(np.argmin(np.abs(r_traj_ts - t_right)))
            r_idx_c = int(np.argmin(np.abs(r_clamp_ts - t_right)))
            r_pos = [
                right_data["traj"].iloc[r_idx_t]['Pos X'], right_data["traj"].iloc[r_idx_t]['Pos Y'], right_data["traj"].iloc[r_idx_t]['Pos Z'],
                right_data["traj"].iloc[r_idx_t]['Q_X'], right_data["traj"].iloc[r_idx_t]['Q_Y'], right_data["traj"].iloc[r_idx_t]['Q_Z'], right_data["traj"].iloc[r_idx_t]['Q_W'],
                right_data["clamp"].iloc[r_idx_c]['clamp']
            ]

            cap_r.set(cv2.CAP_PROP_POS_FRAMES, int(r_cam_fidx[r_cam_idx]))
            ret_r, frame_r = cap_r.read()

            if ret_l and ret_r:
                records["robot_0"]["qpos"].append(l_pos); records["robot_0"]["action"].append(l_pos); records["robot_0"]["images"].append(frame_l)
                records["robot_1"]["qpos"].append(r_pos); records["robot_1"]["action"].append(r_pos); records["robot_1"]["images"].append(frame_r)
                valid_frames += 1

        cap_l.release(); cap_r.release()
        if valid_frames > 0:
            hdf5_path = os.path.join(output_root, f'episode_{episode_idx:06d}.hdf5')
            write_hdf5_dual(hdf5_path, records, camera_names)

    elif mode == 'single':
        single_data = load_arm_data(paths['single'])
        if single_data is None: return

        master_timestamps = single_data["timestamps"].iloc[::step].reset_index(drop=True)
        if master_timestamps.empty: return

        records = {"single": {"qpos": [], "action": [], "images": []}}
        cap = cv2.VideoCapture(single_data["video_path"])
        traj_ts, clamp_ts = single_data["traj"]['timestamp'].to_numpy(), single_data["clamp"]['timestamp'].to_numpy()
        
        valid_frames = 0
        for _, row in master_timestamps.iterrows():
            t_master = row['timestamp']
            idx_t = int(np.argmin(np.abs(traj_ts - t_master)))
            idx_c = int(np.argmin(np.abs(clamp_ts - t_master)))
            pos_quat = [
                single_data["traj"].iloc[idx_t]['Pos X'], single_data["traj"].iloc[idx_t]['Pos Y'], single_data["traj"].iloc[idx_t]['Pos Z'],
                single_data["traj"].iloc[idx_t]['Q_X'], single_data["traj"].iloc[idx_t]['Q_Y'], single_data["traj"].iloc[idx_t]['Q_Z'], single_data["traj"].iloc[idx_t]['Q_W'],
                single_data["clamp"].iloc[idx_c]['clamp']
            ]
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(row['frame_index']))
            ret, frame = cap.read()
            if ret:
                records["single"]["qpos"].append(pos_quat); records["single"]["action"].append(pos_quat); records["single"]["images"].append(frame)
                valid_frames += 1
        
        cap.release()
        if valid_frames > 0:
            hdf5_path = os.path.join(output_root, f'episode_{episode_idx:06d}.hdf5')
            write_hdf5_single(hdf5_path, records, camera_names)

def find_all_sessions(root_path):
    found_sessions = []
    for root, dirs, files in os.walk(root_path):
        for d in dirs:
            if d.startswith("session"):
                found_sessions.append(os.path.join(root, d))
    return sorted(found_sessions)

def convert_task(task_root, output_dir, camera_names, target_fps, num_workers):
    os.makedirs(output_dir, exist_ok=True)
    session_dirs = find_all_sessions(task_root)
    if not session_dirs: return

    episode_indices = list(range(len(session_dirs)))

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        list(tqdm(
            executor.map(
                process_session_auto,
                session_dirs,
                [output_dir] * len(session_dirs),
                episode_indices,
                [camera_names] * len(session_dirs),
                [target_fps] * len(session_dirs)
            ),
            total=len(session_dirs),
            desc="Processing Sessions"
        ))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Recursively process sessions using Merged_Trajectory.")
    parser.add_argument("--task_root", type=str, required=True, help="Root directory of the Task")
    parser.add_argument("--output", type=str, required=True, help="Output directory for HDF5 files")
    parser.add_argument("--frequency", type=int, choices=[20, 30, 60], default=20, help="Target frequency (20, 30, 60Hz)")
    parser.add_argument("--camera_names", type=str, default="front", help="Camera dataset names")
    parser.add_argument("--num_workers", type=int, default=8)

    args = parser.parse_args()
    cam_names = [s.strip() for s in args.camera_names.split(",") if s.strip()]


    convert_task(args.task_root, args.output, cam_names, args.frequency, args.num_workers)
