"""
Convert raw recorded sessions directly into a LeRobot dataset.

This script aligns timestamps, extracts frames from recorded videos,
and writes episodes into a LeRobot dataset (no intermediate HDF5).

Key assumptions:
- Video frames are read with OpenCV (BGR), converted to RGB and resized
    to 224x224 (uint8) to match LeRobot image expectations.
- Single-arm sessions write images to `observation.images.robot_0`.
    Dual-arm sessions populate both `robot_0` and `robot_1` image fields.

Minimal usage example (required flags shown):

    python raw_2_lerobot_V2.1.py \
        --task-root /path/to/raw/task_root \
        --repo-id myorg/myrepo \
        --task-name "Pick and place" \
        --traj-source merge \
        --target-fps 20 \
        --source_camera_fps 60 \
        --mode video \
        --output /path/to/save/lerobot \
        --max-workers 8

Notes:
- The script expects session directories named like `session*` under
  `--task-root` and standard subfolders (`RGB_Images`, `Clamp_Data`,
  `Merged_Trajectory`, etc.). See the code for exact expectations.
- Required Python packages: `opencv-python`, `numpy`, `pandas`, `h5py`,
  and the `lerobot` package providing `LeRobotDataset`.

Recommended `lerobot` dependency (git + pinned revision):

    lerobot = { git = "https://github.com/huggingface/lerobot", rev = "0cf864870cf29f4738d3ade893e6fd13fbd7cdb5" }
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, Literal, Tuple, Dict, Any

import concurrent.futures
from multiprocessing import cpu_count

import dataclasses
import shutil
import cv2
import numpy as np
import pandas as pd
import subprocess
import json

# LeRobotDataset is provided by the `lerobot` package; we import it directly
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME


def _ensure_module_path():
    this_dir = Path(__file__).resolve().parent
    if str(this_dir) not in sys.path:
        sys.path.insert(0, str(this_dir))


def _to_rgb_and_resize(img, size=(224, 224)):
    # img: BGR numpy array from cv2
    if img is None:
        return None
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
    return resized.astype(np.uint8)


# ---------------------------------------------------------------------------
# Module-level helpers (modified and added)
# ---------------------------------------------------------------------------

def sort_timestamps_with_original_indices(ts_arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    added function: Sort timestamps while retaining original indices.
    Args:
        ts_arr: 1D numpy array of timestamps (may contain nan/inf).
    Returns:
        sorted_ts: Sorted numpy array of timestamps (finite values only).
        original_indices: Indices mapping sorted_ts back to original ts_arr.
    """
    ts_arr = np.asarray(ts_arr, dtype=np.float64)
    
    # Filter out nan/inf values (retain original indices, mark invalid timestamps as -1)
    valid_ts_mask = np.isfinite(ts_arr)
    valid_ts = ts_arr[valid_ts_mask]
    valid_original_indices = np.where(valid_ts_mask)[0]
    
    if len(valid_ts) == 0:
        raise ValueError("timestamp array contains no valid finite values to sort")
    
    # Sort and retain original indices
    sorted_indices = np.argsort(valid_ts)
    sorted_ts = valid_ts[sorted_indices]
    sorted_original_indices = valid_original_indices[sorted_indices]
    
    return sorted_ts, sorted_original_indices


def find_nearest_indices(sorted_arr: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Modified points:
    1.  Explicitly check if input sorted_arr is truly sorted
    2.  Return not only the nearest indices but also the corresponding timestamp differences (for subsequent tolerance filtering)
    3.  Add checks for empty and invalid values to avoid silent failures
    """
    # Explicit check: sorted_arr must be non-empty and monotonically increasing
    if len(sorted_arr) == 0:
        raise ValueError("input sorted_arr cannot be an empty array for nearest neighbor search")
    
    if not np.all(np.diff(sorted_arr) >= 0):
        raise ValueError("input sorted_arr is not a monotonically increasing sorted array, binary search (np.searchsorted) will fail")
    
    # Filter out nan/inf values in targets (to avoid search errors)
    targets = np.asarray(targets, dtype=np.float64)
    valid_target_mask = np.isfinite(targets)
    if not np.any(valid_target_mask):
        raise ValueError("input targets contains no valid finite values for nearest neighbor search")
    
    # Original binary search logic retained
    indices = np.searchsorted(sorted_arr, targets)
    indices = np.clip(indices, 1, len(sorted_arr) - 1)
    
    # Calculate differences to the left and right neighbors, choose the closer one
    left = sorted_arr[indices - 1]
    right = sorted_arr[indices]
    left_diff = np.abs(targets - left)
    right_diff = np.abs(targets - right)
    
    # Determine final indices and calculate minimum differences (for tolerance filtering)
    choose_left = left_diff < right_diff
    final_indices = np.where(choose_left, indices - 1, indices)
    final_diffs = np.where(choose_left, left_diff, right_diff)
    
    # For invalid targets (nan/inf), fill with -1 (mark as invalid index) and differences with infinity
    final_indices[~valid_target_mask] = -1
    final_diffs[~valid_target_mask] = np.inf
    
    return final_indices, final_diffs


def extract_frames_sequential(video_path: str, frame_indices):
    """Read selected frames from a video in sequential order.

    This avoids random seeks which are slow for GOP codecs (H.264/H.265).

    Args:
        video_path: Path to the video file.
        frame_indices: Iterable of integer frame indices to extract.

    Returns:
        dict mapping frame_index -> BGR numpy array (as returned by cv2.read()).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {}

    frames = {}
    sorted_indices = sorted(set(frame_indices))

    current_frame = 0
    for target in sorted_indices:
        while current_frame < target:
            # added check for grab() failure
            if not cap.grab():
                print(f"[Warning] Video {video_path} failed to grab frame {current_frame}, possible video corruption")
                current_frame += 1
                continue
            current_frame += 1
        ret, frame = cap.read()
        current_frame += 1
        if ret:
            frames[target] = frame

    cap.release()
    return frames


def clamp_txt_to_csv(txt_path: str, csv_path: str) -> bool:
    """Convert clamp_data text file to CSV if present."""
    try:
        df = pd.read_csv(txt_path, sep=r"\s+", header=None)
        df.columns = ["timestamp", "clamp"]
        df.to_csv(csv_path, index=False)
        return True
    except Exception as e:
        print(f"[Warning] Failed to convert clamp TXT file {txt_path} to CSV: {str(e)}")
        return False


def ensure_clamp_csv_for_path(data_path: str) -> None:
    """Ensure `Clamp_Data/clamp.csv` exists in a data path (convert if needed)."""
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    if not os.path.exists(clamp_dir):
        return
    clamp_txt_path = os.path.join(clamp_dir, "clamp_data_tum.txt")
    clamp_csv_path = os.path.join(clamp_dir, "clamp.csv")
    if os.path.exists(clamp_csv_path):
        return
    if os.path.exists(clamp_txt_path):
        clamp_txt_to_csv(clamp_txt_path, clamp_csv_path)


def detect_layout(session_path: str):
    """Detect whether a session is single-arm or dual-arm.

    Returns:
        tuple(mode, paths) where mode in {'dual','single','invalid'} and
        paths is a dict with keys pointing to the arm directories.
    """
    if not os.path.isdir(session_path):
        return 'invalid', {}
    subdirs = [d for d in os.listdir(session_path) if os.path.isdir(os.path.join(session_path, d))]
    left_dir_name = next((d for d in subdirs if d.startswith('left_hand')), None)
    right_dir_name = next((d for d in subdirs if d.startswith('right_hand')), None)
    if left_dir_name and right_dir_name:
        return 'dual', {'left': os.path.join(session_path, left_dir_name), 'right': os.path.join(session_path, right_dir_name)}
    if os.path.exists(os.path.join(session_path, "RGB_Images")):
        return 'single', {'single': session_path}
    return 'invalid', {}


def get_video_state(video_path: str) -> bool:
    """Check whether a video file is valid using ffprobe (duration > 0)."""
    if not os.path.isfile(video_path):
        return False
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return result.returncode == 0 and float(result.stdout.strip()) > 0
    except Exception as e:
        print(f"[Warning] Failed to validate video {video_path}: {str(e)}. Please check if FFmpeg is installed.")
        return False


def read_trj_txt(txt_path: str) -> pd.DataFrame:
    """Read a trajectory text file into a DataFrame with 8 columns.

    Expected columns: timestamp PosX PosY PosZ Qx Qy Qz Qw
    """
    if not os.path.exists(txt_path):
        raise FileNotFoundError(f"trajectory txt not found: {txt_path}")
    df = pd.read_csv(txt_path, sep=r"\s+", header=None)
    if df.shape[1] < 8:
        raise ValueError(f"Trajectory TXT file {txt_path} has fewer than 8 columns, current columns: {df.shape[1]}")
    df = df.iloc[:, :8]
    df.columns = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
    # Added: Validate that the timestamp column contains valid numeric values
    if not pd.api.types.is_numeric_dtype(df['timestamp']):
        raise ValueError(f"Trajectory TXT file {txt_path} timestamp column is not numeric and cannot be processed")
    return df


def load_trajectory(session_path: str, traj_source: str) -> pd.DataFrame:
    """Load trajectory data from one of the supported sources."""
    if traj_source == 'merge':
        trj_dir = os.path.join(session_path, "Merged_Trajectory")
        trj_csv = os.path.join(trj_dir, "merged_trajectory.csv")
        trj_txt = os.path.join(trj_dir, "merged_trajectory.txt")
        if os.path.exists(trj_csv):
            df = pd.read_csv(trj_csv)
            expected_cols = ['timestamp', 'Pos X', 'Pos Y', 'Pos Z', 'Q_X', 'Q_Y', 'Q_Z', 'Q_W']
            if all(c in df.columns for c in expected_cols):
                # Added: Validate that the timestamp column contains valid numeric values
                if not pd.api.types.is_numeric_dtype(df['timestamp']):
                    raise ValueError(f"Merged trajectory CSV file {trj_csv} timestamp column is not numeric")
                return df[expected_cols]
        if os.path.exists(trj_txt):
            return read_trj_txt(trj_txt)
        raise FileNotFoundError(f"No merged trajectory found in {session_path}")
    elif traj_source == 'slam':
        return read_trj_txt(os.path.join(session_path, "SLAM_Poses", "slam_processed.txt"))
    elif traj_source == 'vive':
        return read_trj_txt(os.path.join(session_path, "Vive_Poses", "vive_data_tum.txt"))
    else:
        raise ValueError(f"Unknown traj_source={traj_source}")


def load_arm_data(data_path: str, traj_source: str):
    """Load video, timestamps, clamp and trajectory for a single arm path.

    Returns a dict with keys: 'traj', 'clamp', 'timestamps', 'video_path', or None when missing.
    """
    if not os.path.isdir(data_path):
        return None
    image_dir = os.path.join(data_path, "RGB_Images")
    clamp_dir = os.path.join(data_path, "Clamp_Data")
    video_path = os.path.join(image_dir, "video.mp4")
    timestamps_path = os.path.join(image_dir, "timestamps.csv")
    clamp_path = os.path.join(clamp_dir, "clamp.csv")
    ensure_clamp_csv_for_path(data_path)
    if not (os.path.exists(video_path) and os.path.exists(timestamps_path) and os.path.exists(clamp_path)):
        print(f"[Warning] Arm data path {data_path} is missing required files (video.mp4 / timestamps.csv / clamp.csv)")
        return None
    if not get_video_state(video_path):
        print(f"[Warning] Video file {video_path} is invalid (corrupted or zero duration)")
        return None
    try:
        traj = load_trajectory(data_path, traj_source)
        clamp = pd.read_csv(clamp_path)
        timestamps = pd.read_csv(timestamps_path)
    except Exception as e:
        print(f"[Warning] Failed to load arm data {data_path}: {str(e)}")
        return None
    
    # Validate and assign timestamp column
    if 'aligned_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['aligned_stamp']
    elif 'header_stamp' in timestamps.columns:
        timestamps['timestamp'] = timestamps['header_stamp']
    else:
        print(f"[Warning] Timestamp file {timestamps_path} does not have a suitable timestamp column (aligned_stamp / header_stamp)")
        return None
    
    # Validate timestamp column
    if not pd.api.types.is_numeric_dtype(timestamps['timestamp']):
        print(f"[Warning] Timestamp file {timestamps_path} timestamp column is not numeric")
        return None
    
    if 'frame_index' not in timestamps.columns:
        timestamps['frame_index'] = np.arange(len(timestamps), dtype=int)
    return {"traj": traj, "clamp": clamp, "timestamps": timestamps, "video_path": video_path}


def find_all_sessions(root_path: str) -> List[str]:
    """Recursively find all directories that start with 'session' under root_path."""
    found_sessions = []
    for root, dirs, _ in os.walk(root_path):
        for d in dirs:
            if d.startswith("session"):
                found_sessions.append(os.path.join(root, d))
    if not found_sessions:
        print(f"[Warning] No session directories found under root path {root_path}")
    return sorted(found_sessions)


def prepare_bimanual_alignment(left: dict, right: dict, target_fps: int, source_camera_fps: int, tolerance_s: float = 1e-4):
    """Prepare timestamp/trajectory indices and pre-extract frames for dual-arm sessions.

    Args:
        left: dict returned by `load_arm_data` for left arm.
        right: dict returned by `load_arm_data` for right arm.
        source_camera_fps: original camera frequency in Hz.
        target_fps: desired output frequency in Hz.
        tolerance_s: time alignment tolerance in seconds; values exceeding this are marked as invalid

    Returns:
        Tuple containing:
        - master_ts (np.ndarray): timestamps used as master timeline
        - master_fidx (np.ndarray): frame indices on the master camera
        - l_traj_arr, l_clamp_arr: left trajectory/clamp arrays
        - r_traj_arr, r_clamp_arr: right trajectory/clamp arrays
        - l_traj_idx, l_clamp_idx, r_traj_idx, r_clamp_idx: precomputed index arrays
        - left_frames, right_frames: dicts mapping frame_index->frame (BGR)
        - r_cam_fidx, r_cam_idx: right camera frame indices and alignment indices
    """
    master = left["timestamps"].iloc[:: max(1, int(source_camera_fps / target_fps))].reset_index(drop=True)
    master_ts = master["timestamp"].to_numpy()
    master_fidx = master["frame_index"].to_numpy().astype(int)

    # ---------------------- Left arm data processing (sorting + preserving original indices + tolerance filtering) ----------------------
    l_traj_ts = left["traj"]["timestamp"].to_numpy()
    l_clamp_ts = left["clamp"]["timestamp"].to_numpy()
    
    # 1. Sort timestamps and preserve original indices (to fix mapping misalignment issues)
    sorted_l_traj_ts, sorted_l_traj_original_indices = sort_timestamps_with_original_indices(l_traj_ts)
    sorted_l_clamp_ts, sorted_l_clamp_original_indices = sort_timestamps_with_original_indices(l_clamp_ts)
    
    # 2. Call the modified find_nearest_indices to get indices and alignment differences
    l_traj_idx_sorted, l_traj_diffs = find_nearest_indices(sorted_l_traj_ts, master_ts)
    l_clamp_idx_sorted, l_clamp_diffs = find_nearest_indices(sorted_l_clamp_ts, master_ts)
    
    # 3. Tolerance filtering: mark indices exceeding tolerance_s as -1 (invalid)
    l_traj_invalid_mask = l_traj_diffs > tolerance_s
    l_clamp_invalid_mask = l_clamp_diffs > tolerance_s
    l_traj_idx_sorted[l_traj_invalid_mask] = -1
    l_clamp_idx_sorted[l_clamp_invalid_mask] = -1
    
    # 4. Map back to original data indices (key: fix the issue where sorted indices do not correspond to original data)
    l_traj_idx = np.full_like(l_traj_idx_sorted, -1)
    valid_l_traj_mask = l_traj_idx_sorted != -1
    l_traj_idx[valid_l_traj_mask] = sorted_l_traj_original_indices[l_traj_idx_sorted[valid_l_traj_mask]]
    
    l_clamp_idx = np.full_like(l_clamp_idx_sorted, -1)
    valid_l_clamp_mask = l_clamp_idx_sorted != -1
    l_clamp_idx[valid_l_clamp_mask] = sorted_l_clamp_original_indices[l_clamp_idx_sorted[valid_l_clamp_mask]]
    
    # 5. Log invalid alignment counts
    l_traj_invalid_count = np.sum(l_traj_invalid_mask)
    l_clamp_invalid_count = np.sum(l_clamp_invalid_mask)
    if l_traj_invalid_count > 0:
        print(f"[Warning] Left arm trajectory {l_traj_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    if l_clamp_invalid_count > 0:
        print(f"[Warning] Left arm clamp {l_clamp_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    # ----------------------------------------------------------------------------------------------

    # ---------------------- Right arm data processing (same logic as left arm) ----------------------
    r_traj_ts = right["traj"]["timestamp"].to_numpy()
    r_clamp_ts = right["clamp"]["timestamp"].to_numpy()
    r_cam_ts = right["timestamps"]["timestamp"].to_numpy()
    r_cam_fidx = right["timestamps"]["frame_index"].to_numpy()
    
    # 1. Sort timestamps and preserve original indices
    sorted_r_traj_ts, sorted_r_traj_original_indices = sort_timestamps_with_original_indices(r_traj_ts)
    sorted_r_clamp_ts, sorted_r_clamp_original_indices = sort_timestamps_with_original_indices(r_clamp_ts)
    sorted_r_cam_ts, sorted_r_cam_original_indices = sort_timestamps_with_original_indices(r_cam_ts)
    
    # 2. First align right arm camera timestamps with master timestamps
    r_cam_idx_sorted, r_cam_diffs = find_nearest_indices(sorted_r_cam_ts, master_ts)
    r_cam_invalid_mask = r_cam_diffs > tolerance_s
    r_cam_idx_sorted[r_cam_invalid_mask] = -1
    
    # Map back to original camera indices
    r_cam_idx = np.full_like(r_cam_idx_sorted, -1)
    valid_r_cam_mask = r_cam_idx_sorted != -1
    r_cam_idx[valid_r_cam_mask] = sorted_r_cam_original_indices[r_cam_idx_sorted[valid_r_cam_mask]]
    
    # 3. Based on aligned right arm camera timestamps, align trajectory and clamp data
    r_cam_ts_aligned = np.full_like(master_ts, np.nan)
    r_cam_ts_aligned[valid_r_cam_mask] = sorted_r_cam_ts[r_cam_idx_sorted[valid_r_cam_mask]]
    
    r_traj_idx_sorted, r_traj_diffs = find_nearest_indices(sorted_r_traj_ts, r_cam_ts_aligned)
    r_clamp_idx_sorted, r_clamp_diffs = find_nearest_indices(sorted_r_clamp_ts, r_cam_ts_aligned)
    
    # 4. Tolerance filtering
    r_traj_invalid_mask = r_traj_diffs > tolerance_s
    r_clamp_invalid_mask = r_clamp_diffs > tolerance_s
    r_traj_idx_sorted[r_traj_invalid_mask] = -1
    r_clamp_idx_sorted[r_clamp_invalid_mask] = -1
    
    # 5. Map back to original data indices
    r_traj_idx = np.full_like(r_traj_idx_sorted, -1)
    valid_r_traj_mask = r_traj_idx_sorted != -1
    r_traj_idx[valid_r_traj_mask] = sorted_r_traj_original_indices[r_traj_idx_sorted[valid_r_traj_mask]]
    
    r_clamp_idx = np.full_like(r_clamp_idx_sorted, -1)
    valid_r_clamp_mask = r_clamp_idx_sorted != -1
    r_clamp_idx[valid_r_clamp_mask] = sorted_r_clamp_original_indices[r_clamp_idx_sorted[valid_r_clamp_mask]]
    
    # 6. Log invalid alignment counts
    r_cam_invalid_count = np.sum(r_cam_invalid_mask)
    r_traj_invalid_count = np.sum(r_traj_invalid_mask)
    r_clamp_invalid_count = np.sum(r_clamp_invalid_mask)
    if r_cam_invalid_count > 0:
        print(f"[Warning] Right arm camera {r_cam_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    if r_traj_invalid_count > 0:
        print(f"[Warning] Right arm trajectory {r_traj_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    if r_clamp_invalid_count > 0:
        print(f"[Warning] Right arm clamp {r_clamp_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    # ----------------------------------------------------------------------------------------------

    # Extract trajectory/clamp arrays
    l_traj_arr = left["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    l_clamp_arr = left["clamp"]["clamp"].to_numpy()
    r_traj_arr = right["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    r_clamp_arr = right["clamp"]["clamp"].to_numpy()

    # Extract valid frame indices (skip invalid alignment indices)
    valid_master_mask = (l_traj_idx != -1) & (l_clamp_idx != -1) & (r_cam_idx != -1) & (r_traj_idx != -1) & (r_clamp_idx != -1)
    valid_master_fidx = master_fidx[valid_master_mask]
    valid_r_cam_fidx = r_cam_fidx[r_cam_idx[valid_master_mask]]
    
    left_frames = extract_frames_sequential(left["video_path"], valid_master_fidx.tolist())
    right_frames = extract_frames_sequential(right["video_path"], valid_r_cam_fidx.tolist())

    return (
        master_ts,
        master_fidx,
        l_traj_arr,
        l_clamp_arr,
        r_traj_arr,
        r_clamp_arr,
        l_traj_idx,
        l_clamp_idx,
        r_traj_idx,
        r_clamp_idx,
        left_frames,
        right_frames,
        r_cam_fidx,
        r_cam_idx,
    )


def prepare_single_alignment(single: dict, target_fps: int, source_camera_fps: int, tolerance_s: float = 1e-4):
    """Prepare timestamp/trajectory indices and pre-extract frames for single-arm sessions.

    Args:
        single: dict returned by `load_arm_data` for the single arm.
        target_fps: desired output frequency in Hz.
        source_camera_fps: original camera frequency in Hz.
        tolerance_s: time alignment tolerance, timestamps exceeding this value are marked as invalid

    Returns:
        Tuple containing:
        - master_ts (np.ndarray): timestamps used as master timeline
        - master_fidx (np.ndarray): frame indices on the master camera
        - traj_arr: trajectory array (N,7)
        - clamp_arr: clamp values (N,)
        - traj_idx, clamp_idx: precomputed index arrays mapping master_ts -> traj/clamp rows
        - frames: dict mapping frame_index->frame (BGR)
    """
    master = single["timestamps"].iloc[:: max(1, int(source_camera_fps / target_fps))].reset_index(drop=True)
    master_ts = master["timestamp"].to_numpy()
    master_fidx = master["frame_index"].to_numpy().astype(int)

    # ---------------------- Single-arm data processing ----------------------
    traj_ts = single["traj"]["timestamp"].to_numpy()
    clamp_ts = single["clamp"]["timestamp"].to_numpy()
    
    # 1. Sort timestamps and retain original indices
    sorted_traj_ts, sorted_traj_original_indices = sort_timestamps_with_original_indices(traj_ts)
    sorted_clamp_ts, sorted_clamp_original_indices = sort_timestamps_with_original_indices(clamp_ts)
    
    # 2. Call the modified find_nearest_indices to get indices and alignment differences
    traj_idx_sorted, traj_diffs = find_nearest_indices(sorted_traj_ts, master_ts)
    clamp_idx_sorted, clamp_diffs = find_nearest_indices(sorted_clamp_ts, master_ts)
    
    # 3. Tolerance filtering: mark indices exceeding tolerance_s as -1
    traj_invalid_mask = traj_diffs > tolerance_s
    clamp_invalid_mask = clamp_diffs > tolerance_s
    traj_idx_sorted[traj_invalid_mask] = -1
    clamp_idx_sorted[clamp_invalid_mask] = -1
    
    # 4. Map back to original data indices
    traj_idx = np.full_like(traj_idx_sorted, -1)
    valid_traj_mask = traj_idx_sorted != -1
    traj_idx[valid_traj_mask] = sorted_traj_original_indices[traj_idx_sorted[valid_traj_mask]]
    
    clamp_idx = np.full_like(clamp_idx_sorted, -1)
    valid_clamp_mask = clamp_idx_sorted != -1
    clamp_idx[valid_clamp_mask] = sorted_clamp_original_indices[clamp_idx_sorted[valid_clamp_mask]]
    
    # 5. Log warnings
    traj_invalid_count = np.sum(traj_invalid_mask)
    clamp_invalid_count = np.sum(clamp_invalid_mask)
    if traj_invalid_count > 0:
        print(f"[Warning] Single arm trajectory {traj_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    if clamp_invalid_count > 0:
        print(f"[Warning] Single arm clamp {clamp_invalid_count} timestamps have alignment differences exceeding tolerance {tolerance_s}s and are marked as invalid")
    # ----------------------------------------------------------------------------------------------

    # Extract trajectory/clamp arrays
    traj_arr = single["traj"][['Pos X','Pos Y','Pos Z','Q_X','Q_Y','Q_Z','Q_W']].to_numpy()
    clamp_arr = single["clamp"]["clamp"].to_numpy()

    # Extract valid frame indices
    valid_master_mask = (traj_idx != -1) & (clamp_idx != -1)
    valid_master_fidx = master_fidx[valid_master_mask]
    frames = extract_frames_sequential(single["video_path"], valid_master_fidx.tolist())

    return master_ts, master_fidx, traj_arr, clamp_arr, traj_idx, clamp_idx, frames


@dataclasses.dataclass(frozen=True)
class DatasetConfig:
    use_videos: bool = True
    tolerance_s: float = 1  # Time alignment tolerance, implemented in alignment logic
    image_writer_processes: int = 10
    image_writer_threads: int = 5
    video_backend: str | None = "pyav"


STATE_NAMES_8 = [
    "x",
    "y",
    "z",
    "qx",
    "qy",
    "qz",
    "qw",
    "gripper_width",
]


def create_empty_dataset(
    *,
    repo_id: str,
    fps: int,
    mode: Literal["video", "image"] = "video",
    robot_type: str = "fastumi",
    bimanual: bool = False,
    dataset_config: DatasetConfig = DatasetConfig(),
) -> LeRobotDataset:
    """Create an empty LeRobotDataset with the expected features for this repo.

    The function mirrors the implementation used in the HDF5->LeRobot converter
    so the produced dataset is compatible.
    """
    features: dict = {}
    if not bimanual:
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES_8,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (8,),
            "names": STATE_NAMES_8,
        }
    else:
        state_names_16 = [f"robot_0_{n}" for n in STATE_NAMES_8] + [f"robot_1_{n}" for n in STATE_NAMES_8]
        features["observation.state"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names_16,
        }
        features["action"] = {
            "dtype": "float32",
            "shape": (16,),
            "names": state_names_16,
        }
        features["robot_0_action"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}
        features["robot_1_action"] = {"dtype": "float32", "shape": (8,), "names": STATE_NAMES_8}

    cams = ["robot_0", "robot_1"] if bimanual else ["robot_0"]
    for cam in cams:
        feat = {
            "dtype": mode,
            "shape": (224, 224, 3),
            "names": ["height", "width", "channels"],
        }
        if mode == "video":
            feat["info"] = {
                "video.height": 224,
                "video.width": 224,
                "video.codec": "libx264",
                "video.pix_fmt": "yuv420p",
                "video.is_depth_map": False,
                "video.fps": fps,
                "video.channels": 3,
                "has_audio": False,
            }
        features[f"observation.images.{cam}"] = feat

    dataset_dir = LEROBOT_HOME / repo_id
    if dataset_dir.exists():
        print(f"[Warning] Dataset directory {dataset_dir} already exists, it will be deleted and recreated")
        try:
            shutil.rmtree(dataset_dir)
        except Exception as e:
            raise RuntimeError(f"Failed to delete existing dataset directory {dataset_dir}: {str(e)}. Please check directory permissions.") from e
    return LeRobotDataset.create(
        repo_id=repo_id,
        fps=fps,
        robot_type=robot_type,
        features=features,
        use_videos=dataset_config.use_videos,
        tolerance_s=dataset_config.tolerance_s,
        image_writer_processes=dataset_config.image_writer_processes,
        image_writer_threads=dataset_config.image_writer_threads,
        video_backend=dataset_config.video_backend,
    )

# ---------------------------------------------------------------------------
# added function: process a single session and return valid frame data
# ---------------------------------------------------------------------------
def process_single_session(
    session_path: str,
    bimanual: bool,
    traj_source: str,
    target_fps: int,
    source_camera_fps: int,
    tolerance_s: float,
    task_name: str,
    mode: str
) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Process a single session and return valid frame data.
    Args:
        All parameters are consistent with the original function to ensure complete parameter passing.
    Returns:
        (valid_frame_count, frame_data_list): Number of valid frames and the list of frame data to be written.
    """
    frame_data_list = []
    valid = 0

    print(f"[Info] Starting to process Session: {session_path}")
    layout, paths = detect_layout(session_path)
    if layout == "invalid":
        print(f"[SKIP] Invalid session layout: {session_path}")
        return valid, frame_data_list

    if layout == "dual":
        left = load_arm_data(paths["left"], traj_source)
        right = load_arm_data(paths["right"], traj_source)
        if left is None or right is None:
            print(f"[SKIP] Bimanual session {session_path} missing left/right arm data, skipping")
            return valid, frame_data_list

        # Call alignment function
        try:
            (
                master_ts,
                master_fidx,
                l_traj_arr,
                l_clamp_arr,
                r_traj_arr,
                r_clamp_arr,
                l_traj_idx,
                l_clamp_idx,
                r_traj_idx,
                r_clamp_idx,
                left_frames,
                right_frames,
                r_cam_fidx,
                r_cam_idx,
            ) = prepare_bimanual_alignment(left, right, target_fps, source_camera_fps, tolerance_s)
        except Exception as e:
            print(f"[SKIP] Bimanual session {session_path} alignment failed: {str(e)}, skipping")
            return valid, frame_data_list

        for i in range(len(master_ts)):
            # Skip invalid indices
            if (l_traj_idx[i] == -1) or (l_clamp_idx[i] == -1) or (r_traj_idx[i] == -1) or (r_clamp_idx[i] == -1) or (r_cam_idx[i] == -1):
                continue
            
            l_fidx = master_fidx[i]
            r_fidx = r_cam_fidx[r_cam_idx[i]]
            if l_fidx not in left_frames or r_fidx not in right_frames:
                continue
            # Validate frame data
            frame_l = _to_rgb_and_resize(left_frames[l_fidx])
            frame_r = _to_rgb_and_resize(right_frames[r_fidx])
            if frame_l is None or frame_r is None:
                print(f"[Warning] Session {session_path} frame {l_fidx}/{r_fidx} invalid, skipping")
                continue

            l_idx_t = l_traj_idx[i]
            l_idx_c = l_clamp_idx[i]
            r_idx_t = r_traj_idx[i]
            r_idx_c = r_clamp_idx[i]

            l_pos = list(l_traj_arr[l_idx_t]) + [float(l_clamp_arr[l_idx_c])]
            r_pos = list(r_traj_arr[r_idx_t]) + [float(r_clamp_arr[r_idx_c])]

            state16 = np.asarray(l_pos + r_pos, dtype=np.float32)
            action16 = state16.copy()

            frame_data = {
                "observation.state": state16,
                "action": action16,
                "observation.images.robot_0": frame_l,
                "observation.images.robot_1": frame_r,
                "robot_0_action": np.asarray(l_pos, dtype=np.float32),
                "robot_1_action": np.asarray(r_pos, dtype=np.float32),
            }

            frame_data_list.append(frame_data)
            valid += 1

    else:  # single
        single = load_arm_data(paths["single"], traj_source)
        if single is None:
            print(f"[SKIP] Single-arm session {session_path} missing data, skipping")
            return valid, frame_data_list

        # Call alignment function
        try:
            (
                master_ts,
                master_fidx,
                traj_arr,
                clamp_arr,
                traj_idx,
                clamp_idx,
                frames,
            ) = prepare_single_alignment(single, target_fps, source_camera_fps, tolerance_s)
        except Exception as e:
            print(f"[SKIP] Single-arm session {session_path} alignment failed: {str(e)}, skipping")
            return valid, frame_data_list

        for i in range(len(master_ts)):
            # Skip invalid indices
            if traj_idx[i] == -1 or clamp_idx[i] == -1:
                continue
            
            fidx = master_fidx[i]
            if fidx not in frames:
                continue
            # Validate frame data
            img = _to_rgb_and_resize(frames[fidx])
            if img is None:
                print(f"[Warning] Session {session_path} frame {fidx} invalid, skipping")
                continue
            
            pos_quat = list(traj_arr[traj_idx[i]]) + [float(clamp_arr[clamp_idx[i]])]

            state8 = np.asarray(pos_quat, dtype=np.float32)
            frame_data = {
                "observation.state": state8,
                "action": state8,
                "observation.images.robot_0": img,
            }

            frame_data_list.append(frame_data)
            valid += 1

    print(f"[Info] Completed processing Session: {session_path}, new valid frames: {valid}")
    return valid, frame_data_list


def convert_raw_to_lerobot(
    task_root: Path,
    repo_id: str,
    task_name: str,
    traj_source: str,
    target_fps: int,
    source_camera_fps: int,
    repo_output: Optional[Path],
    mode: str = "video",
    max_workers: Optional[int] = None,  # New: parallel process count configuration
):
    """Main conversion routine (with added parallel support).

    This will:
    - discover sessions using `find_all_sessions` from `raw2hdf5_new.py`;
    - for each session, load arm data and align timestamps; extract frames
      sequentially and build per-frame records; add frames to the
      LeRobot dataset and save episodes.
    """
    _ensure_module_path()

    task_root = Path(task_root).expanduser().resolve()
    sessions = find_all_sessions(str(task_root))
    if not sessions:
        raise RuntimeError(f"No sessions found under {task_root}")

    # Initialize DatasetConfig (get tolerance value)
    dataset_config = DatasetConfig()
    tolerance_s = dataset_config.tolerance_s  # Extract tolerance value for subsequent alignment

    # Detect session layout (single-arm/dual-arm)
    first_layout, _ = detect_layout(sessions[0])
    bimanual = first_layout == "dual"
    print(f"[Info] Detected session layout: {'Bimanual' if bimanual else 'Single-arm'}, processing accordingly")

    # Create empty dataset
    dataset = create_empty_dataset(repo_id=repo_id, fps=target_fps, mode=mode, bimanual=bimanual, dataset_config=dataset_config)

    total_frames = 0

    # ---------------------------------------------------------------------------
    # Refactor: Parallel processing of Sessions (core acceleration part)
    # ---------------------------------------------------------------------------
    # Configure parallel process count: default to CPU cores-1 (to avoid full load), user can specify manually
    if max_workers is None:
        max_workers = max(1, cpu_count()//5)
    print(f"[Info] Starting parallel processing, number of processes: {max_workers}, number of Sessions to process: {len(sessions)}")

    # Construct parallel task parameters (each Session corresponds to a set of parameters)
    task_params = [
        (
            session_path,
            bimanual,
            traj_source,
            target_fps,
            source_camera_fps,
            tolerance_s,
            task_name,
            mode
        )
        for session_path in sessions
    ]

    # 使用 ProcessPoolExecutor 并行执行
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks and get futures
        future_to_session = {
            executor.submit(process_single_session, *params): params[0]
            for params in task_params
        }

        # Iterate over all completed tasks, aggregate results, and write to dataset
        for future in concurrent.futures.as_completed(future_to_session):
            session_path = future_to_session[future]
            try:
                valid_frames, frame_data_list = future.result()
                # Main process writes to dataset uniformly (to avoid multi-process competition)
                if valid_frames > 0:
                    for frame_data in frame_data_list:
                        dataset.add_frame(frame_data)
                    dataset.save_episode(task=task_name, encode_videos=True)
                    total_frames += valid_frames
            except Exception as e:
                print(f"[Error] Exception occurred while processing Session {session_path}: {str(e)}")

    # ---------------------------------------------------------------------------
    # Finalize dataset
    # ---------------------------------------------------------------------------
    print(f"\n[Info] All sessions processed, starting dataset finalization (computing statistics)")
    dataset.consolidate(run_compute_stats=True, keep_image_files=False)

    # Copy dataset to specified output path (if any)
    if repo_output is not None:
        target = Path(repo_output) / repo_id
        if target.exists():
            print(f"[Info] Target output directory {target} already exists, will delete and copy")
            try:
                shutil.rmtree(target)
            except Exception as e:
                raise RuntimeError(f"Failed to delete target output directory {target}: {str(e)}. Please check directory permissions.") from e
        
        try:
            shutil.copytree(LEROBOT_HOME / repo_id, target, dirs_exist_ok=True)
            print(f"[Info] Dataset copied to: {target}")
        except Exception as e:
            print(f"[Warning] Failed to copy dataset to {target}: {str(e)}. Dataset is still saved in the default path {LEROBOT_HOME / repo_id}")

    print(f"\nDone. Total frames added: {total_frames}")


def _parse_args():
    """Parse command-line arguments (restoring original functionality for easy direct command-line invocation)"""
    parser = argparse.ArgumentParser(description="Convert raw recorded sessions to LeRobot dataset (with parallel acceleration).")
    parser.add_argument("--task-root", type=Path, required=True, help="Path to root directory containing session folders.")
    parser.add_argument("--repo-id", type=str, required=True, help="Repo ID for the LeRobot dataset (e.g., myorg/mytask).")
    parser.add_argument("--task-name", type=str, default="Converted task", help="Name of the task (used in episode metadata).")
    parser.add_argument("--traj-source", type=str, default="merge", choices=["merge", "slam", "vive"], help="Source of trajectory data.")
    parser.add_argument("--target-fps", type=int, default=20, help="Desired output FPS for the dataset.")
    parser.add_argument("--source_camera_fps", type=int, default=60, help="FPS of the original camera recordings.")
    parser.add_argument("--mode", type=str, default="video", choices=["video", "image"], help="Dataset mode (video or image frames).")
    parser.add_argument("--output", type=Path, default=None, help="Optional path to copy the final dataset (default: LEROBOT_HOME/repo-id).")
    # Added: Number of parallel processes parameter
    parser.add_argument("--max-workers", type=int, default=None, help="Number of parallel processes (default: CPU cores - 1).")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    convert_raw_to_lerobot(
        task_root=args.task_root,
        repo_id=args.repo_id,
        task_name=args.task_name,
        # camera_names=cam_names,
        traj_source=args.traj_source,
        target_fps=args.target_fps,
        source_camera_fps=args.source_camera_fps,
        repo_output=args.output,
        mode=args.mode,
        max_workers=args.max_workers,
    )
    # # 示例用法（直接运行，已开启并行）
    # convert_raw_to_lerobot(
    #     task_root='/lumos-vePFS/suzhou/Users/jhf/data/raw_step1/20251231O001/pass_packages',
    #     repo_id= 'fastumi/20251231O001',
    #     task_name= 'Place the solid glue stick into the pen holder！',
    #     traj_source= 'merge',
    #     target_fps=20,
    #     source_camera_fps = 60,
    #     repo_output= '/lumos-vePFS/suzhou/Users/jhf/data/lerobot',
    #     mode= 'video',
    #     max_workers=40,  # 可根据自身CPU核心数调整，例如8核CPU可设为6或7
    # )