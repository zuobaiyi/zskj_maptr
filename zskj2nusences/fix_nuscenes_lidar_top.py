#!/usr/bin/env python3
"""Quick fix script to add LIDAR_TOP entries to existing nuScenes data."""

import json
import hashlib
import uuid
from pathlib import Path
from typing import Dict, List

def generate_token() -> str:
    """Generate a random token (similar to nuScenes format)."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]


def fix_nuscenes_data(nuscenes_dir: Path):
    """Add LIDAR_TOP entries to existing nuScenes data."""
    version_dir = nuscenes_dir / 'v1.0-mini'
    if not version_dir.exists():
        version_dir = nuscenes_dir / 'v1.0-trainval'
    
    if not version_dir.exists():
        print(f"Error: Version directory not found in {nuscenes_dir}")
        return
    
    print(f"Fixing nuScenes data in {version_dir}")
    
    # Load existing data
    with open(version_dir / 'sample.json', 'r') as f:
        samples = json.load(f)
    
    with open(version_dir / 'sample_data.json', 'r') as f:
        sample_data = json.load(f)
    
    with open(version_dir / 'calibrated_sensor.json', 'r') as f:
        calibrated_sensors = json.load(f)
    
    with open(version_dir / 'ego_pose.json', 'r') as f:
        ego_poses = json.load(f)
    
    # Create a mapping from timestamp to ego_pose_token
    timestamp_to_ego_pose = {ep['timestamp']: ep['token'] for ep in ego_poses}
    
    # Check if LIDAR_TOP calibrated sensor exists
    lidar_calib_token = None
    lidar_sensor_token = None
    for cs in calibrated_sensors:
        # Check if this is a lidar sensor (usually has empty camera_intrinsic)
        if not cs.get('camera_intrinsic') or cs.get('camera_intrinsic') == []:
            lidar_calib_token = cs['token']
            lidar_sensor_token = cs.get('sensor_token', generate_token())
            break
    
    # If no lidar sensor exists, create one
    if lidar_calib_token is None:
        lidar_calib_token = generate_token()
        lidar_sensor_token = generate_token()
        lidar_calib = {
            'token': lidar_calib_token,
            'sensor_token': lidar_sensor_token,
            'translation': [0.0, 0.0, 0.0],
            'rotation': [1.0, 0.0, 0.0, 0.0],
        }
        calibrated_sensors.append(lidar_calib)
        print(f"Created LIDAR_TOP calibrated sensor: {lidar_calib_token}")
    
    # Create LIDAR_TOP sample_data entries and update samples
    sample_data_by_token = {sd['token']: sd for sd in sample_data}
    new_sample_data = []
    updated_samples = []
    
    for sample in samples:
        timestamp = sample['timestamp']
        
        # Check if LIDAR_TOP already exists
        if 'LIDAR_TOP' in sample.get('data', {}):
            updated_samples.append(sample)
            continue
        
        # Create LIDAR_TOP sample_data entry
        lidar_token = generate_token()
        ego_pose_token = timestamp_to_ego_pose.get(timestamp, sample.get('ego_pose_token', ''))
        
        lidar_sd = {
            'token': lidar_token,
            'sample_token': sample['token'],
            'ego_pose_token': ego_pose_token,
            'calibrated_sensor_token': lidar_calib_token,
            'timestamp': timestamp,
            'fileformat': 'pcd',
            'is_key_frame': True,
            'filename': f'sweeps/LIDAR_TOP/{timestamp}.pcd',  # Fake path
            'prev': '',
            'next': '',
            'sensor_modality': 'lidar',
        }
        
        new_sample_data.append(lidar_sd)
        
        # Add LIDAR_TOP to sample data
        if 'data' not in sample:
            sample['data'] = {}
        sample['data']['LIDAR_TOP'] = lidar_token
        updated_samples.append(sample)
    
    # Add new sample_data entries
    sample_data.extend(new_sample_data)
    
    print(f"Added {len(new_sample_data)} LIDAR_TOP sample_data entries")
    print(f"Updated {len(updated_samples)} samples")
    
    # Save updated data
    with open(version_dir / 'sample.json', 'w') as f:
        json.dump(updated_samples, f, indent=2)
    
    with open(version_dir / 'sample_data.json', 'w') as f:
        json.dump(sample_data, f, indent=2)
    
    with open(version_dir / 'calibrated_sensor.json', 'w') as f:
        json.dump(calibrated_sensors, f, indent=2)
    
    print(f"✓ Fixed nuScenes data in {version_dir}")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python fix_nuscenes_lidar_top.py <nuscenes_dir>")
        print("Example: python fix_nuscenes_lidar_top.py /path/to/nuscenes")
        sys.exit(1)
    
    nuscenes_dir = Path(sys.argv[1])
    fix_nuscenes_data(nuscenes_dir)

