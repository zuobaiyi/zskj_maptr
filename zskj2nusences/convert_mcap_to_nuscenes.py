#!/usr/bin/env python3
"""
Convert extracted MCAP data to nuScenes format.

This script converts the extracted MCAP data (images, point clouds, CAN bus)
into nuScenes format that can be used with MapTR and other nuScenes-based models.

Usage:
    python inference_only/convert_mcap_to_nuscenes.py \
        --extracted-dir /path/to/extracted \
        --output-dir /path/to/nuscenes \
        [--calib-dir /path/to/calibration] \
        [--scene-name scene_001]
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
from collections import defaultdict
import mmcv
from pyquaternion import Quaternion
import hashlib
import uuid

# Camera name mapping from MCAP to nuScenes
CAMERA_MAPPING = {
    'cam_front_mid': 'CAM_FRONT',
    'cam_front_left': 'CAM_FRONT_LEFT',
    'cam_front_right': 'CAM_FRONT_RIGHT',
    'cam_back': 'CAM_BACK',
    'cam_back_left': 'CAM_BACK_LEFT',
    'cam_back_right': 'CAM_BACK_RIGHT',
    # Additional cameras (will be mapped if available)
    'cam_front_top': 'CAM_FRONT',  # Map to front if no front_mid
    'cam_near_left': 'CAM_FRONT_LEFT',  # Map to front_left if available
    'cam_near_mid': 'CAM_FRONT',  # Map to front if available
    'cam_near_right': 'CAM_FRONT_RIGHT',  # Map to front_right if available
}

# nuScenes standard camera order
NUSCENES_CAMERAS = [
    'CAM_FRONT',
    'CAM_FRONT_RIGHT',
    'CAM_FRONT_LEFT',
    'CAM_BACK',
    'CAM_BACK_LEFT',
    'CAM_BACK_RIGHT',
]


def generate_token() -> str:
    """Generate a random token (similar to nuScenes format)."""
    return hashlib.md5(str(uuid.uuid4()).encode()).hexdigest()[:16]


def load_extraction_summary(extracted_dir: Path) -> Dict:
    """Load extraction summary JSON."""
    summary_path = extracted_dir / 'extraction_summary.json'
    if not summary_path.exists():
        raise FileNotFoundError(f"Extraction summary not found: {summary_path}")
    with open(summary_path, 'r') as f:
        return json.load(f)


def load_can_bus_data(extracted_dir: Path) -> List[Dict]:
    """Load CAN bus messages from JSON file."""
    can_bus_path = extracted_dir / 'can_bus' / 'can_bus_messages.json'
    if not can_bus_path.exists():
        print(f"Warning: CAN bus file not found: {can_bus_path}")
        return []
    
    with open(can_bus_path, 'r') as f:
        return json.load(f)


def get_image_files(extracted_dir: Path, camera_name: str) -> List[Tuple[str, int]]:
    """Get all image files for a camera, sorted by timestamp.
    
    Returns:
        List of (file_path, timestamp_ns) tuples
    """
    camera_dir = extracted_dir / 'images' / camera_name
    if not camera_dir.exists():
        return []
    
    images = []
    for img_file in sorted(camera_dir.glob('*.jpg')):
        # Extract timestamp from filename (e.g., "1767688184164185143.jpg")
        try:
            timestamp_ns = int(img_file.stem)
            images.append((str(img_file), timestamp_ns))
        except ValueError:
            print(f"Warning: Could not parse timestamp from {img_file.name}")
            continue
    
    return sorted(images, key=lambda x: x[1])


def synchronize_frames(extracted_dir: Path, max_delta_ms: int = 100) -> List[Dict]:
    """Synchronize images from different cameras into frames.
    
    Args:
        extracted_dir: Path to extracted data directory
        max_delta_ms: Maximum time delta in milliseconds for synchronization
    
    Returns:
        List of synchronized frames, each containing:
        {
            'timestamp_ns': int,
            'timestamp_sec': float,
            'images': {camera_name: image_path},
            'image_timestamps': {camera_name: timestamp_ns}
        }
    """
    # Get all camera directories
    images_dir = extracted_dir / 'images'
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    
    # Collect all images from all cameras
    all_images = {}
    for camera_dir in images_dir.iterdir():
        if not camera_dir.is_dir():
            continue
        
        camera_name = camera_dir.name
        images = get_image_files(extracted_dir, camera_name)
        if images:
            all_images[camera_name] = images
    
    if not all_images:
        raise ValueError("No images found in extracted directory")
    
    print(f"Found images from {len(all_images)} cameras")
    for cam, imgs in all_images.items():
        print(f"  {cam}: {len(imgs)} images")
    
    # Use the camera with most images as reference
    ref_camera = max(all_images.keys(), key=lambda k: len(all_images[k]))
    ref_images = all_images[ref_camera]
    
    print(f"Using {ref_camera} as reference camera ({len(ref_images)} images)")
    
    # Synchronize frames
    frames = []
    max_delta_ns = max_delta_ms * 1_000_000
    
    for ref_path, ref_ts in ref_images:
        frame = {
            'timestamp_ns': ref_ts,
            'timestamp_sec': ref_ts / 1e9,
            'images': {},
            'image_timestamps': {}
        }
        
        # Add reference camera image
        frame['images'][ref_camera] = ref_path
        frame['image_timestamps'][ref_camera] = ref_ts
        
        # Find matching images from other cameras
        for cam_name, cam_images in all_images.items():
            if cam_name == ref_camera:
                continue
            
            # Find nearest image by timestamp
            best_match = None
            best_delta = float('inf')
            
            for img_path, img_ts in cam_images:
                delta = abs(img_ts - ref_ts)
                if delta < best_delta and delta <= max_delta_ns:
                    best_delta = delta
                    best_match = (img_path, img_ts)
            
            if best_match:
                frame['images'][cam_name] = best_match[0]
                frame['image_timestamps'][cam_name] = best_match[1]
        
        frames.append(frame)
    
    print(f"Synchronized {len(frames)} frames")
    return frames


def load_calibration(calib_dir: Optional[Path]) -> Dict[str, Dict]:
    """Load calibration data.
    
    Returns:
        Dict mapping camera names to calibration data:
        {
            'translation': [x, y, z],
            'rotation': [w, x, y, z],  # quaternion
            'camera_intrinsic': [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        }
    """
    calib_data = {}
    
    if calib_dir is None or not calib_dir.exists():
        print("Warning: No calibration directory provided, using default values")
        return calib_data
    
    # Try to load from individual camera files
    for cam_file in calib_dir.glob('*.json'):
        cam_name = cam_file.stem
        try:
            with open(cam_file, 'r') as f:
                data = json.load(f)
                calib_data[cam_name] = data
        except Exception as e:
            print(f"Warning: Could not load calibration from {cam_file}: {e}")
    
    # Try to load from calibrated_sensor.json
    calib_json = calib_dir / 'calibrated_sensor.json'
    if calib_json.exists():
        try:
            with open(calib_json, 'r') as f:
                sensors = json.load(f)
                if isinstance(sensors, list):
                    for sensor in sensors:
                        # Extract camera name from sensor_token or use index
                        # This is a simplified mapping
                        calib_data[f'sensor_{len(calib_data)}'] = sensor
        except Exception as e:
            print(f"Warning: Could not load calibration from {calib_json}: {e}")
    
    return calib_data


def create_nuscenes_structure(output_dir: Path, version: str = 'v1.0-trainval'):
    """Create nuScenes directory structure."""
    dirs = [
        output_dir / 'samples',
        output_dir / version,
        output_dir / 'can_bus',
    ]
    
    # Create camera subdirectories in samples
    for camera in NUSCENES_CAMERAS:
        dirs.append(output_dir / 'samples' / camera.lower())
    
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
    
    print(f"Created nuScenes directory structure in {output_dir}")


def copy_images_to_nuscenes(frames: List[Dict], extracted_dir: Path, 
                            output_dir: Path) -> Tuple[Dict[str, Dict], List[Dict], Dict[int, str]]:
    """Copy images to nuScenes format and create sample_data records.
    
    Returns:
        Tuple of:
        - Dict mapping (camera_name, timestamp_ns) to sample_data record
        - List of all sample_data records
        - Dict mapping timestamp_ns to LIDAR_TOP sample_data token
    """
    sample_data_records = {}
    sample_data_list = []
    lidar_top_tokens = {}  # timestamp_ns -> token
    
    for frame_idx, frame in enumerate(frames):
        timestamp_ns = frame['timestamp_ns']
        
        # Create a fake LIDAR_TOP sample_data entry (required by nuScenes)
        lidar_token = generate_token()
        lidar_top_tokens[timestamp_ns] = lidar_token
        
        lidar_sample_data = {
            'token': lidar_token,
            'sample_token': '',  # Will be filled later
            'ego_pose_token': '',  # Will be filled later
            'calibrated_sensor_token': '',  # Will be filled later
            'timestamp': timestamp_ns,
            'fileformat': 'pcd',
            'is_key_frame': True,
            'filename': f'sweeps/LIDAR_TOP/{timestamp_ns}.pcd',  # Fake path
            'prev': '',
            'next': '',
            'sensor_modality': 'lidar',
        }
        sample_data_list.append(lidar_sample_data)
        
        for mcap_cam, img_path in frame['images'].items():
            # Map camera name to nuScenes format
            nuscenes_cam = CAMERA_MAPPING.get(mcap_cam, mcap_cam.upper())
            
            # Skip if not a standard nuScenes camera
            if nuscenes_cam not in NUSCENES_CAMERAS:
                continue
            
            # Create sample_data token
            sample_data_token = generate_token()
            
            # Copy image to nuScenes format
            # nuScenes format: samples/CAM_FRONT/n015-2018-07-24-11-22-45+0800__CAM_FRONT__1532402927642405.jpg
            # Simplified: samples/CAM_FRONT/{timestamp_ns}.jpg
            img_filename = f"{timestamp_ns}.jpg"
            dest_path = output_dir / 'samples' / nuscenes_cam.lower() / img_filename
            
            # Copy image
            shutil.copy2(img_path, dest_path)
            
            # Read image dimensions
            try:
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    height, width = img.shape[:2]
                else:
                    height, width = 900, 1600  # Default nuScenes size
            except:
                height, width = 900, 1600  # Default nuScenes size
            
            # Create sample_data record
            sample_data_record = {
                'token': sample_data_token,
                'sample_token': '',  # Will be filled later
                'ego_pose_token': '',  # Will be filled later
                'calibrated_sensor_token': '',  # Will be filled later
                'timestamp': timestamp_ns,
                'fileformat': 'jpg',
                'is_key_frame': True,
                'height': height,
                'width': width,
                'filename': f'samples/{nuscenes_cam.lower()}/{img_filename}',
                'prev': '',
                'next': '',
                'sensor_modality': 'camera',
            }
            
            sample_data_records[(nuscenes_cam, timestamp_ns)] = sample_data_record
            sample_data_list.append(sample_data_record)
    
    print(f"Copied {len([s for s in sample_data_list if s['sensor_modality'] == 'camera'])} images to nuScenes format")
    print(f"Created {len(lidar_top_tokens)} fake LIDAR_TOP entries")
    return sample_data_records, sample_data_list, lidar_top_tokens


def create_nuscenes_metadata(frames: List[Dict], sample_data_records: Dict,
                             lidar_top_tokens: Dict[int, str],
                             all_sample_data_list: List[Dict],
                             output_dir: Path, calib_data: Dict,
                             can_bus_messages: List[Dict], version: str = 'v1.0-trainval') -> Dict:
    """Create nuScenes metadata JSON files.
    
    Creates:
        - sample.json
        - sample_data.json
        - calibrated_sensor.json
        - ego_pose.json
        - scene.json
        - log.json
    """
    metadata_dir = output_dir / version
    
    # Create a mapping from token to sample_data record for easy lookup
    sample_data_by_token = {sd['token']: sd for sd in all_sample_data_list}
    
    # Create samples
    samples = []
    ego_poses = []
    calibrated_sensors = {}
    scene_token = generate_token()
    log_token = generate_token()
    
    # Create a single scene
    scene = {
        'token': scene_token,
        'name': 'scene_001',
        'description': 'Converted from MCAP data',
        'log_token': log_token,
        'nbr_samples': len(frames),
        'first_sample_token': '',
        'last_sample_token': '',
    }
    
    # Create log
    log = {
        'token': log_token,
        'date_captured': '2026-01-06',
        'location': 'unknown',
        'vehicle': 'unknown',
        'logfile': 'rosbag2_2026_01_06-16_29_42',
    }
    
    # Create calibrated sensors for each camera
    for camera in NUSCENES_CAMERAS:
        calib_token = generate_token()
        sensor_token = generate_token()
        
        # Try to get calibration from calib_data
        calib = calib_data.get(camera.lower(), {})
        
        calibrated_sensor = {
            'token': calib_token,
            'sensor_token': sensor_token,
            'translation': calib.get('translation', [0.0, 0.0, 0.0]),
            'rotation': calib.get('rotation', [1.0, 0.0, 0.0, 0.0]),
            'camera_intrinsic': calib.get('camera_intrinsic', []),
        }
        
        calibrated_sensors[camera] = {
            'calibrated_sensor_token': calib_token,
            'sensor_token': sensor_token,
            'calibrated_sensor': calibrated_sensor,
        }
    
    # Create samples and sample_data
    prev_sample_token = ''
    for frame_idx, frame in enumerate(frames):
        timestamp_ns = frame['timestamp_ns']
        sample_token = generate_token()
        
        # Create ego pose
        ego_pose_token = generate_token()
        ego_pose = {
            'token': ego_pose_token,
            'translation': [0.0, 0.0, 0.0],  # Will be filled from CAN bus if available
            'rotation': [1.0, 0.0, 0.0, 0.0],  # Will be filled from CAN bus if available
            'timestamp': timestamp_ns,
        }
        ego_poses.append(ego_pose)
        
        # Try to get ego pose from CAN bus
        # Find nearest CAN bus message
        for can_msg in can_bus_messages:
            if can_msg.get('topic') == '/localization/global_fusion/Location':
                can_ts = can_msg.get('timestamp_ns', 0)
                if abs(can_ts - timestamp_ns) < 100_000_000:  # Within 100ms
                    # Extract position and orientation if available
                    # This requires parsing the actual CAN bus message
                    break
        
        # Create sample
        sample = {
            'token': sample_token,
            'timestamp': timestamp_ns,
            'scene_token': scene_token,
            'prev': prev_sample_token,
            'next': '',
            'data': {},
            'anns': [],  # No annotations
        }
        
        # Add LIDAR_TOP (required by nuScenes)
        if timestamp_ns in lidar_top_tokens:
            lidar_token = lidar_top_tokens[timestamp_ns]
            # Find the LIDAR sample_data record
            lidar_sd = sample_data_by_token.get(lidar_token)
            if lidar_sd:
                lidar_sd['sample_token'] = sample_token
                lidar_sd['ego_pose_token'] = ego_pose_token
                # Use a default calibrated sensor token for lidar (will create one if needed)
                if 'LIDAR_TOP' not in calibrated_sensors:
                    calib_token = generate_token()
                    sensor_token = generate_token()
                    calibrated_sensors['LIDAR_TOP'] = {
                        'calibrated_sensor_token': calib_token,
                        'sensor_token': sensor_token,
                        'calibrated_sensor': {
                            'token': calib_token,
                            'sensor_token': sensor_token,
                            'translation': [0.0, 0.0, 0.0],
                            'rotation': [1.0, 0.0, 0.0, 0.0],
                        }
                    }
                lidar_sd['calibrated_sensor_token'] = calibrated_sensors['LIDAR_TOP']['calibrated_sensor_token']
                sample['data']['LIDAR_TOP'] = lidar_token
        
        # Add sample_data for each camera
        for camera in NUSCENES_CAMERAS:
            key = (camera, timestamp_ns)
            if key in sample_data_records:
                sd_rec = sample_data_records[key]
                sd_rec['sample_token'] = sample_token
                sd_rec['ego_pose_token'] = ego_pose_token
                sd_rec['calibrated_sensor_token'] = calibrated_sensors[camera]['calibrated_sensor_token']
                
                sample['data'][camera] = sd_rec['token']
                # Don't append here, already in sample_data_list
        
        # Update prev/next
        if prev_sample_token:
            prev_sample = next(s for s in samples if s['token'] == prev_sample_token)
            prev_sample['next'] = sample_token
        
        samples.append(sample)
        prev_sample_token = sample_token
        
        if frame_idx == 0:
            scene['first_sample_token'] = sample_token
        if frame_idx == len(frames) - 1:
            scene['last_sample_token'] = sample_token
    
    # Write JSON files
    with open(metadata_dir / 'sample.json', 'w') as f:
        json.dump(samples, f, indent=2)
    
    with open(metadata_dir / 'sample_data.json', 'w') as f:
        json.dump(all_sample_data_list, f, indent=2)
    
    # Include LIDAR_TOP calibrated sensor if it exists
    all_calibrated_sensors = [cs['calibrated_sensor'] for cs in calibrated_sensors.values()]
    if 'LIDAR_TOP' in calibrated_sensors:
        all_calibrated_sensors.append(calibrated_sensors['LIDAR_TOP']['calibrated_sensor'])
    with open(metadata_dir / 'calibrated_sensor.json', 'w') as f:
        json.dump(all_calibrated_sensors, f, indent=2)
    
    with open(metadata_dir / 'ego_pose.json', 'w') as f:
        json.dump(ego_poses, f, indent=2)
    
    with open(metadata_dir / 'scene.json', 'w') as f:
        json.dump([scene], f, indent=2)
    
    with open(metadata_dir / 'log.json', 'w') as f:
        json.dump([log], f, indent=2)
    
    # Create minimal required metadata files for nuScenes
    # These are static files that nuScenes library expects
    
    # category.json - object categories
    categories = [
        {'token': generate_token(), 'name': 'animal', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.adult', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.child', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.construction_worker', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.police_officer', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.stroller', 'description': ''},
        {'token': generate_token(), 'name': 'human.pedestrian.wheelchair', 'description': ''},
        {'token': generate_token(), 'name': 'movable_object.barrier', 'description': ''},
        {'token': generate_token(), 'name': 'movable_object.debris', 'description': ''},
        {'token': generate_token(), 'name': 'movable_object.pushable_pullable', 'description': ''},
        {'token': generate_token(), 'name': 'movable_object.trafficcone', 'description': ''},
        {'token': generate_token(), 'name': 'static_object.bicycle_rack', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.bicycle', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.bus.bendy', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.bus.rigid', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.car', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.construction', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.emergency.ambulance', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.emergency.police', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.motorcycle', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.trailer', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.truck', 'description': ''},
        {'token': generate_token(), 'name': 'flat.driveable_surface', 'description': ''},
        {'token': generate_token(), 'name': 'flat.other', 'description': ''},
        {'token': generate_token(), 'name': 'flat.sidewalk', 'description': ''},
        {'token': generate_token(), 'name': 'flat.terrain', 'description': ''},
        {'token': generate_token(), 'name': 'static.manmade', 'description': ''},
        {'token': generate_token(), 'name': 'static.other', 'description': ''},
        {'token': generate_token(), 'name': 'static.vegetation', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.ego', 'description': ''},
    ]
    with open(metadata_dir / 'category.json', 'w') as f:
        json.dump(categories, f, indent=2)
    
    # attribute.json - object attributes
    attributes = [
        {'token': generate_token(), 'name': 'vehicle.moving', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.parked', 'description': ''},
        {'token': generate_token(), 'name': 'vehicle.stopped', 'description': ''},
        {'token': generate_token(), 'name': 'cycle.with_rider', 'description': ''},
        {'token': generate_token(), 'name': 'cycle.without_rider', 'description': ''},
        {'token': generate_token(), 'name': 'pedestrian.moving', 'description': ''},
        {'token': generate_token(), 'name': 'pedestrian.standing', 'description': ''},
        {'token': generate_token(), 'name': 'pedestrian.sitting_lying_down', 'description': ''},
    ]
    with open(metadata_dir / 'attribute.json', 'w') as f:
        json.dump(attributes, f, indent=2)
    
    # visibility.json - visibility levels
    visibilities = [
        {'token': generate_token(), 'level': '', 'description': ''},
        {'token': generate_token(), 'level': '1', 'description': ''},
        {'token': generate_token(), 'level': '2', 'description': ''},
        {'token': generate_token(), 'level': '3', 'description': ''},
        {'token': generate_token(), 'level': '4', 'description': ''},
    ]
    with open(metadata_dir / 'visibility.json', 'w') as f:
        json.dump(visibilities, f, indent=2)
    
    # instance.json - empty (no instances/annotations)
    with open(metadata_dir / 'instance.json', 'w') as f:
        json.dump([], f, indent=2)
    
    # sample_annotation.json - empty (no annotations)
    with open(metadata_dir / 'sample_annotation.json', 'w') as f:
        json.dump([], f, indent=2)
    
    # map.json - minimal map (nuScenes library requires at least one map)
    map_token = generate_token()
    map_obj = {
        'token': map_token,
        'log_tokens': [log_token],
        'category': '',
        'filename': '',
    }
    with open(metadata_dir / 'map.json', 'w') as f:
        json.dump([map_obj], f, indent=2)
    
    # sensor.json - sensor definitions
    sensors = []
    for camera in NUSCENES_CAMERAS:
        sensor_token = calibrated_sensors[camera]['sensor_token']
        sensors.append({
            'token': sensor_token,
            'channel': camera,
            'modality': 'camera',
        })
    # Add lidar sensor
    lidar_token = generate_token()
    sensors.append({
        'token': lidar_token,
        'channel': 'LIDAR_TOP',
        'modality': 'lidar',
    })
    with open(metadata_dir / 'sensor.json', 'w') as f:
        json.dump(sensors, f, indent=2)
    
    print(f"Created nuScenes metadata files in {metadata_dir}")
    print(f"  - {len(samples)} samples")
    print(f"  - {len(sample_data_list)} sample_data records")
    print(f"  - {len(ego_poses)} ego poses")
    print(f"  - {len(calibrated_sensors)} calibrated sensors")
    
    return {
        'samples': samples,
        'sample_data': sample_data_list,
        'ego_poses': ego_poses,
        'calibrated_sensors': calibrated_sensors,
        'scenes': [scene],
        'logs': [log],
    }


def create_can_bus_data(can_bus_messages: List[Dict], output_dir: Path, 
                       scene_name: str, can_bus_root: Path):
    """Create nuScenes CAN bus data format.
    
    nuScenes expects CAN bus data in data/can_bus/{scene_name}/pose.json format.
    """
    # Create CAN bus directory in the root (not in nuscenes directory)
    can_bus_dir = can_bus_root / 'can_bus' / scene_name
    can_bus_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract pose data from CAN bus messages
    # Look for localization/odometry messages
    pose_messages = []
    for msg in can_bus_messages:
        topic = msg.get('topic', '')
        # Look for odometry or localization topics
        if 'odom' in topic.lower() or 'localization' in topic.lower() or 'location' in topic.lower():
            # Create pose-like structure
            # Note: This is a simplified version - actual parsing requires ROS 2 message definitions
            pose_msg = {
                'utime': msg.get('timestamp_ns', 0) / 1e6,  # Convert to microseconds
                'pos': [0.0, 0.0, 0.0],  # Placeholder - needs actual parsing
                'orientation': [1.0, 0.0, 0.0, 0.0],  # Placeholder - needs actual parsing
                # Add other fields that might be needed
            }
            pose_messages.append(pose_msg)
    
    # If no pose messages found, create empty list
    if not pose_messages:
        print("Warning: No pose messages found in CAN bus data, creating empty pose.json")
        pose_messages = []
    
    # Save pose.json (nuScenes format)
    with open(can_bus_dir / 'pose.json', 'w') as f:
        json.dump(pose_messages, f, indent=2)
    
    # Also save raw messages for reference
    with open(can_bus_dir / 'raw_messages.json', 'w') as f:
        json.dump(can_bus_messages, f, indent=2)
    
    print(f"Created CAN bus data in {can_bus_dir}")
    print(f"  - pose.json: {len(pose_messages)} pose messages")
    print(f"  - raw_messages.json: {len(can_bus_messages)} total messages")


def main():
    parser = argparse.ArgumentParser(description='Convert MCAP data to nuScenes format')
    parser.add_argument('--extracted-dir', type=str, required=True,
                       help='Path to extracted MCAP data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                       help='Path to output nuScenes directory')
    parser.add_argument('--calib-dir', type=str, default=None,
                       help='Path to calibration directory (optional)')
    parser.add_argument('--scene-name', type=str, default='scene_001',
                       help='Scene name for output')
    parser.add_argument('--max-delta-ms', type=int, default=100,
                       help='Maximum time delta in milliseconds for synchronization')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                       choices=['v1.0-trainval', 'v1.0-mini', 'v1.0-test'],
                       help='nuScenes version (default: v1.0-trainval)')
    
    args = parser.parse_args()
    
    extracted_dir = Path(args.extracted_dir)
    output_dir = Path(args.output_dir)
    calib_dir = Path(args.calib_dir) if args.calib_dir else None
    
    print(f"Converting MCAP data to nuScenes format")
    print(f"  Extracted dir: {extracted_dir}")
    print(f"  Output dir: {output_dir}")
    print(f"  Calib dir: {calib_dir}")
    
    # Load extraction summary
    summary = load_extraction_summary(extracted_dir)
    print(f"\nExtraction summary:")
    print(f"  Images: {len(summary.get('extraction_results', {}).get('images', {}))} cameras")
    print(f"  Point clouds: {len(summary.get('extraction_results', {}).get('point_clouds', {}))} sensors")
    print(f"  CAN bus: {len(summary.get('extraction_results', {}).get('can_bus', {}))} topics")
    
    # Load CAN bus data
    can_bus_messages = load_can_bus_data(extracted_dir)
    print(f"\nLoaded {len(can_bus_messages)} CAN bus messages")
    
    # Load calibration
    calib_data = load_calibration(calib_dir)
    if calib_data:
        print(f"Loaded calibration for {len(calib_data)} sensors")
    
    # Synchronize frames
    print(f"\nSynchronizing frames (max_delta_ms={args.max_delta_ms})...")
    frames = synchronize_frames(extracted_dir, max_delta_ms=args.max_delta_ms)
    
    # Create nuScenes structure
    print(f"\nCreating nuScenes directory structure...")
    create_nuscenes_structure(output_dir, version=args.version)
    
    # Copy images
    print(f"\nCopying images to nuScenes format...")
    sample_data_records, sample_data_list, lidar_top_tokens = copy_images_to_nuscenes(
        frames, extracted_dir, output_dir)
    
    # Create metadata
    print(f"\nCreating nuScenes metadata...")
    metadata = create_nuscenes_metadata(
        frames, sample_data_records, lidar_top_tokens, sample_data_list, output_dir, calib_data, can_bus_messages, version=args.version)
    
    # Create CAN bus data
    print(f"\nCreating CAN bus data...")
    can_bus_root = output_dir.parent  # CAN bus should be in data/can_bus, not data/nuscenes/can_bus
    create_can_bus_data(can_bus_messages, output_dir, args.scene_name, can_bus_root)
    
    print(f"\n✓ Conversion complete!")
    print(f"  Output directory: {output_dir}")
    print(f"  Version: {args.version}")
    print(f"  Next step: Run 'python tools/create_data.py nuscenes --root-path {output_dir} --out-dir {output_dir} --extra-tag nuscenes --version {args.version} --canbus {output_dir.parent}'")


if __name__ == '__main__':
    main()

