#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract actual data from MCAP (rosbag2) files.
Extracts images, point clouds, can_bus data, etc. and saves them to files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import cv2
import numpy as np
from collections import defaultdict
import sys

try:
    from mcap.reader import make_reader
    MCAP_AVAILABLE = True
except ImportError:
    MCAP_AVAILABLE = False
    print("Error: mcap library not found. Please install: pip install mcap")
    sys.exit(1)

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    ROSBAG2_AVAILABLE = True
except ImportError:
    ROSBAG2_AVAILABLE = False
    print("Warning: rosbag2_py not found. Will use raw message extraction.")

try:
    from mcap_ros2_support import read_ros2_messages
    MCAP_ROS2_AVAILABLE = True
except ImportError:
    MCAP_ROS2_AVAILABLE = False

try:
    from cv_bridge import CvBridge
    CV_BRIDGE_AVAILABLE = True
except ImportError:
    CV_BRIDGE_AVAILABLE = False
    print("Warning: cv_bridge not found. Will use basic image decoding.")

try:
    import sensor_msgs_py.point_cloud2 as pc2
    POINT_CLOUD_AVAILABLE = True
except ImportError:
    POINT_CLOUD_AVAILABLE = False
    print("Warning: sensor_msgs_py not found. Point cloud extraction may be limited.")


def extract_images_rosbag2(mcap_path: str, output_dir: Path, topics: Optional[List[str]] = None):
    """Extract images using rosbag2_py (proper ROS 2 message decoding)."""
    output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    bridge = CvBridge() if CV_BRIDGE_AVAILABLE else None
    
    topic_dirs = {}
    topic_counts = defaultdict(int)
    
    print(f"Extracting images from {mcap_path} using rosbag2_py...")
    
    storage_options = StorageOptions(uri=mcap_path, storage_id='mcap')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    
    # Filter image topics
    image_topics = []
    for topic_metadata in topic_types:
        topic_name = topic_metadata.name
        if topics and topic_name not in topics:
            continue
        if 'Image' in topic_metadata.type:
            image_topics.append(topic_name)
            topic_dir = output_dir / topic_name.replace('/', '_').strip('_')
            topic_dir.mkdir(parents=True, exist_ok=True)
            topic_dirs[topic_name] = topic_dir
    
    print(f"Found {len(image_topics)} image topics: {image_topics}")
    
    # Read messages
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        
        if topic not in image_topics:
            continue
        
        topic_dir = topic_dirs[topic]
        timestamp_ns = timestamp
        
        try:
            # Deserialize ROS 2 message
            # Note: This requires the message type to be importable
            # For CompressedImage, we can try to decode directly
            if 'CompressedImage' in topic_types[topic].type:
                # Try to extract compressed image data
                # The data is in CDR format, we need to parse it
                # For now, try to find image data in the raw bytes
                img_data = None
                if isinstance(data, bytes):
                    # Try to find JPEG/PNG header
                    for i in range(len(data) - 4):
                        if data[i:i+2] == b'\xff\xd8':  # JPEG header
                            # Find JPEG end
                            for j in range(i+2, len(data)):
                                if data[j:j+2] == b'\xff\xd9':  # JPEG end
                                    img_data = data[i:j+2]
                                    break
                            if img_data:
                                break
                        elif data[i:i+8] == b'\x89PNG\r\n\x1a\n':  # PNG header
                            # Find PNG end (IEND chunk)
                            png_end = data.find(b'IEND\xaeB`\x82', i)
                            if png_end > 0:
                                img_data = data[i:png_end+8]
                                break
                
                if img_data:
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None and img.size > 0:
                        filename = f"{int(timestamp_ns)}.jpg"
                        filepath = topic_dir / filename
                        cv2.imwrite(str(filepath), img)
                        topic_counts[topic] += 1
                        
                        if topic_counts[topic] % 100 == 0:
                            print(f"  Extracted {topic_counts[topic]} images from {topic}")
        except Exception as e:
            # Skip errors for now
            continue
    
    reader.close()
    
    print(f"\nImage extraction complete:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} images")
    
    return topic_counts


def extract_images(mcap_path: str, output_dir: Path, topics: Optional[List[str]] = None):
    """Extract images from MCAP file."""
    if ROSBAG2_AVAILABLE:
        return extract_images_rosbag2(mcap_path, output_dir, topics)
    
    # Fallback to mcap reader - extract from CDR format
    output_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    topic_dirs = {}
    topic_counts = defaultdict(int)
    
    print(f"Extracting images from {mcap_path}...")
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic
            
            if topics and topic_name not in topics:
                continue
            
            if 'Image' not in schema.name:
                continue
            
            if topic_name not in topic_dirs:
                topic_dir = output_dir / topic_name.replace('/', '_').strip('_')
                topic_dir.mkdir(parents=True, exist_ok=True)
                topic_dirs[topic_name] = topic_dir
            
            topic_dir = topic_dirs[topic_name]
            timestamp_ns = message.log_time
            
            try:
                # Get raw CDR data
                if not hasattr(message, 'data'):
                    continue
                
                raw_data = message.data
                img_data = None
                
                # For CompressedImage in CDR format:
                # CDR header (8 bytes) + format string (e.g., "jpeg") + null terminator + image data
                # Find JPEG/PNG header in the data
                jpeg_start = raw_data.find(b'\xff\xd8')
                png_start = raw_data.find(b'\x89PNG')
                
                if jpeg_start >= 0:
                    # Find JPEG end marker
                    jpeg_end = raw_data.find(b'\xff\xd9', jpeg_start + 2)
                    if jpeg_end > jpeg_start:
                        img_data = raw_data[jpeg_start:jpeg_end + 2]
                elif png_start >= 0:
                    # Find PNG IEND chunk
                    iend_pos = raw_data.find(b'IEND\xaeB`\x82', png_start)
                    if iend_pos > png_start:
                        img_data = raw_data[png_start:iend_pos + 8]
                
                if img_data:
                    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
                    
                    if img is not None and img.size > 0:
                        filename = f"{int(timestamp_ns)}.jpg"
                        filepath = topic_dir / filename
                        cv2.imwrite(str(filepath), img)
                        topic_counts[topic_name] += 1
                        
                        if topic_counts[topic_name] % 100 == 0:
                            print(f"  Extracted {topic_counts[topic_name]} images from {topic_name}")
            except Exception as e:
                continue
    
    print(f"\nImage extraction complete:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} images")
    
    return topic_counts


def extract_point_clouds(mcap_path: str, output_dir: Path, topics: Optional[List[str]] = None):
    """Extract point clouds from MCAP file."""
    output_dir = output_dir / "point_clouds"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    topic_dirs = {}
    topic_counts = defaultdict(int)
    
    print(f"\nExtracting point clouds from {mcap_path}...")
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic
            
            # Filter topics if specified
            if topics and topic_name not in topics:
                continue
            
            # Only process PointCloud2 topics
            if 'PointCloud2' not in schema.name:
                continue
            
            # Create directory for this topic
            if topic_name not in topic_dirs:
                topic_dir = output_dir / topic_name.replace('/', '_').strip('_')
                topic_dir.mkdir(parents=True, exist_ok=True)
                topic_dirs[topic_name] = topic_dir
            
            topic_dir = topic_dirs[topic_name]
            timestamp_ns = message.log_time
            timestamp_sec = timestamp_ns / 1e9
            
            # Extract point cloud data
            try:
                # For PointCloud2, save the raw CDR data
                # The data can be parsed later with proper ROS 2 message deserialization
                if hasattr(message, 'data'):
                    filename = f"{int(timestamp_ns)}.bin"
                    filepath = topic_dir / filename
                    with open(filepath, 'wb') as pf:
                        pf.write(message.data)
                    topic_counts[topic_name] += 1
                elif hasattr(message, 'payload'):
                    filename = f"{int(timestamp_ns)}.bin"
                    filepath = topic_dir / filename
                    with open(filepath, 'wb') as pf:
                        pf.write(message.payload)
                    topic_counts[topic_name] += 1
                else:
                    continue
                
                if topic_counts[topic_name] % 50 == 0:
                    print(f"  Extracted {topic_counts[topic_name]} point clouds from {topic_name}")
            except Exception as e:
                continue
    
    print(f"\nPoint cloud extraction complete:")
    for topic, count in topic_counts.items():
        print(f"  {topic}: {count} point clouds")
    
    return topic_counts


def extract_can_bus(mcap_path: str, output_dir: Path, topics: Optional[List[str]] = None):
    """Extract CAN bus data from MCAP file."""
    output_dir = output_dir / "can_bus"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    can_bus_data = []
    topic_counts = defaultdict(int)
    
    print(f"\nExtracting CAN bus data from {mcap_path}...")
    
    # Common CAN bus topic names
    can_topics = ['canbus', 'can_bus', 'chassis', 'vehicle', 'odom']
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic.lower()
            
            # Filter topics if specified
            if topics:
                if not any(t.lower() in topic_name for t in topics):
                    continue
            else:
                # Auto-detect CAN bus topics
                if not any(keyword in topic_name for keyword in can_topics):
                    continue
            
            timestamp_ns = message.log_time
            timestamp_sec = timestamp_ns / 1e9
            
            # Extract message data
            # For CAN bus data, we save the raw CDR data and metadata
            # Proper deserialization requires ROS 2 message definitions
            try:
                data_dict = {
                    'topic': channel.topic,
                    'timestamp_ns': timestamp_ns,
                    'timestamp_sec': timestamp_sec,
                    'message_type': schema.name,
                    'data_length': len(message.data) if hasattr(message, 'data') else 0,
                }
                
                # Save raw data reference (actual deserialization needs ROS 2)
                # For now, just save metadata
                can_bus_data.append(data_dict)
                topic_counts[channel.topic] += 1
                
                if len(can_bus_data) % 1000 == 0:
                    print(f"  Extracted {len(can_bus_data)} CAN bus messages")
            except Exception as e:
                continue
    
    # Save CAN bus data
    if can_bus_data:
        # Save as JSONL (one message per line)
        jsonl_path = output_dir / "can_bus_messages.jsonl"
        with open(jsonl_path, 'w', encoding='utf-8') as f:
            for msg in can_bus_data:
                f.write(json.dumps(msg, ensure_ascii=False) + '\n')
        
        # Also save as JSON array
        json_path = output_dir / "can_bus_messages.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(can_bus_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nCAN bus extraction complete:")
        print(f"  Total messages: {len(can_bus_data)}")
        print(f"  Saved to: {jsonl_path}")
        print(f"  Saved to: {json_path}")
        for topic, count in topic_counts.items():
            print(f"  {topic}: {count} messages")
    
    return topic_counts


def extract_all_data(mcap_path: str, output_dir: Path, 
                     extract_images_flag: bool = True,
                     extract_point_clouds_flag: bool = True,
                     extract_can_bus_flag: bool = True,
                     image_topics: Optional[List[str]] = None,
                     point_cloud_topics: Optional[List[str]] = None,
                     can_bus_topics: Optional[List[str]] = None):
    """Extract all data from MCAP file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"Extracting data from: {mcap_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    results = {}
    
    if extract_images_flag:
        results['images'] = extract_images(mcap_path, output_dir, image_topics)
    
    if extract_point_clouds_flag:
        results['point_clouds'] = extract_point_clouds(mcap_path, output_dir, point_cloud_topics)
    
    if extract_can_bus_flag:
        results['can_bus'] = extract_can_bus(mcap_path, output_dir, can_bus_topics)
    
    # Save extraction summary
    summary = {
        'mcap_file': mcap_path,
        'output_dir': str(output_dir),
        'extraction_results': {
            k: dict(v) if isinstance(v, defaultdict) else v 
            for k, v in results.items()
        }
    }
    
    summary_path = output_dir / "extraction_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    
    print(f"\nExtraction summary saved to: {summary_path}")
    print("="*80)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Extract data (images, point clouds, CAN bus) from MCAP files'
    )
    parser.add_argument(
        'mcap_file',
        type=str,
        help='Path to MCAP file (.mcap)'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        type=str,
        default=None,
        help='Output directory (default: <mcap_file>_extracted)'
    )
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='Skip image extraction'
    )
    parser.add_argument(
        '--no-point-clouds',
        action='store_true',
        help='Skip point cloud extraction'
    )
    parser.add_argument(
        '--no-can-bus',
        action='store_true',
        help='Skip CAN bus extraction'
    )
    parser.add_argument(
        '--image-topics',
        type=str,
        nargs='+',
        default=None,
        help='Specific image topics to extract (default: all image topics)'
    )
    parser.add_argument(
        '--point-cloud-topics',
        type=str,
        nargs='+',
        default=None,
        help='Specific point cloud topics to extract (default: all PointCloud2 topics)'
    )
    parser.add_argument(
        '--can-bus-topics',
        type=str,
        nargs='+',
        default=None,
        help='Specific CAN bus topics to extract (default: auto-detect)'
    )
    args = parser.parse_args()
    
    if not MCAP_AVAILABLE:
        print("Error: mcap library is required. Install: pip install mcap")
        sys.exit(1)
    
    mcap_path = Path(args.mcap_file)
    if not mcap_path.exists():
        print(f"Error: File not found: {mcap_path}")
        sys.exit(1)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = mcap_path.parent / f"{mcap_path.stem}_extracted"
    
    extract_all_data(
        str(mcap_path),
        output_dir,
        extract_images_flag=not args.no_images,
        extract_point_clouds_flag=not args.no_point_clouds,
        extract_can_bus_flag=not args.no_can_bus,
        image_topics=args.image_topics,
        point_cloud_topics=args.point_cloud_topics,
        can_bus_topics=args.can_bus_topics,
    )


if __name__ == '__main__':
    main()

