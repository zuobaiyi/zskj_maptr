#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Simple script to extract data from MCAP files using ros2 bag command.
This script uses ros2 bag command-line tool to extract data.
"""

import argparse
import subprocess
import json
import os
from pathlib import Path
import sys


def extract_with_ros2_bag(mcap_path: str, output_dir: Path, topics: list = None):
    """Extract data using ros2 bag export command."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Extracting data from {mcap_path} using ros2 bag...")
    print(f"Output directory: {output_dir}")
    
    # Check if ros2 is available
    try:
        result = subprocess.run(['ros2', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            print("Error: ros2 command not available")
            return False
    except Exception as e:
        print(f"Error: Cannot run ros2 command: {e}")
        return False
    
    # Get bag info first
    print("\nGetting bag information...")
    try:
        result = subprocess.run(
            ['ros2', 'bag', 'info', mcap_path],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            print(result.stdout)
        else:
            print(f"Warning: ros2 bag info failed: {result.stderr}")
    except Exception as e:
        print(f"Warning: Failed to get bag info: {e}")
    
    # Export topics
    export_dir = output_dir / "ros2_bag_export"
    export_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nExporting to: {export_dir}")
    print("Note: This may take a while for large bags...")
    
    try:
        # Use ros2 bag export (if available) or ros2 bag play with record
        cmd = ['ros2', 'bag', 'export', mcap_path, '--output', str(export_dir)]
        if topics:
            cmd.extend(['--topics'] + topics)
        
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hour timeout
        )
        
        if result.returncode == 0:
            print("Export completed successfully!")
            print(result.stdout)
            return True
        else:
            print(f"Export failed: {result.stderr}")
            return False
    except subprocess.TimeoutExpired:
        print("Export timed out (may still be processing)")
        return False
    except Exception as e:
        print(f"Error during export: {e}")
        return False


def create_extraction_script(mcap_path: str, output_dir: Path):
    """Create a Python script that can be run to extract data."""
    script_path = output_dir / "extract_data.py"
    
    script_content = f'''#!/usr/bin/env python3
# Auto-generated extraction script for {Path(mcap_path).name}

import sys
from pathlib import Path

# Add ros2 Python packages to path if needed
ros2_paths = [
    "/opt/ros/humble/lib/python3.10/site-packages",
    "/opt/ros/humble/local/lib/python3.10/dist-packages",
]
for p in ros2_paths:
    if Path(p).exists() and p not in sys.path:
        sys.path.insert(0, p)

try:
    from rclpy.serialization import deserialize_message
    from sensor_msgs.msg import CompressedImage, Image, PointCloud2
    from cv_bridge import CvBridge
    import cv2
    import numpy as np
    import json
    from mcap.reader import make_reader
    
    bridge = CvBridge()
    
    mcap_path = "{mcap_path}"
    output_dir = Path("{output_dir}")
    
    images_dir = output_dir / "images"
    point_clouds_dir = output_dir / "point_clouds"
    can_bus_dir = output_dir / "can_bus"
    
    images_dir.mkdir(parents=True, exist_ok=True)
    point_clouds_dir.mkdir(parents=True, exist_ok=True)
    can_bus_dir.mkdir(parents=True, exist_ok=True)
    
    print("Extracting data from MCAP file...")
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        for schema, channel, message in reader.iter_messages():
            topic = channel.topic
            timestamp_ns = message.log_time
            
            # Extract images
            if 'Image' in schema.name:
                try:
                    # Try to decode compressed image
                    if hasattr(message, 'data') or hasattr(message, 'payload'):
                        data = getattr(message, 'data', None) or getattr(message, 'payload', None)
                        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if img is not None:
                            topic_dir = images_dir / topic.replace('/', '_').strip('_')
                            topic_dir.mkdir(parents=True, exist_ok=True)
                            cv2.imwrite(str(topic_dir / f"{{int(timestamp_ns)}}.jpg"), img)
                except Exception as e:
                    pass
            
            # Extract point clouds
            elif 'PointCloud2' in schema.name:
                try:
                    if hasattr(message, 'data') or hasattr(message, 'payload'):
                        data = getattr(message, 'data', None) or getattr(message, 'payload', None)
                        topic_dir = point_clouds_dir / topic.replace('/', '_').strip('_')
                        topic_dir.mkdir(parents=True, exist_ok=True)
                        with open(topic_dir / f"{{int(timestamp_ns)}}.bin", 'wb') as f:
                            f.write(data)
                except Exception as e:
                    pass
            
            # Extract CAN bus data
            elif any(kw in topic.lower() for kw in ['canbus', 'chassis', 'location', 'odom']):
                try:
                    # Save as JSON
                    data_dict = {{
                        'topic': topic,
                        'timestamp_ns': timestamp_ns,
                        'timestamp_sec': timestamp_ns / 1e9,
                    }}
                    # Try to extract message fields
                    if hasattr(message, '__dict__'):
                        for k, v in message.__dict__.items():
                            if not k.startswith('_'):
                                try:
                                    if isinstance(v, (np.integer, np.int_, np.intc)):
                                        data_dict[k] = int(v)
                                    elif isinstance(v, (np.floating, np.float_)):
                                        data_dict[k] = float(v)
                                    elif isinstance(v, np.ndarray):
                                        data_dict[k] = v.tolist()
                                    else:
                                        data_dict[k] = str(v)
                                except:
                                    pass
                    
                    can_bus_file = can_bus_dir / "messages.jsonl"
                    with open(can_bus_file, 'a') as f:
                        f.write(json.dumps(data_dict) + '\\n')
                except Exception as e:
                    pass
    
    print("Extraction complete!")
    
except ImportError as e:
    print(f"Error: Missing dependencies: {{e}}")
    print("Please ensure ROS 2 and required packages are installed.")
'''
    
    with open(script_path, 'w') as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)
    print(f"\nCreated extraction script: {script_path}")
    print("You can run it with: python " + str(script_path))
    
    return script_path


def main():
    parser = argparse.ArgumentParser(
        description='Extract data from MCAP files (creates extraction script)'
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
        '--topics',
        type=str,
        nargs='+',
        default=None,
        help='Specific topics to extract (default: all)'
    )
    parser.add_argument(
        '--use-ros2-bag',
        action='store_true',
        help='Try to use ros2 bag export command'
    )
    args = parser.parse_args()
    
    mcap_path = Path(args.mcap_file)
    if not mcap_path.exists():
        print(f"Error: File not found: {mcap_path}")
        sys.exit(1)
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = mcap_path.parent / f"{mcap_path.stem}_extracted"
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print(f"MCAP Data Extraction Tool")
    print("="*80)
    print(f"Input file: {mcap_path}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Try ros2 bag export if requested
    if args.use_ros2_bag:
        success = extract_with_ros2_bag(str(mcap_path), output_dir, args.topics)
        if success:
            print("\nData extraction completed using ros2 bag!")
            return
    
    # Create extraction script
    print("\nCreating Python extraction script...")
    script_path = create_extraction_script(str(mcap_path), output_dir)
    
    print("\n" + "="*80)
    print("Next steps:")
    print("="*80)
    print(f"1. Run the extraction script:")
    print(f"   python {script_path}")
    print("\n2. Or use ros2 bag command directly:")
    print(f"   ros2 bag export {mcap_path} --output {output_dir}/ros2_bag_export")
    print("\n3. Or install mcap-ros2-support for better message decoding:")
    print("   pip install mcap-ros2-support")
    print("="*80)


if __name__ == '__main__':
    main()

