#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extract all information from MCAP (rosbag2) files.
MCAP is the new format for ROS 2 rosbag files.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Any
from collections import defaultdict
import sys

try:
    import mcap
    from mcap.reader import make_reader
    try:
        from mcap.mcap0.reader import read_ros2_messages
    except ImportError:
        pass  # Optional import
    MCAP_AVAILABLE = True
except ImportError as e:
    MCAP_AVAILABLE = False
    # Don't print warning here, will be handled in main

try:
    from rosbag2_py import SequentialReader, StorageOptions, ConverterOptions
    ROSBAG2_AVAILABLE = True
except ImportError:
    ROSBAG2_AVAILABLE = False


def extract_mcap_info_mcap_lib(mcap_path: str) -> Dict[str, Any]:
    """Extract information using mcap library."""
    info = {
        'file_path': mcap_path,
        'file_size_mb': os.path.getsize(mcap_path) / (1024 * 1024),
        'topics': {},
        'channels': {},
        'statistics': {},
        'metadata': {},
    }
    
    topic_stats = defaultdict(lambda: {
        'message_count': 0,
        'first_timestamp': None,
        'last_timestamp': None,
        'message_types': set(),
        'frequency': None,
    })
    
    with open(mcap_path, 'rb') as f:
        reader = make_reader(f)
        
        # Read summary for statistics
        summary = reader.get_summary()
        if summary and summary.statistics:
            info['statistics'] = {
                'message_count': summary.statistics.message_count,
                'channel_count': summary.statistics.channel_count,
                'attachment_count': summary.statistics.attachment_count,
                'metadata_count': summary.statistics.metadata_count,
            }
        
        # Collect channels and schemas from messages
        channels_seen = {}
        schemas_seen = {}
        
        # Read messages and collect statistics
        first_timestamp = None
        last_timestamp = None
        
        for schema, channel, message in reader.iter_messages():
            topic_name = channel.topic
            timestamp_ns = message.log_time
            
            # Store channel and schema info
            if channel.id not in channels_seen:
                channels_seen[channel.id] = channel
                info['channels'][topic_name] = {
                    'channel_id': channel.id,
                    'schema_id': channel.schema_id,
                    'schema_name': schema.name if schema else 'unknown',
                    'message_encoding': channel.message_encoding,
                    'metadata': dict(channel.metadata) if channel.metadata else {},
                }
                topic_stats[topic_name]['message_types'].add(schema.name if schema else 'unknown')
            
            topic_stats[topic_name]['message_count'] += 1
            
            if topic_stats[topic_name]['first_timestamp'] is None:
                topic_stats[topic_name]['first_timestamp'] = timestamp_ns
            topic_stats[topic_name]['last_timestamp'] = timestamp_ns
            
            if first_timestamp is None:
                first_timestamp = timestamp_ns
            last_timestamp = timestamp_ns
        
        # Calculate frequencies
        for topic_name, stats in topic_stats.items():
            if stats['first_timestamp'] and stats['last_timestamp'] and stats['message_count'] > 1:
                duration_ns = stats['last_timestamp'] - stats['first_timestamp']
                duration_sec = duration_ns / 1e9
                if duration_sec > 0:
                    stats['frequency'] = stats['message_count'] / duration_sec
            
            # Convert set to list for JSON serialization
            stats['message_types'] = list(stats['message_types'])
            
            # Convert timestamps to readable format
            if stats['first_timestamp']:
                stats['first_timestamp_sec'] = stats['first_timestamp'] / 1e9
            if stats['last_timestamp']:
                stats['last_timestamp_sec'] = stats['last_timestamp'] / 1e9
        
        info['topics'] = dict(topic_stats)
        info['time_range'] = {
            'first_timestamp_ns': first_timestamp,
            'last_timestamp_ns': last_timestamp,
            'first_timestamp_sec': first_timestamp / 1e9 if first_timestamp else None,
            'last_timestamp_sec': last_timestamp / 1e9 if last_timestamp else None,
            'duration_sec': (last_timestamp - first_timestamp) / 1e9 if (first_timestamp and last_timestamp) else None,
        }
    
    return info


def extract_mcap_info_rosbag2(mcap_path: str) -> Dict[str, Any]:
    """Extract information using rosbag2_py library."""
    info = {
        'file_path': mcap_path,
        'file_size_mb': os.path.getsize(mcap_path) / (1024 * 1024),
        'topics': {},
        'metadata': {},
    }
    
    storage_options = StorageOptions(uri=mcap_path, storage_id='mcap')
    converter_options = ConverterOptions(
        input_serialization_format='cdr',
        output_serialization_format='cdr'
    )
    
    reader = SequentialReader()
    reader.open(storage_options, converter_options)
    
    topic_types = reader.get_all_topics_and_types()
    topic_stats = defaultdict(lambda: {
        'message_count': 0,
        'first_timestamp': None,
        'last_timestamp': None,
        'message_type': None,
        'frequency': None,
    })
    
    # Get topic types
    for topic_metadata in topic_types:
        topic_name = topic_metadata.name
        topic_stats[topic_name]['message_type'] = topic_metadata.type
    
    # Read messages
    first_timestamp = None
    last_timestamp = None
    
    while reader.has_next():
        (topic, data, timestamp) = reader.read_next()
        timestamp_ns = timestamp
        
        topic_stats[topic]['message_count'] += 1
        
        if topic_stats[topic]['first_timestamp'] is None:
            topic_stats[topic]['first_timestamp'] = timestamp_ns
        topic_stats[topic]['last_timestamp'] = timestamp_ns
        
        if first_timestamp is None:
            first_timestamp = timestamp_ns
        last_timestamp = timestamp_ns
    
    # Calculate frequencies
    for topic_name, stats in topic_stats.items():
        if stats['first_timestamp'] and stats['last_timestamp'] and stats['message_count'] > 1:
            duration_ns = stats['last_timestamp'] - stats['first_timestamp']
            duration_sec = duration_ns / 1e9
            if duration_sec > 0:
                stats['frequency'] = stats['message_count'] / duration_sec
        
        # Convert timestamps to readable format
        if stats['first_timestamp']:
            stats['first_timestamp_sec'] = stats['first_timestamp'] / 1e9
        if stats['last_timestamp']:
            stats['last_timestamp_sec'] = stats['last_timestamp'] / 1e9
    
    info['topics'] = dict(topic_stats)
    info['time_range'] = {
        'first_timestamp_ns': first_timestamp,
        'last_timestamp_ns': last_timestamp,
        'first_timestamp_sec': first_timestamp / 1e9 if first_timestamp else None,
        'last_timestamp_sec': last_timestamp / 1e9 if last_timestamp else None,
        'duration_sec': (last_timestamp - first_timestamp) / 1e9 if (first_timestamp and last_timestamp) else None,
    }
    
    return info


def extract_mcap_info_basic(mcap_path: str) -> Dict[str, Any]:
    """Basic file information extraction when libraries are not available."""
    info = {
        'file_path': mcap_path,
        'file_size_mb': os.path.getsize(mcap_path) / (1024 * 1024),
        'error': 'MCAP/rosbag2 libraries not available. Please install: pip install mcap rosbag2_py',
        'topics': {},
    }
    return info


def print_summary(info: Dict[str, Any]):
    """Print a human-readable summary."""
    print("\n" + "="*80)
    print(f"MCAP File Information: {info['file_path']}")
    print("="*80)
    print(f"File Size: {info['file_size_mb']:.2f} MB")
    
    if 'time_range' in info and info['time_range'].get('duration_sec'):
        tr = info['time_range']
        print(f"\nTime Range:")
        print(f"  Start: {tr.get('first_timestamp_sec', 'N/A'):.3f} sec")
        print(f"  End:   {tr.get('last_timestamp_sec', 'N/A'):.3f} sec")
        print(f"  Duration: {tr.get('duration_sec', 0):.2f} sec ({tr.get('duration_sec', 0)/60:.2f} minutes)")
    
    if 'statistics' in info:
        stats = info['statistics']
        print(f"\nStatistics:")
        print(f"  Total Messages: {stats.get('message_count', 0)}")
        print(f"  Channels: {stats.get('channel_count', 0)}")
        print(f"  Attachments: {stats.get('attachment_count', 0)}")
        print(f"  Metadata: {stats.get('metadata_count', 0)}")
    
    if 'topics' in info and info['topics']:
        print(f"\nTopics ({len(info['topics'])}):")
        print("-" * 80)
        for topic_name, stats in sorted(info['topics'].items()):
            msg_type = stats.get('message_type') or stats.get('message_types', ['unknown'])[0] if stats.get('message_types') else 'unknown'
            count = stats.get('message_count', 0)
            freq = stats.get('frequency', 0)
            print(f"  {topic_name}")
            print(f"    Type: {msg_type}")
            print(f"    Messages: {count}")
            if freq:
                print(f"    Frequency: {freq:.2f} Hz")
            if stats.get('first_timestamp_sec'):
                print(f"    Time Range: {stats['first_timestamp_sec']:.3f} - {stats['last_timestamp_sec']:.3f} sec")
            print()
    
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Extract all information from MCAP (rosbag2) files'
    )
    parser.add_argument(
        'mcap_file',
        type=str,
        help='Path to MCAP file (.mcap)'
    )
    parser.add_argument(
        '--output',
        '-o',
        type=str,
        default=None,
        help='Output JSON file path (default: <mcap_file>_info.json)'
    )
    parser.add_argument(
        '--format',
        choices=['json', 'summary', 'both'],
        default='both',
        help='Output format: json, summary, or both (default: both)'
    )
    args = parser.parse_args()
    
    mcap_path = Path(args.mcap_file)
    if not mcap_path.exists():
        print(f"Error: File not found: {mcap_path}")
        sys.exit(1)
    
    # Try different extraction methods
    info = None
    if MCAP_AVAILABLE:
        try:
            print(f"Extracting information using mcap library...")
            info = extract_mcap_info_mcap_lib(str(mcap_path))
        except Exception as e:
            print(f"Error with mcap library: {e}")
            info = None
    
    if info is None and ROSBAG2_AVAILABLE:
        try:
            print(f"Extracting information using rosbag2_py library...")
            info = extract_mcap_info_rosbag2(str(mcap_path))
        except Exception as e:
            print(f"Error with rosbag2_py library: {e}")
            info = None
    
    if info is None:
        print("Warning: Using basic extraction (limited information)")
        info = extract_mcap_info_basic(str(mcap_path))
    
    # Print summary
    if args.format in ['summary', 'both']:
        print_summary(info)
    
    # Save JSON
    if args.format in ['json', 'both']:
        output_path = args.output
        if output_path is None:
            output_path = mcap_path.parent / f"{mcap_path.stem}_info.json"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed information saved to: {output_path}")
    
    # Installation instructions if libraries are missing
    if not MCAP_AVAILABLE and not ROSBAG2_AVAILABLE:
        print("\n" + "="*80)
        print("To get full information extraction, please install:")
        print("  pip install mcap")
        print("  or")
        print("  pip install rosbag2_py")
        print("="*80)


if __name__ == '__main__':
    main()

