#!/usr/bin/env python3
import os
from pathlib import Path
import numpy as np

CAMS = [
    'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT',
    'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_FRONT_LEFT'
]

def parse_ts(filename):
    import re
    tokens = re.findall(r'\d+', filename)
    if tokens:
        return int(tokens[-1])
    return None

def simulate_sync(scene_dir):
    scene_path = Path(scene_dir)
    cam_data = {}
    
    # 1. Load all timestamps
    all_starts = []
    all_ends = []
    
    for cam in CAMS:
        files = sorted((scene_path / cam).glob('*.jpg'))
        ts = np.array([parse_ts(f.name) / 1e6 for f in files]) # to ms
        cam_data[cam] = ts
        all_starts.append(ts[0])
        all_ends.append(ts[-1])
    
    # 2. Find common overlap range
    overlap_start = max(all_starts)
    overlap_end = min(all_ends)
    
    # Use CAM_FRONT as reference
    ref_ts = cam_data['CAM_FRONT']
    # Filter ref frames to only those within overlap
    ref_ts = ref_ts[(ref_ts >= overlap_start) & (ref_ts <= overlap_end)]
    
    print(f"原始参考帧数 (Overlap内): {len(ref_ts)}")
    print("-" * 45)
    print(f"{'Threshold (ms)':<15} | {'Synced Frames':<15} | {'Yield (%)':<10}")
    print("-" * 45)

    thresholds = [20, 40, 60, 80, 100, 150, 200]
    
    for thr in thresholds:
        synced_count = 0
        
        # Simple simulation of nearest neighbor sync
        for t0 in ref_ts:
            ok = True
            for cam in CAMS:
                if cam == 'CAM_FRONT': continue
                
                # Find nearest timestamp in this cam
                target_cam_ts = cam_data[cam]
                idx = np.searchsorted(target_cam_ts, t0)
                
                # Check distances to neighbors
                d1 = abs(target_cam_ts[idx] - t0) if idx < len(target_cam_ts) else float('inf')
                d2 = abs(target_cam_ts[idx-1] - t0) if idx > 0 else float('inf')
                
                if min(d1, d2) > thr:
                    ok = False
                    break
            
            if ok:
                synced_count += 1
        
        yield_pct = (synced_count / len(ref_ts)) * 100
        print(f"{thr:<15} | {synced_count:<15} | {yield_pct:<10.1f}%")

if __name__ == "__main__":
    simulate_sync('inference_only/custom_data_template/scene_000')

