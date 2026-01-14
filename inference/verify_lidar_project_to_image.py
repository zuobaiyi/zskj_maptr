#!/usr/bin/env python3
"""Project LiDAR pointcloud into camera images to visually validate extrinsics.

This is the "王者方案": if the extrinsics + intrinsics are right, projected points
should align with edges (cars, poles, road boundaries) in the image.

Default behavior validates your generated cam_ego/*.yaml by using the chain:
  T_lidar2cam = inv(T_cam2ego) @ inv(T_ego2lidar)

Optionally, you can project using the direct lidar2cam from calib_ori/**.txt
for comparison.

Expected scene layout (same as inference_only/custom_data_template):
  scene_000/
    CAM_FRONT/*.jpg  (filenames are integer timestamps, usually ns)
    ...
    cam_ego/cam_front_ego.yaml
    calib_ori/**.txt (contains Intrinsic/Distortion and lidar2cam)
    calib_ori/lidar*_to_base.yaml (ego->lidar, VALIDATED: requires inversion)
    sensor__lidar__*__PointCloud2/pointcloud2/*.bin + messages.jsonl

IMPORTANT: The lidar*_to_base.yaml files represent ego->lidar transforms
(parent=ego/base, child=lidar). This has been validated by comparing projections
with different interpretations. The default --lidar-yaml-type=ego2lidar is correct.

Outputs:
  inference_only/calib_check_outputs/<scene>_lidar_overlay/<cam>/*.jpg
  ... and a summary JSON with timing + transform deltas.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class CameraCalib:
    K: np.ndarray  # (3,3)
    dist: np.ndarray  # (N,)
    R: np.ndarray  # (3,3) lidar->cam
    t: np.ndarray  # (3,)  lidar->cam


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def _parse_bracket_rows(block: str, expected_rows: int) -> np.ndarray:
    # Parses content like: [a,b,c,d],[...],...
    rows: List[List[float]] = []
    i = 0
    while len(rows) < expected_rows:
        i = block.find("[", i)
        if i < 0:
            break
        j = block.find("]", i)
        if j < 0:
            break
        row = block[i + 1:j]
        vals = [float(x) for x in row.split(",") if x.strip()]
        rows.append(vals)
        i = j + 1
    return np.array(rows, dtype=np.float64)


def parse_calib_txt_lidar2cam(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Parse lidar->cam 4x4 extrinsic from your calib_ori/*.txt."""
    txt = _read_text(p)

    # Prefer "json format" section if present.
    marker = "************* json format *************"
    if marker in txt:
        sub = txt.split(marker, 1)[1]
    else:
        sub = txt

    # Find the first bracketed 4x4 after "Extrinsic:".
    idx = sub.find("Extrinsic:")
    if idx >= 0:
        sub2 = sub[idx:]
    else:
        sub2 = sub

    # Extract 4 rows.
    T = _parse_bracket_rows(sub2, expected_rows=4)
    if T.shape != (4, 4):
        raise ValueError(f"Failed to parse 4x4 extrinsic from: {p}")

    R = T[:3, :3]
    t = T[:3, 3]
    return R, t


def parse_calib_txt_intrinsic_dist(p: Path) -> Tuple[np.ndarray, np.ndarray]:
    txt = _read_text(p)

    # Intrinsic in bracket form (preferred)
    if "Intrinsic:" in txt and "[" in txt:
        idx = txt.find("Intrinsic:")
        sub = txt[idx:]
        K = _parse_bracket_rows(sub, expected_rows=3)
        if K.shape == (3, 3):
            pass
        else:
            K = None
    else:
        K = None

    if K is None:
        # Fallback: three numeric lines after "Intrinsic:"
        lines = txt.splitlines()
        for i, line in enumerate(lines):
            if line.strip().startswith("Intrinsic:"):
                rows = []
                for j in range(i + 1, min(i + 4, len(lines))):
                    parts = [x for x in lines[j].strip().split() if x]
                    if len(parts) != 3:
                        break
                    rows.append([float(x) for x in parts])
                if len(rows) == 3:
                    K = np.array(rows, dtype=np.float64)
                    break
        if K is None:
            raise ValueError(f"Failed to parse intrinsic K from: {p}")

    # Distortion in bracket form
    dist = None
    if "Distortion:" in txt:
        idx = txt.find("Distortion:")
        sub = txt[idx:]
        one = _parse_bracket_rows(sub, expected_rows=1)
        if one.size > 0:
            dist = one.reshape(-1)

    if dist is None:
        dist = np.zeros((0,), dtype=np.float64)

    return K, dist


def quat_xyzw_to_rot(q: Sequence[float]) -> np.ndarray:
    x, y, z, w = [float(v) for v in q]
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n == 0:
        raise ValueError("Zero-norm quaternion")
    x, y, z, w = x / n, y / n, z / n, w / n

    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z

    R = np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )
    return R


def make_T(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t.reshape(3)
    return T


def inv_T(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float64)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def rot_angle_deg(R: np.ndarray) -> float:
    # Return rotation angle from rotation matrix.
    tr = float(np.trace(R))
    v = (tr - 1.0) / 2.0
    v = max(-1.0, min(1.0, v))
    return float(math.degrees(math.acos(v)))


def read_cam_ego_matrix(p: Path) -> np.ndarray:
    # The file is JSON-in-YAML; yaml.safe_load can parse JSON too.
    txt = _read_text(p).strip()
    data = json.loads(txt)
    T = np.array(data["camera2ego"], dtype=np.float64)
    if T.shape != (4, 4):
        raise ValueError(f"camera2ego must be 4x4, got {T.shape} from {p}")
    return T


def read_cam_ego_debug_paths(p: Path) -> Tuple[Optional[str], Optional[str]]:
    txt = _read_text(p).strip()
    data = json.loads(txt)
    dbg = data.get("debug", {})
    return dbg.get("extrinsic_txt"), dbg.get("lidar_yaml")


def resolve_debug_path(path_str: str, scene_dir: Path) -> Path:
    """Resolve a path embedded in cam_ego debug info.

    Historical note: some runs wrote debug paths under a `calib/` folder, while
    the current template uses `calib_ori/`. Newer templates may also put lidar
    poses under `lidar_ego/`. This helper tries these safe rewrites.
    """

    p = Path(path_str)
    if p.exists():
        return p

    candidates: List[Path] = []

    # Common rewrite: calib -> calib_ori
    if "/calib/" in path_str:
        rel = path_str.split("/calib/", 1)[1]
        candidates.append(Path(path_str.replace("/calib/", "/calib_ori/")))
        candidates.append(scene_dir / "calib_ori" / rel)
        candidates.append(scene_dir / "calib" / rel)
        candidates.append(scene_dir / "lidar_ego" / Path(rel).name)

    if "/calib_ori/" in path_str:
        rel = path_str.split("/calib_ori/", 1)[1]
        candidates.append(scene_dir / "calib_ori" / rel)
        candidates.append(scene_dir / "lidar_ego" / Path(rel).name)

    # Allow direct replacement to lidar_ego for lidar yaml paths.
    if Path(path_str).name.endswith("_to_base.yaml"):
        candidates.append(scene_dir / "lidar_ego" / Path(path_str).name)

    for cand in candidates:
        if cand.exists():
            return cand

    # As a last resort, try interpreting it as relative to scene_dir.
    p4 = scene_dir / path_str
    if p4.exists():
        return p4

    return p


def read_ego2lidar_from_yaml(p: Path) -> np.ndarray:
    import yaml  # PyYAML is usually available in ML envs

    data = yaml.safe_load(_read_text(p))
    if not isinstance(data, dict) or "lidar_transform" not in data:
        raise ValueError(f"Unexpected lidar yaml format: {p}")
    lt = data["lidar_transform"]
    t = np.array(lt["translation"], dtype=np.float64)
    q = lt["rotation_quaternion"]
    R = quat_xyzw_to_rot(q)
    return make_T(R, t)


def infer_lidar_folder_from_yaml(lidar_yaml: Path) -> str:
    # Prefer parsing child_frame if possible
    try:
        import yaml

        data = yaml.safe_load(_read_text(lidar_yaml))
        child = data.get("lidar_transform", {}).get("child_frame", "")
        child = str(child)
    except Exception:
        child = ""

    name = ""
    if "lidar_back" in child:
        name = "back"
    elif "lidar_front" in child:
        name = "front"
    elif "lidar_left" in child:
        name = "left"
    elif "lidar_right" in child:
        name = "right"
    elif "lidar_mid" in child or "lidar_top" in child:
        name = "mid"

    if not name:
        stem = lidar_yaml.stem.lower()
        # lidarback_to_base.yaml
        for k in ["back", "front", "left", "right", "mid", "top"]:
            if k in stem:
                name = "mid" if k == "top" else k
                break

    if not name:
        raise ValueError(f"Cannot infer lidar name from yaml: {lidar_yaml}")
    return name


def list_image_timestamps(cam_dir: Path) -> List[int]:
    ts = []
    for p in cam_dir.glob("*.jpg"):
        try:
            ts.append(int(p.stem))
        except ValueError:
            continue
    ts.sort()
    return ts


def pick_evenly_spaced(seq: Sequence[int], n: int) -> List[int]:
    if n <= 0 or not seq:
        return []
    if n >= len(seq):
        return list(seq)
    idxs = np.linspace(0, len(seq) - 1, n, dtype=int)
    return [seq[i] for i in idxs]


def read_pointcloud_index(pointcloud_dir: Path) -> List[int]:
    # Prefer messages.jsonl timestamps; fallback to filenames.
    msg = pointcloud_dir / "messages.jsonl"
    ts: List[int] = []
    if msg.exists():
        with msg.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    j = json.loads(line)
                    ts.append(int(j["log_time_ns"]))
                except Exception:
                    continue
    else:
        for p in pointcloud_dir.glob("*.bin"):
            try:
                ts.append(int(p.stem))
            except ValueError:
                continue
    ts.sort()
    return ts


def nearest_ts(sorted_ts: Sequence[int], target: int) -> Optional[int]:
    if not sorted_ts:
        return None
    i = bisect.bisect_left(sorted_ts, target)
    candidates = []
    if 0 <= i < len(sorted_ts):
        candidates.append(sorted_ts[i])
    if i - 1 >= 0:
        candidates.append(sorted_ts[i - 1])
    if not candidates:
        return None
    return min(candidates, key=lambda x: abs(x - target))


def load_pointcloud_xyz(bin_path: Path, point_step: int = 16) -> np.ndarray:
    # Assumes float32 x,y,z,(intensity) packed with point_step bytes.
    raw = np.fromfile(str(bin_path), dtype=np.uint8)
    if raw.size % point_step != 0:
        # Best effort: truncate
        raw = raw[: raw.size - (raw.size % point_step)]
    pts = raw.view(np.float32)
    # For typical PointCloud2 layout: 4 float32 = 16 bytes
    if pts.size % 4 != 0:
        pts = pts[: pts.size - (pts.size % 4)]
    pts = pts.reshape(-1, 4)
    return pts[:, :3].astype(np.float64)


def project_and_draw(
    *,
    image_path: Path,
    pts_lidar_xyz: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    max_points: int,
    max_depth_m: float,
    point_radius: int,
) -> Tuple[np.ndarray, Dict[str, float]]:
    import cv2

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Failed to read image: {image_path}")
    h, w = img.shape[:2]

    # Use original distortion coefficients for projection
    # (distortion parameters are correctly applied to match the actual image distortion)

    if pts_lidar_xyz.shape[0] == 0:
        return img, {"n_points": 0}

    # Subsample for speed (but keep more points for density)
    if pts_lidar_xyz.shape[0] > max_points:
        stride = int(math.ceil(pts_lidar_xyz.shape[0] / max_points))
        pts = pts_lidar_xyz[::stride]
    else:
        pts = pts_lidar_xyz

    # Store original height (z in lidar frame) for coloring
    pts_height = pts[:, 2].copy()

    # Compute camera-frame depths for filtering
    pts_cam = (R @ pts.T).T + t.reshape(1, 3)
    z = pts_cam[:, 2]
    valid = z > 0.1
    pts = pts[valid]
    pts_cam = pts_cam[valid]
    z = z[valid]
    pts_height = pts_height[valid]

    if pts.shape[0] == 0:
        return img, {"n_points": 0}

    rvec, _ = cv2.Rodrigues(R.astype(np.float64))
    tvec = t.reshape(3, 1).astype(np.float64)

    # cv2 wants shape (N,1,3)
    obj = pts.reshape(-1, 1, 3).astype(np.float64)
    uv, _ = cv2.projectPoints(
        obj,
        rvec,
        tvec,
        K.astype(np.float64),
        dist.astype(np.float64),
    )
    uv = uv.reshape(-1, 2)

    u = uv[:, 0]
    v = uv[:, 1]
    in_img = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    uv = uv[in_img]
    z = z[in_img]
    pts_height = pts_height[in_img]

    n_draw = int(uv.shape[0])
    if n_draw == 0:
        return img, {"n_points": 0}

    # Height -> color (jet colormap)
    # Map height range: typically [-2, 2] meters for road scenes
    height_min, height_max = -2.0, 2.0
    height_norm = np.clip((pts_height - height_min) / (height_max - height_min), 0.0, 1.0)
    height_u8 = (height_norm * 255.0).astype(np.uint8).reshape(-1, 1)
    colors = cv2.applyColorMap(height_u8, cv2.COLORMAP_JET).reshape(
        -1, 3
    )  # BGR: blue=low, red=high

    out = img.copy()
    for (uu, vv), bgr in zip(uv, colors):
        cv2.circle(
            out,
            (int(uu), int(vv)),
            point_radius,
            (int(bgr[0]), int(bgr[1]), int(bgr[2])),
            -1,
        )

    stats = {
        "n_points_loaded": float(pts_lidar_xyz.shape[0]),
        "n_points_depth_valid": float(valid.sum()),
        "n_points_in_image": float(n_draw),
        "img_w": float(w),
        "img_h": float(h),
    }
    return out, stats


def camera_to_cam_ego_filename(cam: str) -> str:
    m = {
        "CAM_FRONT": "cam_front_ego.yaml",
        "CAM_FRONT_LEFT": "cam_left_front_ego.yaml",
        "CAM_FRONT_RIGHT": "cam_right_front_ego.yaml",
        "CAM_BACK": "cam_back_ego.yaml",
        "CAM_BACK_LEFT": "cam_left_back_ego.yaml",
        "CAM_BACK_RIGHT": "cam_right_back_ego.yaml",
    }
    if cam not in m:
        raise ValueError(f"Unknown camera: {cam}")
    return m[cam]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scene-dir",
        required=True,
        help="Path like inference_only/custom_data_template/scene_000",
    )
    ap.add_argument(
        "--cameras",
        nargs="*",
        default=[
            "CAM_FRONT",
            "CAM_FRONT_RIGHT",
            "CAM_BACK_RIGHT",
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_FRONT_LEFT",
        ],
        help="Which camera folders to process",
    )
    ap.add_argument(
        "--num-frames",
        type=int,
        default=5,
        help="How many frames per camera",
    )
    ap.add_argument(
        "--timestamps",
        nargs="*",
        type=int,
        default=None,
        help=(
            "Explicit image timestamps (filestem integers) to process; "
            "if set, overrides --num-frames sampling"
        ),
    )
    ap.add_argument(
        "--lidar-yaml-type",
        choices=["ego2lidar", "lidar2ego"],
        default="ego2lidar",
        help=(
            "How to interpret the lidar YAML: default ego2lidar assumes ego->lidar "
            "(parent=ego/base, child=lidar) and inverts to get lidar->ego. "
            "VALIDATED: ego2lidar is the correct interpretation for this dataset. "
            "Use lidar2ego only for testing/debugging with different calibration sources."
        ),
    )
    ap.add_argument(
        "--max-delta-ms",
        type=float,
        default=30.0,
        help="Max allowed |t_img - t_lidar| for pairing",
    )
    ap.add_argument(
        "--use-direct-lidar2cam",
        action="store_true",
        help=(
            "Project using the direct lidar2cam from calib_ori/*.txt "
            "instead of cam_ego chain"
        ),
    )
    ap.add_argument(
        "--max-points",
        type=int,
        default=50000,
        help="Max points to draw per image (increased for denser visualization)",
    )
    ap.add_argument(
        "--max-depth",
        type=float,
        default=60.0,
        help="Depth clamp for filtering (meters)",
    )
    ap.add_argument(
        "--point-radius",
        type=int,
        default=2,
        help="Circle radius for drawing points (default 2 for better visibility)",
    )
    ap.add_argument(
        "--out-dir",
        default=None,
        help=(
            "Output directory (default: "
            "inference_only/calib_check_outputs/<scene>_lidar_overlay)"
        ),
    )

    args = ap.parse_args()

    scene_dir = Path(args.scene_dir).resolve()
    if not scene_dir.exists():
        raise SystemExit(f"scene-dir not found: {scene_dir}")

    scene_name = scene_dir.name
    default_out = (
        scene_dir.parents[1]
        / "calib_check_outputs"
        / f"{scene_name}_lidar_overlay"
    )
    out_root = Path(args.out_dir).resolve() if args.out_dir else default_out
    out_root.mkdir(parents=True, exist_ok=True)

    summary: Dict[str, object] = {
        "scene_dir": str(scene_dir),
        "scene_name": scene_name,
        "max_delta_ms": float(args.max_delta_ms),
        "use_direct_lidar2cam": bool(args.use_direct_lidar2cam),
        "cameras": {},
    }

    for cam in args.cameras:
        cam_dir = scene_dir / cam
        if not cam_dir.exists():
            print(f"[WARN] Skip {cam}: camera dir missing: {cam_dir}")
            continue

        cam_ego_path = scene_dir / "cam_ego" / camera_to_cam_ego_filename(cam)
        if not cam_ego_path.exists():
            print(f"[WARN] Skip {cam}: cam_ego file missing: {cam_ego_path}")
            continue

        extrinsic_txt_str, lidar_yaml_str = read_cam_ego_debug_paths(
            cam_ego_path
        )
        if not extrinsic_txt_str or not lidar_yaml_str:
            msg = (
                f"[WARN] Skip {cam}: cam_ego debug paths missing in: "
                f"{cam_ego_path}"
            )
            print(msg)
            continue

        extrinsic_txt = resolve_debug_path(extrinsic_txt_str, scene_dir)
        lidar_yaml = resolve_debug_path(lidar_yaml_str, scene_dir)
        if not extrinsic_txt.exists():
            print(f"[WARN] Skip {cam}: extrinsic txt missing: {extrinsic_txt}")
            continue
        if not lidar_yaml.exists():
            print(f"[WARN] Skip {cam}: lidar yaml missing: {lidar_yaml}")
            continue

        K, dist = parse_calib_txt_intrinsic_dist(extrinsic_txt)
        R_direct, t_direct = parse_calib_txt_lidar2cam(extrinsic_txt)

        T_cam2ego = read_cam_ego_matrix(cam_ego_path)
        T_ego2lidar = read_ego2lidar_from_yaml(lidar_yaml)
        if args.lidar_yaml_type == "ego2lidar":
            T_lidar2ego = inv_T(T_ego2lidar)
        else:
            T_lidar2ego = T_ego2lidar

        T_lidar2cam_chain = inv_T(T_cam2ego) @ T_lidar2ego
        R_chain = T_lidar2cam_chain[:3, :3]
        t_chain = T_lidar2cam_chain[:3, 3]

        # Compare chain vs direct
        R_delta = R_chain @ R_direct.T
        angle_deg = rot_angle_deg(R_delta)
        t_err = float(np.linalg.norm(t_chain - t_direct))

        if args.use_direct_lidar2cam:
            R_use, t_use = R_direct, t_direct
            source = "direct_lidar2cam_txt"
        else:
            R_use, t_use = R_chain, t_chain
            source = f"cam_ego_chain_{args.lidar_yaml_type}"

        lidar_name = infer_lidar_folder_from_yaml(lidar_yaml)
        pc_dir = (
            scene_dir
            / f"sensor__lidar__{lidar_name}__PointCloud2"
            / "pointcloud2"
        )
        if not pc_dir.exists():
            print(f"[WARN] Skip {cam}: pointcloud dir missing: {pc_dir}")
            continue

        pc_ts = read_pointcloud_index(pc_dir)
        img_ts_all = list_image_timestamps(cam_dir)
        if args.timestamps:
            img_ts_pick = [int(x) for x in args.timestamps]
        else:
            img_ts_pick = pick_evenly_spaced(img_ts_all, args.num_frames)

        cam_out_dir = out_root / cam
        cam_out_dir.mkdir(parents=True, exist_ok=True)

        cam_report = {
            "cam": cam,
            "cam_dir": str(cam_dir),
            "cam_ego": str(cam_ego_path),
            "extrinsic_txt": str(extrinsic_txt),
            "lidar_yaml": str(lidar_yaml),
            "lidar_name": lidar_name,
            "pointcloud_dir": str(pc_dir),
            "projection_source": source,
            "delta_chain_vs_direct": {
                "rot_angle_deg": float(angle_deg),
                "t_l2_norm_m": float(t_err),
            },
            "pairs": [],
        }

        for ts_img in img_ts_pick:
            img_path = cam_dir / f"{ts_img}.jpg"
            ts_pc = nearest_ts(pc_ts, ts_img)
            if ts_pc is None:
                continue
            dt_ms = abs(ts_pc - ts_img) / 1e6
            if dt_ms > float(args.max_delta_ms):
                continue

            bin_path = pc_dir / f"{ts_pc}.bin"
            if not bin_path.exists():
                # Fallback: some datasets may store by other naming; skip.
                continue

            pts = load_pointcloud_xyz(bin_path)
            try:
                overlay, stats = project_and_draw(
                    image_path=img_path,
                    pts_lidar_xyz=pts,
                    R=R_use,
                    t=t_use,
                    K=K,
                    dist=dist,
                    max_points=int(args.max_points),
                    max_depth_m=float(args.max_depth),
                    point_radius=int(args.point_radius),
                )
            except ModuleNotFoundError as e:
                raise SystemExit(
                    "OpenCV (cv2) not found in this environment. "
                    "Please install opencv-python (or opencv-python-headless)."
                ) from e

            out_name = (
                f"img{ts_img}_pc{ts_pc}_dt{dt_ms:.1f}ms_"
                f"src{source}.jpg"
            )
            out_path = cam_out_dir / out_name

            import cv2

            ok = cv2.imwrite(str(out_path), overlay)
            if not ok:
                print(f"[WARN] Failed to write: {out_path}")
                continue

            cam_report["pairs"].append(
                {
                    "ts_img": int(ts_img),
                    "ts_pc": int(ts_pc),
                    "dt_ms": float(dt_ms),
                    "img": str(img_path),
                    "pc_bin": str(bin_path),
                    "out": str(out_path),
                    "stats": stats,
                }
            )

        # Write per-camera report
        (cam_out_dir / "report.json").write_text(
            json.dumps(cam_report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        summary["cameras"][cam] = cam_report

        n_pairs = len(cam_report["pairs"])
        msg = (
            f"[OK] {cam}: wrote {n_pairs} overlays -> {cam_out_dir} "
            f"| chain-vs-direct: {angle_deg:.3f} deg, {t_err:.3f} m"
        )
        print(msg)

    (out_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nDone. Summary: {out_root / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
