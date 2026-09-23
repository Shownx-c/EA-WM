#!/usr/bin/env python3
"""Build Kinematic Visual Action Fields (KVAFs) from RoboTwin episodes."""

from __future__ import annotations

import argparse
import math
import traceback
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import h5py
import numpy as np
import yaml


def parse_vec(text: Optional[str], dims: int = 3) -> np.ndarray:
    if text is None:
        return np.zeros(dims, dtype=np.float64)
    vals = [float(x) for x in text.split()]
    if len(vals) != dims:
        raise ValueError(f"Expected {dims} values, got {len(vals)} from: {text}")
    return np.asarray(vals, dtype=np.float64)


def rpy_to_matrix(rpy: Sequence[float]) -> np.ndarray:
    r, p, y = rpy
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)

    rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=np.float64)
    ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=np.float64)
    rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=np.float64)
    return rz @ ry @ rx


def quat_wxyz_to_matrix(quat_wxyz: Sequence[float]) -> np.ndarray:
    w, x, y, z = [float(v) for v in quat_wxyz]
    n = math.sqrt(w * w + x * x + y * y + z * z)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def pose7_to_matrix_wxyz(pose7: Sequence[float]) -> np.ndarray:
    pose7 = np.asarray(pose7, dtype=np.float64)
    if pose7.shape[0] != 7:
        raise ValueError("Pose must contain 7 values: x y z qw qx qy qz")
    t = pose7[:3]
    q = pose7[3:]
    out = np.eye(4, dtype=np.float64)
    out[:3, :3] = quat_wxyz_to_matrix(q)
    out[:3, 3] = t
    return out


def make_transform(rotation: Optional[np.ndarray] = None, translation: Optional[Sequence[float]] = None) -> np.ndarray:
    out = np.eye(4, dtype=np.float64)
    if rotation is not None:
        out[:3, :3] = rotation
    if translation is not None:
        out[:3, 3] = np.asarray(translation, dtype=np.float64)
    return out


def axis_angle_transform(axis: Sequence[float], angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    x, y, z = axis
    c, s = math.cos(angle), math.sin(angle)
    C = 1.0 - c
    rot = np.array(
        [
            [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
            [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
            [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
        ],
        dtype=np.float64,
    )
    return make_transform(rotation=rot)


def axis_translation_transform(axis: Sequence[float], distance: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(4, dtype=np.float64)
    axis = axis / norm
    return make_transform(translation=axis * distance)


@dataclass
class JointSpec:
    name: str
    joint_type: str
    parent_link: str
    child_link: str
    origin_xyz: np.ndarray
    origin_rpy: np.ndarray
    axis: np.ndarray


@dataclass
class ArmSpec:
    side: str
    base_joint: str
    revolute_joints: List[str]
    finger_joints: List[str]
    link_chain: List[str]
    gripper_joint_max: float


class URDFModel:
    def __init__(self, urdf_path: Path):
        self.urdf_path = Path(urdf_path)
        tree = ET.parse(str(self.urdf_path))
        root = tree.getroot()
        self.joints: Dict[str, JointSpec] = {}
        for node in root.findall("joint"):
            origin = node.find("origin")
            axis = node.find("axis")
            spec = JointSpec(
                name=node.attrib["name"],
                joint_type=node.attrib["type"],
                parent_link=node.find("parent").attrib["link"],
                child_link=node.find("child").attrib["link"],
                origin_xyz=parse_vec(origin.attrib.get("xyz", "0 0 0")) if origin is not None else np.zeros(3),
                origin_rpy=parse_vec(origin.attrib.get("rpy", "0 0 0")) if origin is not None else np.zeros(3),
                axis=parse_vec(axis.attrib.get("xyz", "1 0 0")) if axis is not None else np.array([1.0, 0.0, 0.0]),
            )
            self.joints[spec.name] = spec

    def joint_origin_transform(self, joint_name: str) -> np.ndarray:
        joint = self.joints[joint_name]
        return make_transform(rotation=rpy_to_matrix(joint.origin_rpy), translation=joint.origin_xyz)

    def apply_joint_motion(self, joint_name: str, value: float) -> np.ndarray:
        joint = self.joints[joint_name]
        if joint.joint_type in ("revolute", "continuous"):
            return axis_angle_transform(joint.axis, value)
        if joint.joint_type == "prismatic":
            return axis_translation_transform(joint.axis, value)
        return np.eye(4, dtype=np.float64)


class SkeletonVisualizer:
    def __init__(
        self,
        urdf_model: URDFModel,
        config: Dict,
        left_cfg: Dict,
        right_cfg: Dict,
        use_white_heatmap: bool = False,
        skeleton_thickness: int = 2,
        joint_radius: int = 3,
    ) -> None:
        self.urdf = urdf_model
        self.config = config
        self.left_cfg = left_cfg
        self.right_cfg = right_cfg
        self.use_white_heatmap = use_white_heatmap
        self.skeleton_thickness = skeleton_thickness
        self.joint_radius = joint_radius

        robot_pose = np.asarray(self.config["robot_pose"][0], dtype=np.float64)
        self.T_world_footprint = pose7_to_matrix_wxyz(robot_pose)

        self.left_arm = self._build_arm_spec("fl", left_cfg)
        self.right_arm = self._build_arm_spec("fr", right_cfg)

        self.depth_palette = self._build_bright_depth_palette()

    @staticmethod
    def _build_bright_depth_palette() -> np.ndarray:
        anchor_pos = np.array([0.0, 0.33, 0.66, 1.0], dtype=np.float64)
        anchor_bgr = np.array([
            [255, 220, 80],
            [255, 120, 0],
            [80, 220, 255],
            [235, 255, 255],
        ], dtype=np.float64)
        x = np.linspace(0.0, 1.0, 256)
        palette = np.empty((256, 3), dtype=np.uint8)
        for ch in range(3):
            palette[:, ch] = np.clip(np.interp(x, anchor_pos, anchor_bgr[:, ch]), 0, 255).astype(np.uint8)
        return palette

    def _build_arm_spec(self, side: str, side_cfg: Dict) -> ArmSpec:
        base_joint = f"{side}_base_joint"
        revolute_joints = [f"{side}_joint{i}" for i in range(1, 7)]
        finger_joints = [f"{side}_joint7", f"{side}_joint8"]
        link_chain = [
            f"{side}_base_link",
            f"{side}_link1",
            f"{side}_link2",
            f"{side}_link3",
            f"{side}_link4",
            f"{side}_link5",
            f"{side}_link6",
            f"{side}_link7",
            f"{side}_link8",
        ]

        gripper_joint_max = 0.04
        try:
            lock_joints = side_cfg["robot_cfg"]["kinematics"].get("lock_joints", {})
            vals = [float(v) for k, v in lock_joints.items() if k.startswith(f"{side}_joint")]
            if vals:
                gripper_joint_max = float(np.median(vals))
        except Exception:
            pass

        return ArmSpec(
            side=side,
            base_joint=base_joint,
            revolute_joints=revolute_joints,
            finger_joints=finger_joints,
            link_chain=link_chain,
            gripper_joint_max=gripper_joint_max,
        )

    def normalized_gripper_to_prismatic(self, normalized_value: float, arm: ArmSpec) -> float:
        v = float(np.clip(normalized_value, 0.0, 1.0))
        return v * arm.gripper_joint_max

    def compute_arm_keypoints(
        self,
        arm: ArmSpec,
        joint_values_6: Sequence[float],
        gripper_normalized: float,
    ) -> Dict[str, np.ndarray]:
        joint_values_6 = np.asarray(joint_values_6, dtype=np.float64)
        if joint_values_6.shape != (6,):
            raise ValueError(f"Expected 6 arm joint values, got {joint_values_6.shape}")

        points: Dict[str, np.ndarray] = {}
        transforms: Dict[str, np.ndarray] = {}

        T_cur = self.T_world_footprint @ self.urdf.joint_origin_transform(arm.base_joint)
        T_cur = T_cur @ self.urdf.apply_joint_motion(arm.base_joint, 0.0)
        points[f"{arm.side}_base_link"] = T_cur[:3, 3].copy()
        transforms[f"{arm.side}_base_link"] = T_cur.copy()

        for idx, joint_name in enumerate(arm.revolute_joints, start=1):
            T_cur = T_cur @ self.urdf.joint_origin_transform(joint_name)
            T_cur = T_cur @ self.urdf.apply_joint_motion(joint_name, float(joint_values_6[idx - 1]))
            link_name = f"{arm.side}_link{idx}"
            points[link_name] = T_cur[:3, 3].copy()
            transforms[link_name] = T_cur.copy()

        finger_distance = self.normalized_gripper_to_prismatic(gripper_normalized, arm)
        T_link6 = transforms[f"{arm.side}_link6"]
        for finger_joint, finger_link in zip(arm.finger_joints, [f"{arm.side}_link7", f"{arm.side}_link8"]):
            T_finger = T_link6 @ self.urdf.joint_origin_transform(finger_joint)
            T_finger = T_finger @ self.urdf.apply_joint_motion(finger_joint, finger_distance)
            points[finger_link] = T_finger[:3, 3].copy()
            transforms[finger_link] = T_finger.copy()

        return points

    @staticmethod
    def point_camera(point_3d: np.ndarray, extrinsic_3x4: np.ndarray) -> np.ndarray:
        point_h = np.array([point_3d[0], point_3d[1], point_3d[2], 1.0], dtype=np.float64)
        return extrinsic_3x4 @ point_h

    def project_point(self, point_3d: np.ndarray, intrinsic: np.ndarray, extrinsic_3x4: np.ndarray) -> Optional[Tuple[float, float]]:
        point_cam = self.point_camera(point_3d, extrinsic_3x4)
        if point_cam[2] <= 1e-6:
            return None
        image_h = intrinsic @ point_cam[:3]
        u = image_h[0] / image_h[2]
        v = image_h[1] / image_h[2]
        return float(u), float(v)

    @staticmethod
    def in_bounds(uv: Optional[Tuple[float, float]], width: int, height: int) -> bool:
        if uv is None:
            return False
        u, v = uv
        return 0 <= u < width and 0 <= v < height

    @staticmethod
    def clip_segment_to_canvas(
        p0: Optional[Tuple[float, float]],
        p1: Optional[Tuple[float, float]],
        width: int,
        height: int,
    ) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
        if p0 is None or p1 is None:
            return None
        p0_i = (int(round(p0[0])), int(round(p0[1])))
        p1_i = (int(round(p1[0])), int(round(p1[1])))
        ok, q0, q1 = cv2.clipLine((0, 0, width, height), p0_i, p1_i)
        if not ok:
            return None
        return (int(q0[0]), int(q0[1])), (int(q1[0]), int(q1[1]))

    def color_for_depth(self, depth: float, depth_min: float, depth_max: float) -> Tuple[int, int, int]:
        if not np.isfinite(depth):
            return (255, 255, 255)
        if not np.isfinite(depth_min) or not np.isfinite(depth_max) or abs(depth_max - depth_min) < 1e-9:
            idx = 220
        else:
            norm = (depth - depth_min) / (depth_max - depth_min)
            norm = float(np.clip(norm, 0.0, 1.0))
            idx = int(round((1.0 - norm) * 255.0))
        bgr = self.depth_palette[idx]
        return int(bgr[0]), int(bgr[1]), int(bgr[2])

    def draw_gradient_segment(
        self,
        canvas: np.ndarray,
        p0: Optional[Tuple[float, float]],
        p1: Optional[Tuple[float, float]],
        depth0: Optional[float],
        depth1: Optional[float],
        depth_min: float,
        depth_max: float,
        thickness: Optional[int] = None,
    ) -> None:
        if p0 is None or p1 is None or depth0 is None or depth1 is None:
            return
        if depth0 <= 1e-6 or depth1 <= 1e-6:
            return
        thickness = self.skeleton_thickness if thickness is None else thickness

        length = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
        steps = max(1, int(math.ceil(length / 4.0)))
        prev = np.array(p0, dtype=np.float64)
        for i in range(steps):
            t0 = i / steps
            t1 = (i + 1) / steps
            cur = (1.0 - t1) * np.array(p0, dtype=np.float64) + t1 * np.array(p1, dtype=np.float64)
            dmid = (1.0 - 0.5 * (t0 + t1)) * depth0 + 0.5 * (t0 + t1) * depth1
            color = self.color_for_depth(float(dmid), depth_min, depth_max)
            cv2.line(
                canvas,
                (int(round(prev[0])), int(round(prev[1]))),
                (int(round(cur[0])), int(round(cur[1]))),
                color,
                thickness,
                lineType=cv2.LINE_AA,
            )
            prev = cur

    def draw_arm(
        self,
        canvas: np.ndarray,
        arm: ArmSpec,
        points_3d: Dict[str, np.ndarray],
        intrinsic: np.ndarray,
        extrinsic_3x4: np.ndarray,
        depth_min: float,
        depth_max: float,
    ) -> None:
        h, w = canvas.shape[:2]
        projected: Dict[str, Optional[Tuple[float, float]]] = {}
        depths: Dict[str, Optional[float]] = {}
        for link_name, point in points_3d.items():
            point_cam = self.point_camera(point, extrinsic_3x4)
            depth = float(point_cam[2])
            depths[link_name] = depth if depth > 1e-6 else None
            projected[link_name] = self.project_point(point, intrinsic, extrinsic_3x4)

        main_chain = [
            f"{arm.side}_base_link",
            f"{arm.side}_link1",
            f"{arm.side}_link2",
            f"{arm.side}_link3",
            f"{arm.side}_link4",
            f"{arm.side}_link5",
            f"{arm.side}_link6",
        ]
        finger_links = [f"{arm.side}_link7", f"{arm.side}_link8"]
        finger_branches = [
            (f"{arm.side}_link6", f"{arm.side}_link7"),
            (f"{arm.side}_link6", f"{arm.side}_link8"),
        ]

        main_segments = []
        for a, b in zip(main_chain[:-1], main_chain[1:]):
            d0, d1 = depths.get(a), depths.get(b)
            if d0 is None or d1 is None:
                continue
            main_segments.append((0.5 * (d0 + d1), a, b))
        for _, a, b in sorted(main_segments, key=lambda x: x[0], reverse=True):
            self.draw_gradient_segment(
                canvas,
                projected.get(a),
                projected.get(b),
                depths.get(a),
                depths.get(b),
                depth_min,
                depth_max,
            )

        white = (255, 255, 255)
        finger_segments = []
        for a, b in finger_branches:
            d0, d1 = depths.get(a), depths.get(b)
            if d0 is None or d1 is None:
                continue
            finger_segments.append((0.5 * (d0 + d1), a, b))
        for _, a, b in sorted(finger_segments, key=lambda x: x[0], reverse=True):
            clipped = self.clip_segment_to_canvas(projected.get(a), projected.get(b), w, h)
            if clipped is not None:
                cv2.line(canvas, clipped[0], clipped[1], white, self.skeleton_thickness, lineType=cv2.LINE_AA)

        main_joints = []
        for link_name in main_chain:
            uv, depth = projected.get(link_name), depths.get(link_name)
            if depth is None or not self.in_bounds(uv, w, h):
                continue
            main_joints.append((depth, link_name))
        for _, link_name in sorted(main_joints, key=lambda x: x[0], reverse=True):
            uv = projected[link_name]
            color = self.color_for_depth(float(depths[link_name]), depth_min, depth_max)
            cv2.circle(
                canvas,
                (int(round(uv[0])), int(round(uv[1]))),
                self.joint_radius,
                color,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

        finger_joints = []
        for link_name in finger_links:
            uv, depth = projected.get(link_name), depths.get(link_name)
            if depth is None or not self.in_bounds(uv, w, h):
                continue
            finger_joints.append((depth, link_name))
        for _, link_name in sorted(finger_joints, key=lambda x: x[0], reverse=True):
            uv = projected[link_name]
            cv2.circle(
                canvas,
                (int(round(uv[0])), int(round(uv[1]))),
                self.joint_radius,
                white,
                thickness=-1,
                lineType=cv2.LINE_AA,
            )

    def draw_heatmap(
        self,
        canvas: np.ndarray,
        center_uv: Optional[Tuple[float, float]],
        sigma_px: float = 16.0,
        radius_px: Optional[int] = None,
    ) -> None:
        if center_uv is None:
            return

        h, w = canvas.shape[:2]
        cx, cy = float(center_uv[0]), float(center_uv[1])
        if radius_px is None:
            radius_px = int(math.ceil(3.0 * sigma_px))
        if cx + radius_px < 0 or cy + radius_px < 0 or cx - radius_px >= w or cy - radius_px >= h:
            return

        x0 = max(0, int(math.floor(cx - radius_px)))
        x1 = min(w, int(math.ceil(cx + radius_px + 1)))
        y0 = max(0, int(math.floor(cy - radius_px)))
        y1 = min(h, int(math.ceil(cy + radius_px + 1)))

        xs = np.arange(x0, x1, dtype=np.float64)
        ys = np.arange(y0, y1, dtype=np.float64)
        xx, yy = np.meshgrid(xs, ys)
        dist2 = (xx - cx) ** 2 + (yy - cy) ** 2
        intensity = np.exp(-0.5 * dist2 / (sigma_px ** 2))
        intensity *= (dist2 <= radius_px * radius_px)
        intensity = intensity[..., None]

        roi = canvas[y0:y1, x0:x1].astype(np.float32)
        if self.use_white_heatmap:
            overlay = 255.0 * intensity
            overlay = np.repeat(overlay, 3, axis=2)
        else:
            warm_bgr = np.array([170.0, 220.0, 255.0], dtype=np.float32)
            white_core = np.array([255.0, 255.0, 255.0], dtype=np.float32)
            overlay = warm_bgr * intensity + white_core * (intensity ** 2.2) * 0.55

        roi = np.clip(roi + overlay.astype(np.float32), 0.0, 255.0)
        canvas[y0:y1, x0:x1] = roi.astype(np.uint8)

    def draw_pose_axes(
        self,
        canvas: np.ndarray,
        endpose_xyz_qwxyz: Sequence[float],
        intrinsic: np.ndarray,
        extrinsic_3x4: np.ndarray,
        axis_length_m: float = 0.08,
    ) -> None:
        endpose = np.asarray(endpose_xyz_qwxyz, dtype=np.float64)
        if endpose.shape != (7,):
            raise ValueError(f"Expected 7D endpose, got shape={endpose.shape}")
        origin = endpose[:3]
        rot = quat_wxyz_to_matrix(endpose[3:])

        axis_colors = [
            (0, 0, 255),
            (0, 255, 0),
            (255, 0, 0),
        ]
        unit_axes = np.eye(3, dtype=np.float64)

        origin_uv = self.project_point(origin, intrinsic, extrinsic_3x4)
        if origin_uv is None:
            return

        h, w = canvas.shape[:2]

        for axis_vec_local, color in zip(unit_axes, axis_colors):
            endpoint = origin + rot @ (axis_vec_local * axis_length_m)
            endpoint_uv = self.project_point(endpoint, intrinsic, extrinsic_3x4)
            clipped = self.clip_segment_to_canvas(origin_uv, endpoint_uv, w, h)
            if clipped is not None:
                cv2.line(canvas, clipped[0], clipped[1], color, 2, lineType=cv2.LINE_AA)

        if self.in_bounds(origin_uv, w, h):
            cv2.circle(canvas, (int(round(origin_uv[0])), int(round(origin_uv[1]))), 3, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)


def load_yaml(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_image_size(h5f: h5py.File, camera_name: str, intrinsic: np.ndarray) -> Tuple[int, int]:
    rgb_key = f"observation/{camera_name}/rgb"
    if rgb_key in h5f:
        try:
            encoded = h5f[rgb_key][0]
            arr = np.frombuffer(encoded, dtype=np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return int(img.shape[1]), int(img.shape[0])
        except Exception:
            pass

    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]
    width = int(round(cx * 2))
    height = int(round(cy * 2))
    return width, height


def build_extrinsic_3x4(h5f: h5py.File, camera_name: str) -> np.ndarray:
    extrinsic_key = f"observation/{camera_name}/extrinsic_cv"
    cam2world_key = f"observation/{camera_name}/cam2world_gl"

    if extrinsic_key in h5f:
        return h5f[extrinsic_key][:].astype(np.float64)

    if cam2world_key in h5f:
        cam2world = h5f[cam2world_key][:].astype(np.float64)
        out = np.zeros((cam2world.shape[0], 3, 4), dtype=np.float64)
        for i in range(cam2world.shape[0]):
            world2cam = np.linalg.inv(cam2world[i])
            out[i] = world2cam[:3, :]
        return out

    raise KeyError(f"Missing both extrinsic_cv and cam2world_gl for camera: {camera_name}")


def collect_visible_depths(
    visualizer: SkeletonVisualizer,
    extrinsic: np.ndarray,
    left_arm_values: np.ndarray,
    right_arm_values: np.ndarray,
    left_gripper: np.ndarray,
    right_gripper: np.ndarray,
    hide_left_arm: bool,
    hide_right_arm: bool,
    num_frames: int,
) -> Tuple[float, float]:
    depth_values: List[float] = []
    for frame_idx in range(num_frames):
        ext = extrinsic[frame_idx]
        if not hide_left_arm:
            left_points = visualizer.compute_arm_keypoints(
                visualizer.left_arm,
                joint_values_6=left_arm_values[frame_idx],
                gripper_normalized=float(left_gripper[frame_idx]),
            )
            for link_name in [
                "fl_base_link", "fl_link1", "fl_link2", "fl_link3", "fl_link4", "fl_link5", "fl_link6"
            ]:
                depth = float(visualizer.point_camera(left_points[link_name], ext)[2])
                if depth > 1e-6:
                    depth_values.append(depth)

        if not hide_right_arm:
            right_points = visualizer.compute_arm_keypoints(
                visualizer.right_arm,
                joint_values_6=right_arm_values[frame_idx],
                gripper_normalized=float(right_gripper[frame_idx]),
            )
            for link_name in [
                "fr_base_link", "fr_link1", "fr_link2", "fr_link3", "fr_link4", "fr_link5", "fr_link6"
            ]:
                depth = float(visualizer.point_camera(right_points[link_name], ext)[2])
                if depth > 1e-6:
                    depth_values.append(depth)

    if not depth_values:
        return 0.2, 1.2

    depth_arr = np.asarray(depth_values, dtype=np.float64)
    depth_min = float(np.percentile(depth_arr, 5.0))
    depth_max = float(np.percentile(depth_arr, 95.0))
    if depth_max - depth_min < 1e-6:
        depth_min = float(np.min(depth_arr))
        depth_max = float(np.max(depth_arr) + 1e-3)
    return depth_min, depth_max


def build_visualizer(args: argparse.Namespace) -> SkeletonVisualizer:
    config = load_yaml(args.config)
    left_cfg = load_yaml(args.left_yaml)
    right_cfg = load_yaml(args.right_yaml)
    urdf_model = URDFModel(args.urdf)
    return SkeletonVisualizer(
        urdf_model=urdf_model,
        config=config,
        left_cfg=left_cfg,
        right_cfg=right_cfg,
        use_white_heatmap=args.white_heatmap,
        skeleton_thickness=args.skeleton_thickness,
        joint_radius=args.joint_radius,
    )


def render_episode(
    episode_hdf5: Path,
    output_mp4: Optional[Path],
    output_frames: Optional[Path],
    visualizer: SkeletonVisualizer,
    args: argparse.Namespace,
    preview_path: Optional[Path] = None,
) -> None:
    if output_mp4 is None and output_frames is None:
        raise ValueError("At least one KVAF output must be enabled.")

    with h5py.File(episode_hdf5, "r") as h5f:
        left_arm = h5f["joint_action/left_arm"][:].astype(np.float64)
        right_arm = h5f["joint_action/right_arm"][:].astype(np.float64)
        left_endpose = h5f["endpose/left_endpose"][:].astype(np.float64)
        right_endpose = h5f["endpose/right_endpose"][:].astype(np.float64)
        left_gripper = h5f["endpose/left_gripper"][:].astype(np.float64)
        right_gripper = h5f["endpose/right_gripper"][:].astype(np.float64)
        intrinsic = h5f[f"observation/{args.camera}/intrinsic_cv"][:].astype(np.float64)
        extrinsic = build_extrinsic_3x4(h5f, args.camera)

        width, height = infer_image_size(h5f, args.camera, intrinsic[0])

        num_frames = min(
            left_arm.shape[0],
            right_arm.shape[0],
            left_endpose.shape[0],
            right_endpose.shape[0],
            left_gripper.shape[0],
            right_gripper.shape[0],
            intrinsic.shape[0],
            extrinsic.shape[0],
        )
        if args.max_frames is not None:
            num_frames = min(num_frames, int(args.max_frames))
        if num_frames <= 0:
            raise ValueError(f"No frames available to render in {episode_hdf5}")

        auto_depth_min, auto_depth_max = collect_visible_depths(
            visualizer=visualizer,
            extrinsic=extrinsic,
            left_arm_values=left_arm,
            right_arm_values=right_arm,
            left_gripper=left_gripper,
            right_gripper=right_gripper,
            hide_left_arm=args.hide_left_arm,
            hide_right_arm=args.hide_right_arm,
            num_frames=num_frames,
        )
        if args.depth_range_mode == "absolute":
            if args.depth_min_m is None or args.depth_max_m is None:
                raise ValueError("When --depth-range-mode=absolute, both --depth-min-m and --depth-max-m are required.")
            depth_min = float(args.depth_min_m)
            depth_max = float(args.depth_max_m)
            if depth_max <= depth_min:
                raise ValueError("--depth-max-m must be greater than --depth-min-m")
            print(
                f"[{episode_hdf5.name}] depth mode=absolute, fixed=[{depth_min:.4f}, {depth_max:.4f}] m; "
                f"auto-estimated visible depth range=[{auto_depth_min:.4f}, {auto_depth_max:.4f}] m"
            )
        else:
            depth_min, depth_max = auto_depth_min, auto_depth_max
            print(f"[{episode_hdf5.name}] depth mode=video, range=[{depth_min:.4f}, {depth_max:.4f}] m")

        writer = None
        if output_mp4 is not None:
            output_mp4.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_mp4), fourcc, float(args.fps), (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open video writer for: {output_mp4}")
        if output_frames is not None:
            output_frames.mkdir(parents=True, exist_ok=True)

        preview_frame_index = max(0, min(num_frames - 1, num_frames // 2))

        try:
            for frame_idx in range(num_frames):
                canvas = np.zeros((height, width, 3), dtype=np.uint8)
                K = intrinsic[frame_idx]
                ext = extrinsic[frame_idx]

                if not args.hide_left_arm:
                    left_points = visualizer.compute_arm_keypoints(
                        visualizer.left_arm,
                        joint_values_6=left_arm[frame_idx],
                        gripper_normalized=float(left_gripper[frame_idx]),
                    )
                    visualizer.draw_arm(canvas, visualizer.left_arm, left_points, K, ext, depth_min, depth_max)
                    left_center_uv = visualizer.project_point(left_endpose[frame_idx, :3], K, ext)
                    visualizer.draw_heatmap(
                        canvas,
                        center_uv=left_center_uv,
                        sigma_px=float(args.heatmap_sigma),
                        radius_px=int(args.heatmap_radius),
                    )
                    visualizer.draw_pose_axes(
                        canvas,
                        endpose_xyz_qwxyz=left_endpose[frame_idx],
                        intrinsic=K,
                        extrinsic_3x4=ext,
                        axis_length_m=float(args.axis_length),
                    )
                    if visualizer.in_bounds(left_center_uv, width, height):
                        cv2.circle(canvas, (int(round(left_center_uv[0])), int(round(left_center_uv[1]))), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                if not args.hide_right_arm:
                    right_points = visualizer.compute_arm_keypoints(
                        visualizer.right_arm,
                        joint_values_6=right_arm[frame_idx],
                        gripper_normalized=float(right_gripper[frame_idx]),
                    )
                    visualizer.draw_arm(canvas, visualizer.right_arm, right_points, K, ext, depth_min, depth_max)
                    right_center_uv = visualizer.project_point(right_endpose[frame_idx, :3], K, ext)
                    visualizer.draw_heatmap(
                        canvas,
                        center_uv=right_center_uv,
                        sigma_px=float(args.heatmap_sigma),
                        radius_px=int(args.heatmap_radius),
                    )
                    visualizer.draw_pose_axes(
                        canvas,
                        endpose_xyz_qwxyz=right_endpose[frame_idx],
                        intrinsic=K,
                        extrinsic_3x4=ext,
                        axis_length_m=float(args.axis_length),
                    )
                    if visualizer.in_bounds(right_center_uv, width, height):
                        cv2.circle(canvas, (int(round(right_center_uv[0])), int(round(right_center_uv[1]))), 4, (255, 255, 255), thickness=-1, lineType=cv2.LINE_AA)

                if writer is not None:
                    writer.write(canvas)
                if output_frames is not None:
                    frame_path = output_frames / f"frame_{frame_idx:04d}.png"
                    if not cv2.imwrite(str(frame_path), canvas):
                        raise RuntimeError(f"Failed to write KVAF frame: {frame_path}")

                if preview_path is not None and frame_idx == preview_frame_index:
                    preview_path.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(preview_path), canvas)
        finally:
            if writer is not None:
                writer.release()

    if output_mp4 is not None:
        print(f"Saved KVAF video to: {output_mp4}")
    if output_frames is not None:
        print(f"Saved {num_frames} KVAF frames to: {output_frames}")
    if preview_path is not None:
        print(f"Saved preview frame to: {preview_path}")


def find_batch_episodes(dataset_root: Path, embodiment_subdir: str, input_subdir: str, episode_glob: str) -> List[Path]:
    search_root = Path(dataset_root)
    pattern = f"*/{embodiment_subdir}/{input_subdir}/{episode_glob}"
    episodes = sorted(search_root.glob(pattern))
    return [p for p in episodes if p.is_file()]


def derive_batch_output_paths(
    episode_hdf5: Path,
    embodiment_subdir: str,
    input_subdir: str,
    output_subdir: str,
) -> Tuple[Path, Path]:
    parts = list(Path(episode_hdf5).parts)
    try:
        emb_idx = len(parts) - 1 - parts[::-1].index(embodiment_subdir)
    except ValueError as e:
        raise ValueError(f"Cannot find embodiment subdir '{embodiment_subdir}' in path: {episode_hdf5}") from e

    if emb_idx + 2 >= len(parts):
        raise ValueError(f"Unexpected episode path layout: {episode_hdf5}")
    if parts[emb_idx + 1] != input_subdir:
        raise ValueError(
            f"Expected input subdir '{input_subdir}' immediately under '{embodiment_subdir}', got '{parts[emb_idx + 1]}' for {episode_hdf5}"
        )

    embodiment_dir = Path(*parts[: emb_idx + 1])
    output_root = embodiment_dir / output_subdir
    return output_root / f"{episode_hdf5.stem}.mp4", output_root / episode_hdf5.stem


def select_outputs(
    output_format: str,
    video_path: Path,
    frames_path: Path,
) -> Tuple[Optional[Path], Optional[Path]]:
    output_mp4 = video_path if output_format in {"video", "both"} else None
    output_frames = frames_path if output_format in {"frames", "both"} else None
    return output_mp4, output_frames


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Kinematic Visual Action Fields from RoboTwin episodes.")

    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument("--hdf5", type=Path, help="Path to a single episode HDF5")
    mode_group.add_argument("--dataset-root", type=Path, help="Batch mode: dataset root that contains task folders")

    parser.add_argument("--urdf", type=Path, required=True, help="Path to URDF file")
    parser.add_argument("--config", type=Path, required=True, help="Path to embodiment config.yml")
    parser.add_argument("--left-yaml", type=Path, required=True, help="Path to curobo_left.yml")
    parser.add_argument("--right-yaml", type=Path, required=True, help="Path to curobo_right.yml")
    parser.add_argument("--camera", type=str, default="head_camera", help="Camera name to use, default=head_camera")

    parser.add_argument("--output", type=Path, default=None, help="Single-episode output path. Use a directory for frames or an .mp4 path for video.")
    parser.add_argument(
        "--output-format",
        choices=["frames", "video", "both"],
        default="frames",
        help="Write training-ready PNG frames, a preview video, or both.",
    )
    parser.add_argument("--fps", type=float, default=30, help="Preview-video frame rate.")
    parser.add_argument("--heatmap-sigma", type=float, default=8, help="Heatmap Gaussian sigma in pixels")
    parser.add_argument("--heatmap-radius", type=int, default=27, help="Heatmap cutoff radius in pixels")
    parser.add_argument("--axis-length", type=float, default=0.08, help="Pose axis length in meters")
    parser.add_argument("--skeleton-thickness", type=int, default=2, help="Skeleton line thickness")
    parser.add_argument("--joint-radius", type=int, default=3, help="Joint point radius")
    parser.add_argument("--white-heatmap", action="store_true", help="Use pure white heatmap instead of warm fade")
    parser.add_argument("--hide-left-arm", action="store_true", help="Hide left arm visualization")
    parser.add_argument("--hide-right-arm", action="store_true", help="Hide right arm visualization")
    parser.add_argument("--save-preview", type=Path, default=None, help="Single-file mode only: optional PNG path for saving a mid-frame preview")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional limit on number of frames to render")
    parser.add_argument(
        "--depth-range-mode",
        type=str,
        default="video",
        choices=["video", "absolute"],
        help="Depth normalization mode: 'video' uses this video's depth percentiles; 'absolute' uses fixed meter bounds.",
    )
    parser.add_argument("--depth-min-m", type=float, default=None, help="Fixed near depth bound in meters when --depth-range-mode=absolute")
    parser.add_argument("--depth-max-m", type=float, default=None, help="Fixed far depth bound in meters when --depth-range-mode=absolute")

    parser.add_argument("--embodiment-subdir", type=str, default="aloha-agilex_clean_50", help="Batch mode: fixed embodiment subdir under each task folder")
    parser.add_argument("--input-subdir", type=str, default="data", help="Batch mode: where episode hdf5 files live under the embodiment folder")
    parser.add_argument("--output-subdir", type=str, default="kvaf", help="Batch mode: output directory under each embodiment folder")
    parser.add_argument("--episode-glob", type=str, default="episode*.hdf5", help="Batch mode: glob for episode files inside the input subdir")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing KVAF outputs")
    parser.add_argument("--continue-on-error", action="store_true", help="Batch mode: continue rendering other episodes if one episode fails")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    visualizer = build_visualizer(args)

    if args.hdf5 is not None:
        if args.output is None:
            raise ValueError("Single-file mode requires --output")
        if args.output.suffix.lower() == ".mp4":
            video_path = args.output
            frames_path = args.output.with_suffix("")
        else:
            frames_path = args.output
            video_path = args.output.with_suffix(".mp4")
        output_mp4, output_frames = select_outputs(args.output_format, video_path, frames_path)
        existing = [path for path in (output_mp4, output_frames) if path is not None and path.exists()]
        if existing and not args.overwrite:
            raise FileExistsError(f"Output already exists: {existing[0]}. Use --overwrite to replace it.")
        render_episode(
            episode_hdf5=args.hdf5,
            output_mp4=output_mp4,
            output_frames=output_frames,
            visualizer=visualizer,
            args=args,
            preview_path=args.save_preview,
        )
        return

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    episodes = find_batch_episodes(
        dataset_root=dataset_root,
        embodiment_subdir=args.embodiment_subdir,
        input_subdir=args.input_subdir,
        episode_glob=args.episode_glob,
    )
    if not episodes:
        raise FileNotFoundError(
            f"No episode files found under {dataset_root} matching */{args.embodiment_subdir}/{args.input_subdir}/{args.episode_glob}"
        )

    print(f"Found {len(episodes)} episode files for batch rendering.")
    success = 0
    skipped = 0
    failed = 0

    for idx, episode_hdf5 in enumerate(episodes, start=1):
        video_path, frames_path = derive_batch_output_paths(
            episode_hdf5=episode_hdf5,
            embodiment_subdir=args.embodiment_subdir,
            input_subdir=args.input_subdir,
            output_subdir=args.output_subdir,
        )
        output_mp4, output_frames = select_outputs(args.output_format, video_path, frames_path)
        rel_episode = episode_hdf5.relative_to(dataset_root)
        displayed_output = output_frames if output_frames is not None else output_mp4
        rel_output = displayed_output.relative_to(dataset_root)
        print(f"[{idx}/{len(episodes)}] {rel_episode} -> {rel_output}")

        existing = [path for path in (output_mp4, output_frames) if path is not None and path.exists()]
        if existing and not args.overwrite:
            print(f"  skip: output exists (use --overwrite to replace): {existing[0]}")
            skipped += 1
            continue

        try:
            render_episode(
                episode_hdf5=episode_hdf5,
                output_mp4=output_mp4,
                output_frames=output_frames,
                visualizer=visualizer,
                args=args,
                preview_path=None,
            )
            success += 1
        except Exception as exc:
            failed += 1
            print(f"  error: {episode_hdf5}\n{exc}")
            if args.continue_on_error:
                traceback.print_exc()
                continue
            raise

    print(f"Batch finished. success={success}, skipped={skipped}, failed={failed}")


if __name__ == "__main__":
    main()
