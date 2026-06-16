from __future__ import annotations

import cv2
import numpy as np
import torch

from isaaclab.utils.math import quat_apply, quat_mul

EE_TO_CAMERA_OFFSET_POS = [0.0, 0.0, 0.0]
EE_TO_CAMERA_OFFSET_QUAT = [1.0, 0.0, 0.0, 0.0]
ARUCO_LOCAL_POS_IN_OBJECT = [0.0, 0.25162, 0.1208]

def _get_num_envs(env) -> int:
    if hasattr(env, "num_envs"):
        return env.num_envs
    return env.scene.num_envs


def _get_device(env):
    if hasattr(env, "device"):
        return env.device
    return env.scene.device


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    q_conj = q.clone()
    q_conj[..., 1:] *= -1.0
    return q_conj


def _ensure_perception_buffers(env):
    num_envs = _get_num_envs(env)
    device = _get_device(env)

    if not hasattr(env, "box_pos_w_from_perception"):
        env.box_pos_w_from_perception = torch.zeros(
            num_envs, 3, dtype=torch.float32, device=device
        )

    if not hasattr(env, "box_quat_w_from_perception"):
        env.box_quat_w_from_perception = torch.zeros(
            num_envs, 4, dtype=torch.float32, device=device
        )
        env.box_quat_w_from_perception[:, 0] = 1.0

    if not hasattr(env, "marker_detected_once"):
        env.marker_detected_once = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )

    if not hasattr(env, "marker_pose_missed_count"):
        env.marker_pose_missed_count = torch.zeros(
            num_envs, dtype=torch.long, device=device
        )

    if not hasattr(env, "use_perception_pose"):
        env.use_perception_pose = False

    if not hasattr(env, "use_perception_pose_per_env"):
        env.use_perception_pose_per_env = torch.zeros(
            num_envs, dtype=torch.bool, device=device
        )


def object_visible_obs(env) -> torch.Tensor:
    _ensure_perception_buffers(env)
    visible = (env.marker_pose_missed_count <= 3).float()
    return visible.unsqueeze(-1)


def get_aruco_marker_gt_pose_w(env):
    _ensure_perception_buffers(env)

    obj = env.scene["object"]

    device = _get_device(env)

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w

    aruco_local_pos = torch.tensor(
        ARUCO_LOCAL_POS_IN_OBJECT,
        dtype=obj_pos_w.dtype,
        device=device,
    ).unsqueeze(0).repeat(obj_pos_w.shape[0], 1)

    marker_pos_w = obj_pos_w + quat_apply(obj_quat_w, aruco_local_pos)
    marker_quat_w = obj_quat_w

    # if obj_pos_w.shape[0] > 0:
    #     print("[ARUCO CHECK]")
    #     print("box   :", obj_pos_w[0].detach().cpu().numpy())
    #     print("aruco :", marker_pos_w[0].detach().cpu().numpy())
    #     print("diff  :", (marker_pos_w[0] - obj_pos_w[0]).detach().cpu().numpy())

    marker_pos_w = torch.nan_to_num(
        marker_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    marker_quat_w = torch.nan_to_num(
        marker_quat_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return marker_pos_w, marker_quat_w


def reset_aruco_perception_state(env, env_ids: torch.Tensor | None = None):
    _ensure_perception_buffers(env)

    num_envs = _get_num_envs(env)
    device = _get_device(env)

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    marker_gt_pos_w, marker_gt_quat_w = get_aruco_marker_gt_pose_w(env)

    env.box_pos_w_from_perception[env_ids] = marker_gt_pos_w[env_ids]
    env.box_quat_w_from_perception[env_ids] = marker_gt_quat_w[env_ids]
    env.marker_detected_once[env_ids] = False
    env.marker_pose_missed_count[env_ids] = 0
    env.use_perception_pose_per_env[env_ids] = False
    env.use_perception_pose = bool(env.use_perception_pose_per_env.any().item())


def set_box_pose_from_perception(
    env,
    box_pos_w: torch.Tensor,
    box_quat_w: torch.Tensor,
    env_ids=None,
):
    _ensure_perception_buffers(env)

    if env_ids is None:
        env.box_pos_w_from_perception[:] = box_pos_w
        env.box_quat_w_from_perception[:] = box_quat_w
        env.marker_detected_once[:] = True
        env.use_perception_pose_per_env[:] = True
    else:
        env.box_pos_w_from_perception[env_ids] = box_pos_w
        env.box_quat_w_from_perception[env_ids] = box_quat_w
        env.marker_detected_once[env_ids] = True
        env.use_perception_pose_per_env[env_ids] = True

    env.use_perception_pose = bool(env.use_perception_pose_per_env.any().item())


def get_box_pose_w(env):
    _ensure_perception_buffers(env)

    marker_gt_pos_w, marker_gt_quat_w = get_aruco_marker_gt_pose_w(env)

    use_per_env = env.use_perception_pose_per_env.unsqueeze(-1)

    marker_pos_w = torch.where(
        use_per_env,
        env.box_pos_w_from_perception,
        marker_gt_pos_w,
    )

    marker_quat_w = torch.where(
        use_per_env,
        env.box_quat_w_from_perception,
        marker_gt_quat_w,
    )

    marker_pos_w = torch.nan_to_num(
        marker_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    marker_quat_w = torch.nan_to_num(
        marker_quat_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return marker_pos_w, marker_quat_w


def object_position_in_ee_frame_ext(env):
    marker_pos_w, _ = get_box_pose_w(env)

    ee_frame = env.scene["ee_frame"]

    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    rel_pos_w = marker_pos_w - ee_pos_w

    ee_quat_inv = quat_conjugate(ee_quat_w)
    marker_pos_ee = quat_apply(ee_quat_inv, rel_pos_w)

    marker_pos_ee = torch.nan_to_num(
        marker_pos_ee,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    return torch.clamp(marker_pos_ee, -2.0, 2.0)


def update_aruco_perception_pose(env, env_ids: torch.Tensor | None = None):
    _ensure_perception_buffers(env)

    num_envs = _get_num_envs(env)
    device = _get_device(env)

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)
    elif not isinstance(env_ids, torch.Tensor):
        env_ids = torch.tensor(env_ids, dtype=torch.long, device=device)
    else:
        env_ids = env_ids.to(device=device, dtype=torch.long)

    cam = env.scene["wrist_camera"]
    ee_frame = env.scene["ee_frame"]

    rgb_tensors = cam.data.output["rgb"]
    depth_tensors = cam.data.output["distance_to_image_plane"]
    intrinsic_matrices = cam.data.intrinsic_matrices

    ee_pos_w = ee_frame.data.target_pos_w[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]

    if hasattr(cam.data, "pos_w") and hasattr(cam.data, "quat_w_ros"):
        cam_pos_w = cam.data.pos_w
        final_cam_quat = cam.data.quat_w_ros
    else:
        offset_quat = torch.tensor(
            EE_TO_CAMERA_OFFSET_QUAT,
            device=device,
            dtype=torch.float32,
        ).repeat(num_envs, 1)

        final_cam_quat = quat_mul(ee_quat_w, offset_quat)

        offset_pos = torch.tensor(
            EE_TO_CAMERA_OFFSET_POS,
            device=device,
            dtype=torch.float32,
        ).repeat(num_envs, 1)

        cam_pos_w = ee_pos_w + quat_apply(ee_quat_w, offset_pos)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())

    updated_pos_w = env.box_pos_w_from_perception.clone()
    detected = torch.zeros(num_envs, dtype=torch.bool, device=device)

    for env_id_tensor in env_ids:
        env_id = int(env_id_tensor.item())

        rgb_np = rgb_tensors[env_id].detach().cpu().numpy()

        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]

        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)

        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)

        if ids is None:
            continue

        ids_flat = ids.flatten()

        if 0 not in ids_flat:
            continue

        idx = np.where(ids_flat == 0)[0][0]
        pts = corners[idx].reshape(4, 2)

        u = int(pts[:, 0].mean())
        v = int(pts[:, 1].mean())

        depth_np = depth_tensors[env_id].squeeze().detach().cpu().numpy()

        h, w = depth_np.shape[:2]
        r = 3

        patch = depth_np[
            max(v - r, 0):min(v + r + 1, h),
            max(u - r, 0):min(u + r + 1, w),
        ]

        valid = patch[np.isfinite(patch)]
        valid = valid[valid > 0]

        if len(valid) == 0:
            continue

        depth_val = float(np.median(valid))

        if not np.isfinite(depth_val) or depth_val <= 0.0:
            continue

        K = intrinsic_matrices[env_id]

        fx = float(K[0, 0])
        fy = float(K[1, 1])
        cx = float(K[0, 2])
        cy = float(K[1, 2])

        if not np.isfinite(fx) or not np.isfinite(fy):
            continue

        if abs(fx) < 1e-6 or abs(fy) < 1e-6:
            continue

        z_c = depth_val
        x_c = (float(u) - cx) * z_c / fx
        y_c = (float(v) - cy) * z_c / fy

        p_cam_ros = torch.tensor(
            [x_c, y_c, z_c],
            device=device,
            dtype=torch.float32,
        )

        p_world = cam_pos_w[env_id] + quat_apply(
            final_cam_quat[env_id].unsqueeze(0),
            p_cam_ros.unsqueeze(0),
        ).squeeze(0)

        if not torch.isfinite(p_world).all():
            continue

        updated_pos_w[env_id] = p_world
        detected[env_id] = True

    env.box_pos_w_from_perception[:] = updated_pos_w
    env.box_quat_w_from_perception[:] = env.scene["object"].data.root_quat_w

    detected_env_ids = env_ids[detected[env_ids]]
    missed_env_ids = env_ids[~detected[env_ids]]

    env.marker_detected_once[detected_env_ids] = True
    env.marker_pose_missed_count[detected_env_ids] = 0
    env.marker_pose_missed_count[missed_env_ids] += 1

    env.use_perception_pose_per_env[:] = env.marker_detected_once
    env.use_perception_pose = bool(env.marker_detected_once.any().item())
