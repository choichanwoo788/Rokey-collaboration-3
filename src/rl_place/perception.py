from __future__ import annotations

import cv2
import numpy as np
import torch
from isaaclab.utils.math import quat_apply, quat_mul

# ======================================================================
# 🔥 준호님의 PC 환경 절대 경로에 맞게 수정하세요!
# ======================================================================
BOX_USD_PATH = r"C:/Users/milkc/OneDrive/Desktop/Collected_box_aruco2/Collected_box_aruco2/box_aruco2.usd"

# [캘리브레이션] 손목 TCP 중심에서 카메라 렌즈 중심까지의 물리적 오프셋
EE_TO_CAMERA_OFFSET_POS = [0.0, 0.0, 0.0]  # 필요시 조절 (m 단위)
EE_TO_CAMERA_OFFSET_QUAT = [1.0, 0.0, 0.0, 0.0]  # [w, x, y, z]

def _get_num_envs(env) -> int:
    if hasattr(env, "num_envs"):
        return env.num_envs
    return env.scene.num_envs

def _get_device(env):
    if hasattr(env, "device"):
        return env.device
    return env.scene.device

def _ensure_perception_buffers(env):
    num_envs = _get_num_envs(env)
    device = _get_device(env)

    if not hasattr(env, "box_pos_w_from_perception"):
        env.box_pos_w_from_perception = torch.zeros(num_envs, 3, dtype=torch.float32, device=device)
    if not hasattr(env, "box_quat_w_from_perception"):
        env.box_quat_w_from_perception = torch.zeros(num_envs, 4, dtype=torch.float32, device=device)
        env.box_quat_w_from_perception[:, 0] = 1.0
    if not hasattr(env, "use_perception_pose"):
        env.use_perception_pose = False

def set_box_pose_from_perception(env, box_pos_w: torch.Tensor, box_quat_w: torch.Tensor, env_ids=None):
    _ensure_perception_buffers(env)
    if env_ids is None:
        env.box_pos_w_from_perception[:] = box_pos_w
        env.box_quat_w_from_perception[:] = box_quat_w
    else:
        env.box_pos_w_from_perception[env_ids] = box_pos_w
        env.box_quat_w_from_perception[env_ids] = box_quat_w
    env.use_perception_pose = True

def get_box_pose_w(env):
    _ensure_perception_buffers(env)
    obj = env.scene["object"]
    if env.use_perception_pose:
        return env.box_pos_w_from_perception, env.box_quat_w_from_perception
    return obj.data.root_pos_w, obj.data.root_quat_w

def object_position_in_robot_root_frame_ext(env):
    box_pos_w, _ = get_box_pose_w(env)
    robot = env.scene["robot"]
    robot_root_pos_w = robot.data.root_pos_w
    return box_pos_w - robot_root_pos_w

def update_aruco_perception_pose(env):
    """매 스텝 호출되어 RGB-D기반 마커 실시간 추적 및 가려짐 방지 기억력 탑재"""
    _ensure_perception_buffers(env)
    num_envs = _get_num_envs(env)
    device = _get_device(env)
    cam = env.scene["wrist_camera"]
    ee_frame = env.scene["ee_frame"]
    
    rgb_tensors = cam.data.output["rgb"]
    depth_tensors = cam.data.output["distance_to_image_plane"]
    intrinsic_matrices = cam.data.intrinsic_matrices
    
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]
    
    if hasattr(cam.data, "quat_w_ros"):
        final_cam_quat = cam.data.quat_w_ros
    else:
        offset_quat = torch.tensor(EE_TO_CAMERA_OFFSET_QUAT, device=device, dtype=torch.float32).repeat(num_envs, 1)
        final_cam_quat = quat_mul(ee_quat_w, offset_quat)

    offset_pos = torch.tensor(EE_TO_CAMERA_OFFSET_POS, device=device, dtype=torch.float32).repeat(num_envs, 1)
    cam_pos_w = ee_pos_w + quat_apply(ee_quat_w, offset_pos)

    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(dictionary, cv2.aruco.DetectorParameters())
    updated_pos_w = env.box_pos_w_from_perception.clone()

    for env_id in range(num_envs):
        rgb_np = rgb_tensors[env_id].detach().cpu().numpy()
        if rgb_np.shape[-1] == 4:
            rgb_np = rgb_np[..., :3]
        if rgb_np.max() <= 1.0:
            rgb_np = (rgb_np * 255).astype(np.uint8)
        else:
            rgb_np = rgb_np.astype(np.uint8)
            
        gray = cv2.cvtColor(rgb_np, cv2.COLOR_RGB2GRAY)
        corners, ids, _ = detector.detectMarkers(gray)
        
        # 마커 탐지 성공 시 정밀 복원
        if ids is not None and 0 in ids.flatten():
            idx = np.where(ids.flatten() == 0)[0][0]
            pts = corners[idx].reshape(4, 2)
            u, v = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            
            depth_np = depth_tensors[env_id].squeeze().detach().cpu().numpy()
            h, w = depth_np.shape[:2]
            r = 3
            patch = depth_np[max(v-r, 0):min(v+r+1, h), max(u-r, 0):min(u+r+1, w)]
            valid = patch[np.isfinite(patch)]
            valid = valid[valid > 0]
            
            if len(valid) > 0:
                depth_val = np.median(valid)
                K = intrinsic_matrices[env_id]
                z_c = float(depth_val)
                x_c = (float(u) - float(K[0, 2])) * z_c / float(K[0, 0])
                y_c = (float(v) - float(K[1, 2])) * z_c / float(K[1, 1])
                
                p_cam_ros = torch.tensor([x_c, y_c, z_c], device=device, dtype=torch.float32)
                p_world = cam_pos_w[env_id] + quat_apply(final_cam_quat[env_id].unsqueeze(0), p_cam_ros.unsqueeze(0)).squeeze(0)
                updated_pos_w[env_id] = p_world
        else:
            # 💡 [보완 완료] 그리퍼에 마커가 가려지면 완전히 영점으로 밀지 않고 이전 스텝의 계산값을 유지(기억력)
            updated_pos_w[env_id] = env.box_pos_w_from_perception[env_id]

    env.box_pos_w_from_perception[:] = updated_pos_w
    env.box_quat_w_from_perception[:] = env.scene["object"].data.root_quat_w
    env.use_perception_pose = True