# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom reward functions for the M0609 lift task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

from .perception import get_box_pose_w

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    std: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    흡착 전 접근 보상.

    suction_tcp와 현재 target marker 사이 거리가 가까울수록 보상을 준다.
    suction_attached=True가 되면 이 보상은 0으로 꺼진다.
    """

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # 현재 target marker world position.
    # get_box_pose_w 내부에서 GT fallback 또는 perception 좌표를 선택한다.
    marker_pos_w, _ = get_box_pose_w(env)

    # suction_tcp world position
    ee_w = ee_frame.data.target_pos_w[:, 0, :]

    # suction_tcp와 marker 중심 사이 거리
    dist = torch.norm(marker_pos_w - ee_w, dim=1)
    dist = torch.nan_to_num(dist, nan=10.0, posinf=10.0, neginf=10.0)

    # 거리가 가까울수록 1에 가까운 보상
    std = max(float(std), 1e-6)
    proximity_reward = 1.0 - torch.tanh(dist / std)

    # 흡착 성공 후에는 접근 보상을 제거
    if hasattr(env, "suction_attached"):
        proximity_reward = proximity_reward * (1.0 - env.suction_attached.float())

    return torch.nan_to_num(
        proximity_reward,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )


def joint_position_target_reward(
    env,
    target_joint_pos: list[float],
    std: float = 0.5,
    minimal_height: float = 0.35,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    물체가 일정 높이 이상 올라간 뒤, 목표 joint 자세에 가까워질수록 보상.

    현재 구조에서는 object root z가 minimal_height보다 높을 때만 활성화된다.
    """

    robot = env.scene[asset_cfg.name]
    obj = env.scene[object_cfg.name]

    # 현재 joint position
    joint_pos = robot.data.joint_pos

    # 목표 joint position tensor 생성
    target = torch.tensor(
        target_joint_pos,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    ).unsqueeze(0)

    # 목표 joint와 현재 joint 사이 오차
    joint_error = joint_pos - target
    error_norm = torch.norm(joint_error, dim=1)

    # joint 오차가 작을수록 1에 가까운 보상
    std = max(float(std), 1e-6)
    joint_reward = torch.exp(-error_norm / std)

    # 물체가 minimal_height 이상 올라간 경우에만 목표 자세 보상 활성화
    object_z = obj.data.root_pos_w[:, 2]
    lifted_mask = object_z > minimal_height

    return lifted_mask.float() * joint_reward


def ee_low_height_penalty(
    env,
    min_height: float = 0.2,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    EE가 너무 낮게 내려가면 penalty 원값 1.0 반환.

    RewardsCfg에서 음수 weight를 곱해 실제 penalty로 사용한다.
    """

    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    # suction_tcp world position
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_z = ee_pos_w[:, 2]

    # EE 높이가 기준보다 낮으면 penalty 발생
    too_low = ee_z < min_height

    return too_low.float()


def box_rack_collision_penalty(
    env,
    threshold: float = 1.0,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("box_rack_contact"),
) -> torch.Tensor:
    """
    박스와 RackFrame 충돌 penalty.

    contact force가 threshold보다 크면 1.0을 반환한다.
    RewardsCfg에서 음수 weight를 곱해 실제 penalty로 사용한다.
    """

    contact_sensor = env.scene.sensors[sensor_cfg.name]

    # filter_prim_paths_expr를 사용한 contact sensor인 경우
    if hasattr(contact_sensor.data, "force_matrix_w") and contact_sensor.data.force_matrix_w is not None:
        forces = contact_sensor.data.force_matrix_w
        force_norm = torch.norm(forces, dim=-1)
        max_force = force_norm.amax(dim=(1, 2))

    # 일반 contact sensor fallback
    else:
        forces = contact_sensor.data.net_forces_w
        force_norm = torch.norm(forces, dim=-1)
        max_force = force_norm.amax(dim=1)

    # 접촉 힘이 threshold보다 크면 충돌로 판단
    collision = max_force > threshold

    return collision.float()


def suction_attached_reward(
    env,
) -> torch.Tensor:
    """
    suction attached 상태 보상.

    env.suction_attached=True이면 1.0을 반환한다.
    즉 흡착 순간 1회 보상이 아니라, 흡착 상태가 유지되는 동안 계속 들어가는 보상이다.
    """

    if hasattr(env, "suction_attached"):
        return env.suction_attached.float()

    # suction buffer가 아직 생성되지 않은 초기 상태 fallback
    num_envs = env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs
    device = env.device if hasattr(env, "device") else env.scene.device

    return torch.zeros(
        num_envs,
        dtype=torch.float32,
        device=device,
    )