# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Custom MDP terms for the M0609 pre-attached carry-and-place task.

Task definition:
- The box starts already attached to ``suction_tcp`` at every reset.
- While attached, the box is kinematically slaved to the suction TCP with a fixed offset.
- Release is automatic when the attached box reaches the commanded target region.
- Success is a stable, detached box near the target/platform height.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms, quat_apply, quat_mul

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _num_envs(env) -> int:
    return env.num_envs if hasattr(env, "num_envs") else env.scene.num_envs


def _device(env):
    return env.device if hasattr(env, "device") else env.scene.device


def _env_ids_tensor(env, env_ids) -> torch.Tensor:
    all_ids = torch.arange(_num_envs(env), device=_device(env), dtype=torch.long)
    if env_ids is None:
        return all_ids
    if isinstance(env_ids, slice):
        return all_ids[env_ids]
    return env_ids.to(device=_device(env), dtype=torch.long)


def _ensure_suction_buffers(env):
    num_envs = _num_envs(env)
    device = _device(env)
    if not hasattr(env, "suction_attached"):
        env.suction_attached = torch.ones(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "suction_released_once"):
        env.suction_released_once = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "suction_release_step"):
        env.suction_release_step = torch.full((num_envs,), -1, dtype=torch.long, device=device)


def _target_pos_w(env, command_name: str = "object_pose", robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Convert the generated command position from robot-root frame to world frame."""
    robot = env.scene[robot_cfg.name]
    command = env.command_manager.get_command(command_name)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    return des_pos_w


def _attached_object_pose(env, attach_offset_pos=(0.0, 0.0, -0.075), attach_offset_quat=(1.0, 0.0, 0.0, 0.0)):
    """Return object world pose implied by suction_tcp and a fixed TCP->object offset."""
    device = _device(env)
    ee_frame: FrameTransformer = env.scene["ee_frame"]
    ee_pos_w = ee_frame.data.target_pos_w[:, 0]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0]

    offset_pos = torch.tensor(attach_offset_pos, dtype=torch.float32, device=device).repeat(ee_pos_w.shape[0], 1)
    offset_quat = torch.tensor(attach_offset_quat, dtype=torch.float32, device=device).repeat(ee_pos_w.shape[0], 1)

    object_pos_w = ee_pos_w + quat_apply(ee_quat_w, offset_pos)
    object_quat_w = quat_mul(ee_quat_w, offset_quat)
    return object_pos_w, object_quat_w


# -----------------------------------------------------------------------------
# Reset / attach / detach events
# -----------------------------------------------------------------------------


def reset_suction_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None):
    """Reset suction state to pre-attached for each reset environment."""
    _ensure_suction_buffers(env)
    ids = _env_ids_tensor(env, env_ids)
    env.suction_attached[ids] = True
    env.suction_released_once[ids] = False
    env.suction_release_step[ids] = -1


def attach_object_to_suction_tcp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    attach_offset_pos=(0.0, 0.0, -0.075),
    attach_offset_quat=(1.0, 0.0, 0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Place reset objects directly at suction_tcp + fixed offset and mark them attached."""
    _ensure_suction_buffers(env)
    ids = _env_ids_tensor(env, env_ids)
    obj: RigidObject = env.scene[object_cfg.name]

    pos_w, quat_w = _attached_object_pose(env, attach_offset_pos, attach_offset_quat)
    pose = torch.cat([pos_w[ids], quat_w[ids]], dim=-1)
    vel = torch.zeros((len(ids), 6), dtype=torch.float32, device=_device(env))

    obj.write_root_pose_to_sim(pose, env_ids=ids)
    obj.write_root_velocity_to_sim(vel, env_ids=ids)
    env.suction_attached[ids] = True
    env.suction_released_once[ids] = False
    env.suction_release_step[ids] = -1


def update_suction_attachment(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    attach_offset_pos=(0.0, 0.0, -0.075),
    attach_offset_quat=(1.0, 0.0, 0.0, 0.0),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Keep attached objects slaved to suction_tcp.

    This intentionally does NOT use a distance threshold: the task starts attached,
    so attachment state is controlled by reset/release state, not by approaching the box.
    """
    _ensure_suction_buffers(env)
    ids = _env_ids_tensor(env, env_ids)
    if len(ids) == 0:
        return

    attached_ids = ids[env.suction_attached[ids]]
    if len(attached_ids) == 0:
        return

    obj: RigidObject = env.scene[object_cfg.name]
    pos_w, quat_w = _attached_object_pose(env, attach_offset_pos, attach_offset_quat)
    pose = torch.cat([pos_w[attached_ids], quat_w[attached_ids]], dim=-1)
    vel = torch.zeros((len(attached_ids), 6), dtype=torch.float32, device=_device(env))

    obj.write_root_pose_to_sim(pose, env_ids=attached_ids)
    obj.write_root_velocity_to_sim(vel, env_ids=attached_ids)


def automatic_release_at_target(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    command_name: str = "object_pose",
    xy_threshold: float = 0.04,
    z_threshold: float = 0.05,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
):
    """Detach suction automatically once the attached box reaches the target region."""
    _ensure_suction_buffers(env)
    ids = _env_ids_tensor(env, env_ids)
    if len(ids) == 0:
        return

    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    diff = obj.data.root_pos_w - target
    in_region = (torch.linalg.vector_norm(diff[:, :2], dim=1) < xy_threshold) & (torch.abs(diff[:, 2]) < z_threshold)
    release_ids = ids[env.suction_attached[ids] & in_region[ids]]
    if len(release_ids) == 0:
        return

    env.suction_attached[release_ids] = False
    env.suction_released_once[release_ids] = True
    if hasattr(env, "episode_length_buf"):
        env.suction_release_step[release_ids] = env.episode_length_buf[release_ids].long()
    else:
        env.suction_release_step[release_ids] = 0
    # Let physics take over from a gentle, deterministic release.
    obj.write_root_velocity_to_sim(torch.zeros((len(release_ids), 6), dtype=torch.float32, device=_device(env)), env_ids=release_ids)


# -----------------------------------------------------------------------------
# Rewards / penalties
# -----------------------------------------------------------------------------


def keep_object_attached(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward maintaining attachment before the target release condition."""
    _ensure_suction_buffers(env)
    return env.suction_attached.float()


def attached_object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reward for moving the carried object toward the target."""
    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    dist = torch.linalg.vector_norm(obj.data.root_pos_w - target, dim=1)
    return 1.0 - torch.tanh(dist / std)


def attached_object_xy_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str = "object_pose",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Fine reward for horizontal alignment over the platform."""
    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    xy_dist = torch.linalg.vector_norm((obj.data.root_pos_w - target)[:, :2], dim=1)
    return 1.0 - torch.tanh(xy_dist / std)


def release_near_target(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.05,
    z_threshold: float = 0.06,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse reward when suction has released in the target region."""
    _ensure_suction_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    diff = obj.data.root_pos_w - target
    near = (torch.linalg.vector_norm(diff[:, :2], dim=1) < xy_threshold) & (torch.abs(diff[:, 2]) < z_threshold)
    return ((~env.suction_attached) & near).float()


def stable_placement(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.05,
    z_threshold: float = 0.06,
    lin_vel_threshold: float = 0.05,
    ang_vel_threshold: float = 0.2,
    settle_steps: int = 5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward a detached, near-target, low-velocity object."""
    _ensure_suction_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    diff = obj.data.root_pos_w - target
    near = (torch.linalg.vector_norm(diff[:, :2], dim=1) < xy_threshold) & (torch.abs(diff[:, 2]) < z_threshold)
    lin_slow = torch.linalg.vector_norm(obj.data.root_vel_w[:, :3], dim=1) < lin_vel_threshold
    ang_slow = torch.linalg.vector_norm(obj.data.root_vel_w[:, 3:], dim=1) < ang_vel_threshold
    if hasattr(env, "episode_length_buf"):
        settled = (env.episode_length_buf.long() - env.suction_release_step) >= settle_steps
    else:
        settled = torch.ones_like(env.suction_attached, dtype=torch.bool)
    return ((~env.suction_attached) & near & lin_slow & ang_slow & settled).float()


def early_drop_penalty(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.08,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty for becoming detached before reaching the target XY region."""
    _ensure_suction_buffers(env)
    obj: RigidObject = env.scene[object_cfg.name]
    target = _target_pos_w(env, command_name)
    xy_dist = torch.linalg.vector_norm((obj.data.root_pos_w - target)[:, :2], dim=1)
    return ((~env.suction_attached) & (xy_dist > xy_threshold)).float()


# -----------------------------------------------------------------------------
# Terminations
# -----------------------------------------------------------------------------


def object_stable_at_target(
    env: ManagerBasedRLEnv,
    command_name: str = "object_pose",
    xy_threshold: float = 0.05,
    z_threshold: float = 0.06,
    lin_vel_threshold: float = 0.03,
    ang_vel_threshold: float = 0.15,
    settle_steps: int = 5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Success termination for pre-attached carry-and-place."""
    return stable_placement(
        env,
        command_name=command_name,
        xy_threshold=xy_threshold,
        z_threshold=z_threshold,
        lin_vel_threshold=lin_vel_threshold,
        ang_vel_threshold=ang_vel_threshold,
        settle_steps=settle_steps,
        object_cfg=object_cfg,
    ).bool()
