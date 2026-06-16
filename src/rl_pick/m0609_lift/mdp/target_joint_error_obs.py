from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg


def target_joint_error_obs(
    env,
    target_joint_pos: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:

    robot = env.scene[asset_cfg.name]
    joint_pos = robot.data.joint_pos

    target = torch.tensor(
        target_joint_pos,
        dtype=joint_pos.dtype,
        device=joint_pos.device,
    ).unsqueeze(0)

    return target - joint_pos