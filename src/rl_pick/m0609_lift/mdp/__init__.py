# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Re-exports the upstream lift-task MDP functions (claude_rl.md §5).

This thin shim lets m0609_lift modules do ``from . import mdp`` and access
all of ``isaaclab_tasks.manager_based.manipulation.lift.mdp`` without
modifying upstream files.
"""

from isaaclab_tasks.manager_based.manipulation.lift.mdp import *  # noqa: F401, F403
from .rewards import (
    gripper_close_near_object, 
    joint_position_target_reward, 
    ee_low_height_penalty,  # noqa: F401
    box_rack_collision_penalty,
    suction_attached_reward,
)
from .target_joint_error_obs import target_joint_error_obs
from .suction2 import reset_suction_state, update_suction_attachment  # noqa: F401

from .perception import (
    get_box_pose_w,
    object_position_in_ee_frame_ext,
    object_visible_obs,
    update_aruco_perception_pose,
    reset_aruco_perception_state,
)  # noqa: F401

