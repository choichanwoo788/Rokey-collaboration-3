# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract LiftEnvCfg for the M0609 suction lift task."""

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.sensors import ContactSensorCfg
from . import mdp


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            # pos=[-2.0, 2.0 ,-0.9],
            pos=[-2.0, 0.0, -0.3],
            # rot=[1.0, 0.0, 0.0, 0.0],
            rot=[0.707, 0.0, 0.0, 0.707],
            
        ),
        spawn=UsdFileCfg(
            usd_path=f"/home/rokey/Downloads/pick_the_box/Table.usd"
        ),
    )

    RackFrame = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RackFrame",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-2.0, 2.0 ,-0.9],
            # pos=[-2.0, 0.0 ,0.0],
            # rot=[1.0, 0.0, 0.0, 0.0],
            rot=[0.707, 0.0, 0.0, 0.707],
        ),
        spawn=UsdFileCfg(
            usd_path=f"/home/rokey/Downloads/pick_the_box/RackFrame.usd"
        ),
    )
    
    box_rack_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Object/box_aruco2",
        update_period=0.0,
        history_length=3,
        debug_vis=False,
        track_air_time=False,
        filter_prim_paths_expr=[
            "{ENV_REGEX_NS}/RackFrame/.*",
        ],
    )

    # Keep the ground below the table.  The table top is treated as z=0.
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, -0.9]),
        spawn=GroundPlaneCfg(),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


@configclass
class ActionsCfg:
    """Action specs; filled in by concrete env cfg."""

    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class ObservationsCfg:
    """Single policy observation group."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)  # 6 arm joints
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_ee_frame_ext)
        object_visible = ObsTerm(func=mdp.object_visible_obs)
        target_joint_error = ObsTerm(
            func=mdp.target_joint_error_obs,
            params={
                "target_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )
        actions = ObsTerm(func=mdp.last_action)  # 6 arm actions

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset and suction events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Start with small randomization around cube init position.
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.10, 0.10), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="box_aruco2"),
        },
    )

    reset_aruco_perception = EventTerm(
        func=mdp.reset_aruco_perception_state,
        mode="reset",
    )

    aruco_perception_update = EventTerm(
        func=mdp.update_aruco_perception_pose,
        mode="interval",
        interval_range_s=(0.05, 0.05),
    )

    reset_suction = EventTerm(
        func=mdp.reset_suction_state,
        mode="reset",
    )

    suction_attach = EventTerm(
        func=mdp.update_suction_attachment,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={
            "threshold": 0.1,
        },
    )

@configclass
class RewardsCfg:
    """Reward terms — Franka-lift recipe baseline."""

    reaching_object = RewTerm(
        func=mdp.gripper_close_near_object,
        params={"std": 0.2},
        weight=6.0,
    )

    suction_success = RewTerm(
        func=mdp.suction_attached_reward,
        weight=2.0,
    )

    joint_target_tracking = RewTerm(
        func=mdp.joint_position_target_reward,
        params={
            "target_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": 3.0,
            "asset_cfg": SceneEntityCfg("robot"),
            "minimal_height": 0.35,
        }, 
        weight=5.0,
    )

    joint_fine_target_tracking = RewTerm(
        func=mdp.joint_position_target_reward,
        params={
            "target_joint_pos": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "minimal_height": 0.35,
        }, 
        weight=10.0,
    )


    ee_low_penalty = RewTerm(
        func=mdp.ee_low_height_penalty,
        params={
            "min_height": 0.2,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=-3.0,
    )

    box_rack_collision = RewTerm(
        func=mdp.box_rack_collision_penalty,
        params={
            "threshold": 1.0,
            "sensor_cfg": SceneEntityCfg("box_rack_contact"),
        },
        weight=-8.0,
    )

    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2e-3)
    
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)

    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-3,
        # weight=-1e-3,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class TerminationsCfg:
    """Termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )

@configclass
class CurriculumCfg:
    """Curriculum terms."""

    action_rate = CurrTerm(
        func=mdp.modify_reward_weight,
        # params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 5_000_000},
        params={"term_name": "action_rate", "weight": -5e-3, "num_steps": 1_000},
    )

    joint_vel = CurrTerm(
        func=mdp.modify_reward_weight,
        # params={"term_name": "joint_vel", "weight": -3e-3, "num_steps": 5_000_000},
        params={"term_name": "joint_vel", "weight": -3e-3, "num_steps": 1_000},
    )

    reaching_object = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "reaching_object",
            "weight": 4.0,
            "num_steps": 1_000,
            # "num_steps": 5_000_000,
        },
    )

    suction_success = CurrTerm(
        func=mdp.modify_reward_weight,
        params={
            "term_name": "suction_success",
            "weight": 1.0,
            "num_steps": 1_000,
            # "num_steps": 5_000_000,
        },
    )

@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Abstract lift environment configuration for M0609."""

    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 3
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
