# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

"""Abstract LiftEnvCfg for the M0609 suction lift task customized for Junho."""

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

from . import mdp

# ======================================================================
# 🔥 준호님의 PC 환경 절대 경로에 맞게 수정하세요!
# ======================================================================
MY_WORLD_BG_USD = r"C:/Users/milkc/OneDrive/Desktop/my_scene_layout.usd"
MY_ROBOT_USD = "/home/rokey/dev_ws/issac_sim/assets/doosan_m0609.usd"

@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    robot: ArticulationCfg = MISSING
    ee_frame: FrameTransformerCfg = MISSING
    object: RigidObjectCfg = MISSING

    # 💡 준호님이 만든 [정사각형 판 + 직육면체 받침대]가 포함된 단일 배경 무대 USD 로드
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.9, 0.0, 0.0],  # 로봇 팔 고정 중심점에 오도록 배경 위치 연동
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=UsdFileCfg(usd_path=MY_WORLD_BG_USD),
    )

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
class CommandsCfg:
    """상자를 최종적으로 배달하여 내려놓을 직육면체 받침대 영역 범위 타겟 지정"""

    object_pose = mdp.UniformPoseCommandCfg(
        asset_name="robot",
        body_name=MISSING,
        resampling_time_range=(6.0, 6.0),
        debug_vis=True,
        ranges=mdp.UniformPoseCommandCfg.Ranges(
            pos_x=(-0.200, 0.200),
            pos_y=(-1.050, -0.850),
            pos_z=(0.030, 0.070),
            roll=(0.0, 0.0),
            pitch=(0.0, 0.0),
            yaw=(0.0, 0.0),
        ),
    )


@configclass
class ActionsCfg:
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg | None = None


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        # 💡 실시간 아르코 버퍼 포인트를 활용하는 관측 축 유지
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame_ext)
        target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_suction = EventTerm(func=mdp.reset_suction_state, mode="reset")
    reset_object_attached = EventTerm(
        func=mdp.attach_object_to_suction_tcp,
        mode="reset",
        params={"attach_offset_pos": (0.0, 0.0, -0.075)},
    )

    # Pre-attached task: no distance-threshold pickup.  While attached, the box follows suction_tcp.
    suction_follow = EventTerm(
        func=mdp.update_suction_attachment,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"attach_offset_pos": (0.0, 0.0, -0.075)},
    )
    suction_release = EventTerm(
        func=mdp.automatic_release_at_target,
        mode="interval",
        interval_range_s=(0.01, 0.01),
        params={"command_name": "object_pose", "xy_threshold": 0.04, "z_threshold": 0.05},
    )


@configclass
class RewardsCfg:
    keep_attached = RewTerm(func=mdp.keep_object_attached, weight=0.5)
    object_goal_tracking = RewTerm(
        func=mdp.attached_object_goal_distance,
        params={"std": 0.30, "command_name": "object_pose"},
        weight=8.0,
    )
    object_xy_alignment = RewTerm(
        func=mdp.attached_object_xy_alignment,
        params={"std": 0.06, "command_name": "object_pose"},
        weight=3.0,
    )
    release_near_target = RewTerm(
        func=mdp.release_near_target,
        params={"command_name": "object_pose", "xy_threshold": 0.05, "z_threshold": 0.06},
        weight=5.0,
    )
    stable_placement = RewTerm(
        func=mdp.stable_placement,
        params={"command_name": "object_pose", "xy_threshold": 0.05, "z_threshold": 0.06},
        weight=15.0,
    )
    early_drop_penalty = RewTerm(
        func=mdp.early_drop_penalty,
        params={"command_name": "object_pose", "xy_threshold": 0.08},
        weight=-8.0,
    )
    dropping_penalty = RewTerm(func=mdp.is_terminated_term, params={"term_keys": "object_dropping"}, weight=-5.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
    joint_vel = RewTerm(func=mdp.joint_vel_l2, weight=-1e-4, params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=mdp.object_stable_at_target,
        params={"command_name": "object_pose", "xy_threshold": 0.05, "z_threshold": 0.06},
    )
    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")},
    )


@configclass
class CurriculumCfg:
    action_rate = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-2, "num_steps": 50_000_000})
    joint_vel = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-2, "num_steps": 50_000_000})
    early_drop_penalty = CurrTerm(func=mdp.modify_reward_weight, params={"term_name": "early_drop_penalty", "weight": -12.0, "num_steps": 30_000_000})


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        self.decimation = 2
        self.episode_length_s = 5.0
        self.sim.dt = 0.01
        self.sim.render_interval = self.decimation
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625