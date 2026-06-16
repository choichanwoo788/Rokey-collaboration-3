from isaaclab.assets import RigidObjectCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from . import mdp
from .doosan import DOOSAN_M0609_RG2_LIFT_CFG, EE_BODY_NAME
from .lift_env_cfg import LiftEnvCfg
from isaaclab.sensors import CameraCfg


BOX_USD_PATH = "/home/rokey/Downloads/pick_the_box/box_aruco3.usd"


@configclass
class M0609LiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = DOOSAN_M0609_RG2_LIFT_CFG.replace(
            prim_path="{ENV_REGEX_NS}/Robot"
        )

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint_[1-6]"],
            scale=0.5,
            use_default_offset=True,
        )

        self.actions.gripper_action = None

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=[-1.8, 0.0, 0.3],
                rot=[0.707, 0.0, 0.0, -0.707],
            ),
            spawn=UsdFileCfg(
                usd_path=BOX_USD_PATH,
                activate_contact_sensors=True,
                scale=(1.0, 1.0, 1.0),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"

        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robot_arm/Xform/m0609_suction_gripper/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/robot_arm/Xform/m0609_suction_gripper/{EE_BODY_NAME}",
                    name="end_effector",
                    offset=OffsetCfg(pos=[0.0, 0.0, 0.0]),
                ),
            ],
        )


        self.scene.wrist_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/robot_arm/Xform/m0609_suction_gripper/gripper_body/realsense_d455/RSD455/Camera_Pseudo_Depth",
            update_period=0.05,
            height=240,
            width=320,
            data_types=["rgb", "distance_to_image_plane"],
            spawn=None,
        )


@configclass
class M0609LiftEnvCfg_PLAY(M0609LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False