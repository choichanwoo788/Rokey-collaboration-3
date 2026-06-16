from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg


HOME_POSE: tuple[float, ...] = (3.14159, 0.0, 1.5708, 0.0, 0.0, -1.5708)

EE_BODY_NAME = "suction_tcp"

ROBOT_USD_PATH = "/home/rokey/Downloads/pick_the_box/robot_arm.usd"

ROBOT_BASE_POS: tuple[float, float, float] = (-0.6, 0.0, 0.0)


DOOSAN_M0609_RG2_LIFT_CFG = ArticulationCfg(
    spawn=UsdFileCfg(
        usd_path=ROBOT_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            fix_root_link=True,
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=1,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=ROBOT_BASE_POS,
        rot=(1.0, 0.0, 0.0, 0.0),
        joint_pos={
            "joint_1": HOME_POSE[0],
            "joint_2": HOME_POSE[1],
            "joint_3": HOME_POSE[2],
            "joint_4": HOME_POSE[3],
            "joint_5": HOME_POSE[4],
            "joint_6": HOME_POSE[5],
        },
    ),
    actuators={
        "m0609_arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit_sim=9600.0,
            velocity_limit_sim=2.618,
            stiffness=3000.0,
            damping=200.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)