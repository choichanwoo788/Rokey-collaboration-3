from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils

# 손끝 수션 링크 이름 (준호님 URDF/USD 조인트 이름에 맞춰 매핑)
EE_BODY_NAME = "suction_tcp" 

DOOSAN_M0609_RG2_LIFT_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=r"C:\Users\milkc\OneDrive\Desktop\Collected_robot_arm\Collected_robot_arm\robot_arm.usd",
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(-0.9, 2.0, 0.8),
        rot=(1.0, 0.0, 0.0, 0.0),
        # 로봇이 처음 스폰되었을 때 취할 기본 관절 각도 (라디안 단위)
        joint_pos={
            "joint_1": 0.0,
            "joint_2": 0.0,
            "joint_3": 1.57,
            "joint_4": 0.0,
            "joint_5": 1.57,
            "joint_6": 0.0,
        },
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint_[1-6]"],
            effort_limit_sim=100.0,
            velocity_limit_sim=2.0,
            stiffness=400.0,
            damping=40.0,
        ),
    },
)