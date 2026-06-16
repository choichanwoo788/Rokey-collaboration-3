import gymnasium as gym

from .joint_pos_env_cfg import M0609LiftEnvCfg
from .lift_env_cfg import LiftEnvCfg

gym.register(
    id="Isaac-M0609-Lift-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:M0609LiftEnvCfg",
        "skrl_cfg_entry_point": "isaaclab_tasks.manager_based.manipulation.lift.config.franka.agents:skrl_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{__name__}.agents.rsl_rl_cfg:LiftPPORunnerCfg",
    },
)