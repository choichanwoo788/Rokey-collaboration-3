from __future__ import annotations

import torch
from isaaclab.utils.math import quat_apply, quat_mul

from .perception import get_box_pose_w


def _get_num_envs(env) -> int:
    # env 객체 종류에 따라 num_envs 접근 방식이 다를 수 있어 공통 처리
    if hasattr(env, "num_envs"):
        return env.num_envs
    return env.scene.num_envs


def _get_device(env):
    # tensor를 생성할 device를 env 기준으로 통일
    if hasattr(env, "device"):
        return env.device
    return env.scene.device


def _quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    # quaternion inverse 계산용.
    # 단위 quaternion 기준 conjugate = inverse.
    q_conj = q.clone()
    q_conj[..., 1:] *= -1.0
    return q_conj


def _ensure_suction_buffers(env):
    # suction 상태를 env에 buffer로 저장한다.
    # Isaac Lab Manager 함수들은 class member를 직접 추가해서 상태를 유지할 수 있다.
    num_envs = _get_num_envs(env)
    device = _get_device(env)

    # 현재 env별 흡착 여부
    if not hasattr(env, "suction_attached"):
        env.suction_attached = torch.zeros(
            num_envs,
            dtype=torch.bool,
            device=device,
        )

    # 흡착 순간의 object 회전을 EE 기준 상대 회전으로 저장
    if not hasattr(env, "suction_rel_quat_ee"):
        env.suction_rel_quat_ee = torch.zeros(
            num_envs,
            4,
            dtype=torch.float32,
            device=device,
        )
        env.suction_rel_quat_ee[:, 0] = 1.0

    # 흡착 순간의 marker 중심 -> object root 벡터를 EE 좌표계 기준으로 저장
    if not hasattr(env, "suction_root_from_marker_ee"):
        env.suction_root_from_marker_ee = torch.zeros(
            num_envs,
            3,
            dtype=torch.float32,
            device=device,
        )


def reset_suction_state(env, env_ids: torch.Tensor | None):
    # episode reset 시 suction 상태 초기화.
    # 이전 episode의 attached 상태가 다음 episode로 넘어가지 않게 한다.
    _ensure_suction_buffers(env)

    if env_ids is None:
        env.suction_attached[:] = False

        env.suction_rel_quat_ee[:] = 0.0
        env.suction_rel_quat_ee[:, 0] = 1.0

        env.suction_root_from_marker_ee[:] = 0.0

    else:
        env.suction_attached[env_ids] = False

        env.suction_rel_quat_ee[env_ids] = 0.0
        env.suction_rel_quat_ee[env_ids, 0] = 1.0

        env.suction_root_from_marker_ee[env_ids] = 0.0


def update_suction_attachment(
    env,
    env_ids: torch.Tensor | None,
    threshold: float = 0.035,
):
    """
    ArUco marker 중심 기준 suction attachment 처리.

    핵심 흐름:
    1. suction_tcp와 marker 중심 거리 계산
    2. threshold 이내면 suction_attached=True
    3. 흡착 순간의 object 상대 pose 저장
    4. 이후 매 step marker 중심이 suction_tcp에 오도록 object root 갱신
    """

    _ensure_suction_buffers(env)

    obj = env.scene["object"]
    ee_frame = env.scene["ee_frame"]

    num_envs = _get_num_envs(env)
    device = _get_device(env)

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device)

    # suction_tcp의 world pose
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    # 현재 target marker world pose.
    # 검출 전에는 GT fallback, 검출 후에는 perception 좌표를 사용.
    marker_pos_w, _ = get_box_pose_w(env)

    ee_pos_w = torch.nan_to_num(
        ee_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    marker_pos_w = torch.nan_to_num(
        marker_pos_w,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    # suction_tcp와 marker 중심 사이 거리
    dist = torch.norm(
        ee_pos_w[env_ids] - marker_pos_w[env_ids],
        dim=-1,
    )

    dist = torch.nan_to_num(
        dist,
        nan=999.0,
        posinf=999.0,
        neginf=999.0,
    )

    # threshold 이내로 들어온 env만 새 흡착 후보
    newly_attached_local = dist < threshold
    newly_attached_env_ids = env_ids[newly_attached_local]

    # 이미 attached인 env는 다시 초기화하지 않음
    if newly_attached_env_ids.numel() > 0:
        newly_attached_env_ids = newly_attached_env_ids[
            ~env.suction_attached[newly_attached_env_ids]
        ]

    if newly_attached_env_ids.numel() > 0:
        env.suction_attached[newly_attached_env_ids] = True

        obj_pos_w = obj.data.root_pos_w
        obj_quat_w = obj.data.root_quat_w

        ee_quat_inv = _quat_conjugate(ee_quat_w[newly_attached_env_ids])

        # 흡착 순간 object 회전을 EE 기준 상대 회전으로 저장
        env.suction_rel_quat_ee[newly_attached_env_ids] = quat_mul(
            ee_quat_inv,
            obj_quat_w[newly_attached_env_ids],
        )

        # marker 중심에서 object root까지의 world 벡터
        root_from_marker_w = (
            obj_pos_w[newly_attached_env_ids]
            - marker_pos_w[newly_attached_env_ids]
        )

        # 위 벡터를 EE local frame 기준으로 저장.
        # 이후 EE가 움직여도 같은 상대 위치를 유지하기 위함.
        env.suction_root_from_marker_ee[newly_attached_env_ids] = quat_apply(
            ee_quat_inv,
            root_from_marker_w,
        )

    # 현재 attached 상태인 env들만 object root를 강제로 따라오게 갱신
    attached_ids = env.suction_attached.nonzero(as_tuple=False).squeeze(-1)

    if attached_ids.numel() == 0:
        return

    new_root_state = obj.data.root_state_w.clone()

    # 저장된 marker->root 벡터를 현재 EE 회전 기준 world frame으로 변환
    root_from_marker_w = quat_apply(
        ee_quat_w[attached_ids],
        env.suction_root_from_marker_ee[attached_ids],
    )

    # marker 중심이 suction_tcp 위치에 오도록 object root 위치 계산
    new_obj_pos_w = ee_pos_w[attached_ids] + root_from_marker_w

    new_root_state[attached_ids, 0:3] = new_obj_pos_w

    # 흡착 순간 저장한 상대 회전을 현재 EE 회전에 합성해서 object 회전 갱신
    new_root_state[attached_ids, 3:7] = quat_mul(
        ee_quat_w[attached_ids],
        env.suction_rel_quat_ee[attached_ids],
    )

    # 흡착 중에는 object 속도를 0으로 만들어 물리적으로 튀는 현상 완화
    new_root_state[attached_ids, 7:13] = 0.0

    # 계산한 root state를 시뮬레이터에 적용
    obj.write_root_state_to_sim(
        new_root_state[attached_ids],
        env_ids=attached_ids,
    )