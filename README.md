# Rokey Collaboration 3

Chunsik Mars는 분산형 VLA 기반 로봇 에이전트 시스템입니다. 단일 PC에서 모든 추론과 제어를 처리하는 한계를 줄이기 위해 역할을 PC별로 나누고, 자연어 명령, 비전 인식, 코드 생성, Isaac Sim 실행, 강화학습 정책을 연결합니다.

현재 구조는 기존 `pc1`, `pc2`, `pc3` 소스에 더해 Isaac Lab 강화학습과 Isaac Sim Extension 실행 구조가 `rl_pick`, `rl_place`로 분리되어 있습니다.

## 프로젝트 목표

- 자연어 기반 작업 명령을 로봇 실행 흐름으로 변환합니다.
- VLM/LLM 추론, 코드 생성, 강화학습, 시뮬레이션 제어를 분산 처리합니다.
- Isaac Sim / Isaac Lab 환경에서 M0609 Pick & Place 작업을 검증합니다.
- TCP 기반 Isaac Sim Extension 구조로 Python 환경 충돌과 Stage 접근 문제를 해결합니다.
- Pick 정책과 Place 환경 설정을 분리해 학습/실행 구조를 명확히 관리합니다.

## 핵심 구조

```text
PC1: Cognitive Reasoning / Task Planning
  - Streamlit UI
  - 자연어 명령 해석
  - Task Packet 생성 및 검증
  - PC2/PC3와 WebSocket 통신

PC2: Perception / Action Generation
  - Vision AI
  - Code LLM
  - Action Generator
  - ROS2 / Isaac Sim 명령 전달
  - WebSocket 서버

RL / Isaac Sim:
  - rl_pick: M0609 lift/pick 학습, 재생, policy export, Isaac Sim Extension
  - rl_place: pre-attached carry-and-place 환경 설정과 reward/MDP 구성
```

## 기술 스택

- Python
- Streamlit
- WebSocket
- Ollama
- Florence-2
- DeepSeek-R1
- Qwen Coder
- ROS2 Humble / Jazzy
- Isaac Sim
- Isaac Lab
- RSL-RL
- Gymnasium
- TCP Socket
- USD Stage API
- TorchScript / ONNX Policy

## 폴더 구조

```text
src/
  pc1/
    vla_brain_center.py
    AGENTS.md
    README_Chunsik_VLA_Brain.md

  pc2/
    code_llm_module8.py
    capture_module.py
    move_controller_1.py
    pick_module.py
    warehouse.yaml

  pc3/
    Spot reinforcement learning/
      train/
      usd/
      README.txt

  rl_pick/
    train.py
    play.py
    cli_args.py
    send_policy_command.py
    m0609_lift/
      agents/
      mdp/
      cache/
      joint_pos_env_cfg.py
      lift_env_cfg.py
      doosan.py
    exts/
      cobot3.policy_suction/
        config/extension.toml
        cobot3_policy_suction/
          __init__.py
          extension.py
    logs/
      rsl_rl/my_test_m0609_lift/.../exported/policy.pt
      rsl_rl/my_test_m0609_lift/.../exported/policy.onnx

  rl_place/
    __init__.py
    doosan.py
    joint_pos_env_cfg.py
    lift_env_cfg.py
    mdp.py
    perception.py
    rewards.py
```

## 주요 실행 단위

### PC1 작업 계획 UI

`src/pc1/vla_brain_center.py`는 Streamlit 기반 작업 계획 UI입니다. 자연어 명령을 받고 Task Packet을 생성하며, 다른 PC와 WebSocket으로 통신합니다.

```bash
cd src/pc1
streamlit run vla_brain_center.py
```

코드 기준 PC1의 수신 WebSocket 포트는 `8889`입니다.

### PC2 Code LLM / Action 서버

`src/pc2/code_llm_module8.py`는 PC2의 WebSocket 서버입니다. 작업 명령을 받아 계획을 생성하고, move/pick/place 계열 작업 실행 흐름을 관리합니다.

```bash
cd src/pc2
python3 code_llm_module8.py
```

코드 기준 PC2 명령 서버는 `0.0.0.0:9999`에서 요청을 받습니다.

### RL Pick 학습

`src/rl_pick/train.py`는 Isaac Lab + RSL-RL 기반 M0609 lift/pick 학습 스크립트입니다. `m0609_lift/__init__.py`에서 다음 task id를 등록합니다.

```text
My_Isaac-M0609-v0
My_Isaac-M0609-Play-v0
```

실행 예시:

```bash
cd src/rl_pick
python train.py --task My_Isaac-M0609-v0 --num_envs 4096 --max_iterations 1500
```

환경에 따라 Isaac Lab 실행 래퍼를 통해 실행해야 할 수 있습니다.

```bash
./isaaclab.sh -p src/rl_pick/train.py --task My_Isaac-M0609-v0 --num_envs 4096
```

### RL Pick 정책 재생 및 export

`src/rl_pick/play.py`는 학습된 checkpoint를 재생하고 `policy.pt`, `policy.onnx`를 export합니다.

```bash
cd src/rl_pick
python play.py --task My_Isaac-M0609-Play-v0 --checkpoint logs/rsl_rl/my_test_m0609_lift/<run>/model_*.pt
```

export 결과는 기본적으로 해당 run의 `exported/` 폴더에 저장됩니다.

```text
logs/rsl_rl/my_test_m0609_lift/<run>/exported/policy.pt
logs/rsl_rl/my_test_m0609_lift/<run>/exported/policy.onnx
```

### Isaac Sim Extension 실행

현재 Isaac Sim Extension은 `src/rl_pick/exts/cobot3.policy_suction` 아래에 있습니다.

```text
src/rl_pick/exts/cobot3.policy_suction/config/extension.toml
src/rl_pick/exts/cobot3.policy_suction/cobot3_policy_suction/extension.py
```

Isaac Sim에서 Extension Search Path에 아래 경로를 추가한 뒤 `COBOT3 M0609 POLICY SUCTION` Extension을 활성화합니다.

```text
src/rl_pick/exts
```

이 Extension은 열린 Isaac Sim GUI Stage에 접근해 policy 실행, suction attach/detach, object pose 갱신, TCP 명령 처리를 담당합니다.

### Pick & Place TCP 명령 전송

Extension이 켜져 있고 TCP 서버가 준비된 상태에서 `src/rl_pick/send_policy_command.py`로 명령을 보냅니다. 코드 기준 접속 주소는 `127.0.0.1:8765`입니다.

```bash
cd src/rl_pick
python send_policy_command.py run_pick
python send_policy_command.py policy_stop
python send_policy_command.py run_place
```

기본 usage 문구에는 `start|stop|reset|status`가 표시되지만, 프로젝트 Pick & Place 흐름에서는 `run_pick`, `policy_stop`, `run_place` 명령도 사용됩니다.

### RL Place 환경 설정

`src/rl_place`는 pre-attached carry-and-place 환경 설정, reward, MDP term을 담고 있습니다. `__init__.py`는 다음 task id를 등록합니다.

```text
Isaac-M0609-Lift-v0
```

주요 파일:

```text
rl_place/lift_env_cfg.py       # table/world/robot USD, command, observation, reward 구성
rl_place/joint_pos_env_cfg.py  # M0609 joint-position 환경 구성
rl_place/rewards.py            # stable placement, release, drop penalty 등 custom reward
rl_place/perception.py         # perception 관련 term
rl_place/doosan.py             # M0609/RG2 articulation config
```

`lift_env_cfg.py` 안에는 로컬 절대 경로가 포함되어 있으므로 실행 전 반드시 환경에 맞게 수정해야 합니다.

```text
MY_WORLD_BG_USD = C:/Users/milkc/OneDrive/Desktop/my_scene_layout.usd
MY_ROBOT_USD = /home/rokey/dev_ws/issac_sim/assets/doosan_m0609.usd
```

## 시스템 실행 순서 예시

전체 시나리오를 나눠 실행할 때의 기본 순서는 다음과 같습니다.

```bash
# 1. PC1 작업 계획 UI
cd src/pc1
streamlit run vla_brain_center.py

# 2. PC2 Code LLM / Action 서버
cd src/pc2
python3 code_llm_module8.py

# 3. Isaac Sim에서 Extension 경로 등록 및 활성화
# Extension path: src/rl_pick/exts

# 4. Extension TCP 서버에 policy 명령 전송
cd src/rl_pick
python send_policy_command.py run_pick
python send_policy_command.py policy_stop
python send_policy_command.py run_place
```

학습 또는 policy export가 필요한 경우에는 먼저 `rl_pick/train.py`, `rl_pick/play.py`를 실행해 policy 파일을 준비합니다.

## 실행 전 확인 사항

- Isaac Sim, Isaac Lab, RSL-RL 버전이 현재 Python 환경과 맞아야 합니다.
- `rl_pick/play.py`가 export한 `policy.pt`를 Extension이 참조하는 경로에 맞게 배치해야 합니다.
- `rl_place/lift_env_cfg.py`의 `MY_WORLD_BG_USD`, `MY_ROBOT_USD`는 로컬 PC 경로에 맞게 수정해야 합니다.
- PC 간 WebSocket 주소와 포트는 실제 네트워크 IP에 맞게 수정해야 합니다.
- `pc2/pick_module.py`는 기존 경로 구조를 참조할 수 있으므로, 통합 실행 시 `send_policy_command.py` 위치가 현재 `src/rl_pick` 구조와 맞는지 확인해야 합니다.
- `__pycache__`, 학습 로그, checkpoint 파일은 GitHub 업로드 시 필요에 따라 `.gitignore` 또는 Git LFS/Release로 관리하는 것이 좋습니다.
