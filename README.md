# Rokey Collaboration 3

Chunsik Mars는 분산형 VLA 기반 로봇 에이전트 시스템입니다. 단일 PC에서 모든 추론과 제어를 처리하는 한계를 줄이기 위해 역할을 3대의 PC로 나누고, 자연어 명령, 비전 인식, 코드 생성, Isaac Sim 실행, 강화학습 정책을 연결합니다.

## 프로젝트 목표

- 자연어 기반 작업 명령을 로봇 실행 흐름으로 변환합니다.
- VLM/LLM 추론, 코드 생성, 강화학습, 시뮬레이션 제어를 분산 처리합니다.
- Isaac Sim / Isaac Lab 환경에서 Pick & Place 작업 구조를 검증합니다.
- TCP 기반 Isaac Sim Extension 구조로 Python 환경 충돌과 Stage 접근 문제를 해결합니다.

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
  - Isaac Sim 명령 전달
  - WebSocket 서버

PC3: Reinforcement Learning / Isaac Sim
  - Isaac Lab 기반 강화학습
  - Pick & Place policy 실행
  - Isaac Sim Extension TCP 서버
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
- TCP Socket
- USD Stage API
- TorchScript Policy

## 폴더 구조

```text
src/
  pc1/
    vla_brain_center.py            # PC1 Streamlit 작업 계획 UI
    AGENTS.md
    README_Chunsik_VLA_Brain.md

  pc2/
    code_llm_module8.py            # PC2 WebSocket / Code LLM 서버
    extension.py                   # Isaac Sim Extension 내부 실행 코드
    send_policy_command.py         # Extension TCP 서버로 명령 전송
    pick_module.py                 # policy 명령 래퍼
    move_controller_1.py           # ROS2 Nav2 / RL 이동 제어
    capture_module.py              # ROS2 이미지 캡처 모듈
    warehouse.yaml

  pc3/
    Spot reinforcement learning/
      train/
      usd/
      README.txt
```

## 시스템 실행 방법

### 1. 공통 환경 준비

각 PC 역할에 맞게 필요한 패키지를 설치합니다.

```bash
pip install streamlit websockets websocket-client streamlit-autorefresh
pip install ollama torch pillow transformers opencv-python numpy
```

ROS2/Isaac Sim을 사용하는 PC는 별도 환경 구성이 필요합니다.

```bash
source /opt/ros/humble/setup.bash
```

Isaac Sim / Isaac Lab은 로컬 설치 경로와 Python 버전이 중요합니다. `extension.py`는 Isaac Sim Extension으로 등록해 Isaac Sim 내부에서 실행하는 코드입니다.

### 2. PC1 작업 계획 UI 실행

PC1에서 Streamlit UI를 실행합니다.

```bash
cd src/pc1
streamlit run vla_brain_center.py
```

`vla_brain_center.py`는 PC3 또는 외부 비전 데이터 수신용 WebSocket 서버도 사용하며, 코드 기준 기본 수신 포트는 `8889`입니다.

### 3. PC2 Code LLM / Action 서버 실행

PC2에서 실행합니다.

```bash
cd src/pc2
python3 code_llm_module8.py
```

코드 기준 PC2 명령 서버는 `0.0.0.0:9999`에서 WebSocket 요청을 기다립니다. 동시에 PC1로 비전 데이터를 전달하는 루프도 함께 실행됩니다.

### 4. Isaac Sim Extension 실행

`src/pc2/extension.py`는 일반 터미널에서 단독 실행하는 스크립트가 아니라 Isaac Sim Extension 코드입니다. Isaac Sim에서 Extension으로 등록한 뒤 Stage가 열린 상태에서 활성화해야 합니다.

Extension이 정상 실행되면 내부 TCP 서버가 명령을 받을 수 있어야 합니다. `send_policy_command.py` 기준 접속 주소는 다음과 같습니다.

```text
HOST = 127.0.0.1
PORT = 8765
```

### 5. Pick & Place 명령 전송

Isaac Sim Extension TCP 서버가 켜진 상태에서 PC2 터미널에서 명령을 보냅니다.

```bash
cd src/pc2
python3 send_policy_command.py run_pick
python3 send_policy_command.py policy_stop
python3 send_policy_command.py run_place
```

코드에 표시된 기본 사용 안내는 `start|stop|reset|status`이지만, 프로젝트 흐름에서는 `run_pick`, `policy_stop`, `run_place` 명령이 Pick & Place 전환에 사용됩니다.

### 6. 강화학습 모듈 실행

PC3의 Spot reinforcement learning 폴더에서 학습 또는 재생 스크립트를 실행합니다.

```bash
cd "src/pc3/Spot reinforcement learning"
python3 train/train.py
python3 train/play.py
```

Isaac Lab 환경, USD asset 경로, policy 파일 경로는 로컬 설치 상태에 맞춰 확인해야 합니다.

## 실행 전 확인 사항

- `pc2/pick_module.py`에는 Isaac Lab 가상환경과 작업 경로가 절대 경로로 들어 있습니다. 로컬 경로에 맞게 수정해야 합니다.
- `pc2/move_controller_1.py`는 `move.pt` 정책 파일을 같은 폴더에서 찾습니다. 모델 파일을 준비하거나 경로를 수정해야 합니다.
- Isaac Sim Extension은 Isaac Sim 내부 Stage 접근 권한이 필요합니다.
- PC 간 WebSocket 주소와 포트는 실제 네트워크 IP에 맞게 수정해야 합니다.
- ROS2 DDS 통신, Isaac Sim 버전, Python 버전이 서로 맞지 않으면 PC별 가상환경을 분리해야 합니다.
