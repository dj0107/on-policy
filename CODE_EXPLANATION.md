# Nalpari 코드 전체 설명

이 문서는 프로젝트의 모든 핵심 파일을 한 줄씩 상세하게 설명합니다.  
논문: 멀티 UAV 협동 다중 표적 추적 (Dec-POMDP + AAI + MAPPO)

---

## 목차

1. [uav_env.py — 핵심 환경](#1-uav_envpy--핵심-환경)
2. [uav_vec_env.py — 벡터화 래퍼](#2-uav_vec_envpy--벡터화-래퍼)
3. [uav_runner.py — 학습 루프](#3-uav_runnerpy--학습-루프)
4. [train_uav.py — 학습 진입점](#4-train_uavpy--학습-진입점)
5. [llm_aai.py — LLM 기반 AAI](#5-llm_aaipy--llm-기반-aai)
6. [run_evaluation.py — 에피소드 실행기](#6-run_evaluationpy--에피소드-실행기)
7. [evaluate_trained.py — 평가 CLI](#7-evaluate_trainedpy--평가-cli)
8. [test_llm_call.py — LLM 단독 테스트](#8-test_llm_callpy--llm-단독-테스트)

---

# 1. uav_env.py — 핵심 환경

**역할**: 논문의 Dec-POMDP 환경 전체를 구현. UAV 이동, 레이더 탐지, EKF 추적, 
보상 함수까지 모든 물리 모델이 여기에 있음.

---

## 1-1. 파일 헤더 주석 (1~18줄)

```
# ============================================================================
# (12) 버전 - Dec-POMDP, AAI 출력: epsilon_kt, W_kt, p_ut
```

"(12)"는 논문 버전 번호. 파일이 어떤 논문 버전을 구현하는지 명시.

```
#   - SNR 단위 명시 (dB ↔ linear)
```
센서 SNR을 dB와 선형(linear) 단위로 혼동하지 않도록 명시적으로 변환.

```
#   - 측정 잡음 분산에 floor (수치 안정성)
```
EKF의 잡음 분산이 0에 너무 가까워지면 역행렬 계산이 불안정해지므로 최솟값(floor)을 둠.

```
#   - 보상 스케일 재조정 (충돌이 학습 신호 지배 방지)
```
초기 버전에서 충돌 페널티가 너무 커서 에이전트가 충돌 회피만 학습하는 문제를 수정.

```
#   - 타겟/UAV 위치 클리핑 (식 28b/28c 강제)
```
논문 식 28b/28c: UAV와 타겟이 맵 경계를 벗어나지 않도록 강제.

```
#   - alpha=0 시 e_comm 절감 (absence signal 모델)
```
탐지 실패(alpha=0)시에도 "탐지 못 했다"는 신호를 BS에 보내지만, 
이때는 전력을 1%만 사용하도록 모델링.

```
#   - 식 (28f) 미탐지 페널티 추가
```
논문 식 28f: 어떤 UAV도 특정 타겟을 탐지하지 못하면 추가 페널티.

```
#   - 학습 시 AAI 파라미터 randomization (domain randomization)
```
학습 중 W_kt, epsilon_kt, p_ut를 무작위로 바꿔가며 MAPPO가 
다양한 AAI 출력에 적응하도록 함.

```
#   - 평가 시 외부 AAI(LLM) 주입 hook
```
평가 시 llm_aai.py의 callback 함수를 받아 LLM이 W/eps/p를 결정.

```
#   - 시각화/로깅용 _episode_log 누적
```
log_episode=True로 켜면 매 스텝 데이터를 저장하여 나중에 plot_episode.py로 시각화 가능.

---

## 1-2. Target 클래스 (21~54줄)

```python
class Target:
    """타겟 상태, EKF용 fused estimate, BFIM 관리"""
```

타겟 하나를 나타내는 클래스. 물리적 위치/속도 외에 **추적 상태(EKF)**와 
**정보행렬(BFIM)**도 함께 관리.

```python
    def __init__(self, target_id, initial_pos, initial_velocity):
        self.id = target_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array(initial_velocity, dtype=np.float32)
```
실제(ground truth) 위치와 속도. 환경에서만 알고 있고 UAV는 모름.

```python
        self.S_global = np.concatenate([self.pos, self.velocity])
```
BS(기지국)가 융합한 전역 추정 상태 벡터 [x, y, vx, vy]. 
처음엔 true 값과 동일하지만 EKF 업데이트를 거치면서 추정값이 됨.

```python
        self.P_global = np.eye(4, dtype=np.float32) * 1.0
```
전역 추정의 공분산(불확실도). 4×4 행렬. 초기값 I (표준편차 1 단위).

```python
        self.J_matrix = np.eye(4, dtype=np.float32) * 0.1
```
**베이즈 피셔 정보 행렬(BFIM, Bayesian Fisher Information Matrix)**.  
J의 역행렬이 PCRLB(추적 오차 하한)를 줌. 초기값 0.1·I.

```python
        self.F_kt = 0.0
```
현재 스텝의 PCRLB 추적 오차. 논문 식 F_k,t = tr(Λ · J^{-1} · Λ^T).  
값이 클수록 추적이 부정확.

```python
        self.measurement_info = np.zeros((4, 4), dtype=np.float32)
```
이번 스텝에서 이 타겟을 탐지한 모든 UAV의 측정 정보 합산.  
매 스텝 초기화 후 탐지마다 누적.

```python
        self.epsilon_kt = 5.0
        self.W_kt = 1.0
```
AAI가 설정하는 값들.  
- epsilon_kt: 추적 정확도 임계값 (관측 내에 있으면 OK 기준)  
- W_kt: 이 타겟의 우선순위 가중치 (보상 곱셈 인자)

```python
        self.d_Z_kt = float('inf')
```
이 타겟이 가장 가까운 위험구역(critical zone)까지의 거리. 초기값 무한대.

---

### Target.step() (36~54줄)

```python
    def step(self, dt, Q=None, map_min=-500.0, map_max=500.0):
        """식 (1) + 경계 반사"""
```
논문 식 (1): 타겟 운동 모델. 등속 직선 운동 + 프로세스 잡음.

```python
        self.pos += self.velocity * dt
```
위치 업데이트: x_{t+1} = x_t + v·dt

```python
        if Q is not None:
            w_kt = np.random.multivariate_normal(np.zeros(4), Q).astype(np.float32)
            self.pos += w_kt[:2]
            self.velocity += w_kt[2:]
```
프로세스 잡음 w ~ N(0, Q) 추가. Q는 sigma_w_sq 파라미터로 조절.  
[:2]는 위치 잡음, [2:]는 속도 잡음.

```python
        for d in range(2):
            if self.pos[d] < map_min:
                self.pos[d] = map_min
                self.velocity[d] = abs(self.velocity[d])
            elif self.pos[d] > map_max:
                self.pos[d] = map_max
                self.velocity[d] = -abs(self.velocity[d])
```
맵 경계(±500m)에 부딪히면 반사(반발). 경계에 고정 + 속도 방향 반전.

```python
        v_norm = np.linalg.norm(self.velocity)
        v_max_target = 8.0
        if v_norm > v_max_target:
            self.velocity = self.velocity / v_norm * v_max_target
```
타겟 최대 속도 8m/s 제한. 잡음으로 인해 속도가 너무 빨라지는 것 방지.

---

## 1-3. UAV 클래스 (57~145줄)

```python
class UAV:
    """UAV 행동/에너지. AAI가 p_ut 외부 주입"""
```
UAV 하나를 나타내는 클래스. 이동, 에너지 소모, 레이더/통신 모델 포함.

### __init__ (59~90줄)

```python
        self.H = altitude     # 고도 (90m로 고정)
        self.v_max = max_speed  # 최대 속도 (10m/s)
        self.energy = max_energy  # 현재 에너지 (초기 100,000 J)
        self.max_energy = max_energy
```

```python
        self.p_ut = 1.0       # 기본 송신 전력 (W) — AAI가 매 스텝 재설정
        self.tau_s_ratio = 0.5  # 센싱 시간 비율 (전체 dt 중 50%)
```
AAI가 p_ut를 조절. tau_s_ratio로 센싱 시간 τ_s = 0.5·dt.

```python
        self.is_detected_per_target = {}
```
이번 스텝에서 각 타겟을 탐지했는지 저장: {target_id: 0 또는 1}

```python
        self.last_e_tot = 0.0   # 이번 스텝 총 에너지
        self.last_e_sen = 0.0   # 센싱 에너지
        self.last_e_comm = 0.0  # 통신 에너지
        self.last_e_move = 0.0  # 이동 에너지
```
보상 계산 및 로깅용 이전 스텝 에너지 기록.

```python
        self.f_c = 2.4e9         # 반송 주파수 2.4 GHz
        self.lam = 3e8 / self.f_c  # 파장 λ = c/f = 0.125m
        self.G_t = 1.0           # 송신 안테나 이득 (수정됨: 원래 100)
        self.G_r = 30.0          # 수신 안테나 이득 (수정됨: 원래 1000)
        self.sigma = 1.0         # 레이더 반사 단면적 (RCS)
        self.N0 = 1e-14          # 잡음 전력
```
레이더 방정식 파라미터. G_t·G_r = 30으로 최대 탐지 거리 ~307m.

```python
        self.c1, self.c2 = 11.9, 0.13
        self.eta_LoS, self.eta_NLoS = 1.0, 100.0
        self.B_b = 1e6           # 통신 대역폭 1 MHz
```
통신 채널 모델(공대지 통신, LoS/NLoS 확률) 파라미터.

```python
        self.P0, self.P1 = 3.4, 20.0    # 프로파일 드래그/유도 전력 계수
        self.u_tip, self.v0 = 60.0, 5.4  # 블레이드 팁 속도, 유도 속도
        self.d0, self.rho, self.s0, self.A = 0.3, 1.225, 0.03, 0.28
```
멀티로터 드론 이동 전력 모델 파라미터.

---

### UAV.apply_action() (92~109줄)

```python
    def apply_action(self, action, dt, map_min=-500.0, map_max=500.0):
        delta_q = np.array(action, dtype=np.float32) * self.v_max * dt
```
MAPPO가 출력하는 action은 [-1, 1]^2 범위. v_max·dt를 곱해 실제 이동 거리로 변환.  
dt=1s, v_max=10m/s이면 최대 10m 이동.

```python
        delta_norm = np.linalg.norm(delta_q)
        if delta_norm > self.v_max * dt:
            delta_q = (delta_q / delta_norm) * self.v_max * dt
```
대각선 이동 시 속도 초과 방지. 방향 유지한 채 크기만 제한.

```python
        boundary_hit = False
        for d in range(2):
            if new_pos[d] < map_min:
                new_pos[d] = map_min
                boundary_hit = True
```
맵 경계 강제. 경계에 부딪히면 boundary_hit=True 반환 → 보상에서 페널티.

```python
        actual_delta = new_pos - self.pos
        self.velocity = actual_delta / dt if dt > 0 else np.zeros(2)
```
실제 이동한 거리 기반으로 속도 계산. 경계에서 멈췄으면 속도가 줄어듦.

---

### UAV.get_sensing_model() (111~117줄)

```python
    def get_sensing_model(self, target_pos, dt):
        d_uk = math.sqrt(np.sum((self.pos - target_pos)**2) + self.H**2)
```
3D 거리 = sqrt(2D 수평 거리² + 고도²). UAV는 항상 H=100m에 있음.

```python
        tau_s = self.tau_s_ratio * dt
        snr_sen = (self.p_ut * self.G_t * self.G_r * (self.lam**2) * self.sigma * tau_s) \
                  / (self.N0 * (4 * math.pi)**3 * (d_uk**4))
```
**레이더 방정식 (Radar Range Equation)**:
- 분자: 송신 전력 × 안테나 이득들 × 파장² × RCS × 센싱 시간
- 분모: 잡음 × (4π)³ × 거리⁴  
거리의 4제곱에 반비례하는 것이 레이더 특성 (전파→타겟→반사→수신 두 번 전파).

```python
        e_sen = self.p_ut * tau_s
```
센싱 에너지 = 전력 × 시간. (단순 선형)

---

### UAV.get_comm_model() (119~135줄)

```python
    def get_comm_model(self, bs_pos, dt, alpha_ukt):
        d_ut_2d = np.linalg.norm(self.pos - bs_pos)
        d_ut_3d = math.sqrt(d_ut_2d**2 + self.H**2)
        theta_ut = math.atan(self.H / d_ut_2d) if d_ut_2d > 0 else math.pi/2
```
BS(기지국)까지의 3D 거리와 고도각(elevation angle) θ 계산.

```python
        p_los = 1 / (1 + self.c1 * math.exp(-self.c2 * (math.degrees(theta_ut) - self.c1)))
```
**LoS(가시선) 확률** 계산. 고도각이 높을수록(UAV가 BS 바로 위에 있을수록) LoS 확률↑.  
공대지(Air-to-Ground) 채널 모델에서 널리 쓰이는 sigmoid 함수 형태.

```python
        h_c = ((p_los * self.eta_LoS + p_nlos * self.eta_NLoS) * (self.lam**2)) \
              / ((4 * math.pi * d_ut_3d)**2 + 1e-12)
```
**평균 경로 손실 채널 이득** h_c.  
eta_LoS=1 (LoS는 손실 작음), eta_NLoS=100 (NLoS는 손실 100배 큼).  
1e-12는 거리=0일 때 분모가 0이 되는 것 방지.

```python
        snr_comm = (self.p_ut * h_c) / self.N0
        tau_c = (1 - self.tau_s_ratio) * dt
        R_c_ut = alpha_ukt * tau_c * self.B_b * math.log2(1 + snr_comm)
```
**Shannon 채널 용량** (bit/s). alpha=1일 때만(탐지 성공) 실제 데이터 전송.  
R_c_ut는 이 스텝에서 실제 전송한 비트 수.

```python
        if alpha_ukt == 1:
            e_comm = self.p_ut * tau_c       # 탐지 성공: 전력 풀로 사용
        else:
            e_comm = self.p_ut * tau_c * 0.01  # 탐지 실패: 1%만 사용 (absence signal)
```
탐지 실패 시에도 "탐지 못 했다"는 짧은 신호를 BS에 보내야 하므로 1%만 소비.

---

### UAV.get_mobility_energy() (137~145줄)

```python
    def get_mobility_energy(self, dt):
        v_norm = np.linalg.norm(self.velocity)
        if v_norm == 0:
            return (self.P0 + self.P1) * dt
```
정지(hover) 시에도 로터가 돌아야 하므로 기본 전력 (P0+P1)·dt 소비.

```python
        term1 = self.P0 * (1 + (3 * v_norm**2) / (self.u_tip**2))
```
**블레이드 프로파일 전력**: 속도가 빠를수록 비례하여 증가.

```python
        inner_sqrt = math.sqrt(1 + (v_norm**4) / (4 * self.v0**4))
        term2 = self.P1 * math.sqrt(max(inner_sqrt - (v_norm**2) / (2 * self.v0**2), 0))
```
**유도 전력(induced power)**: 빠를수록 줄어드는 항 (로터가 더 효율적으로 작동).  
max(..., 0)은 음수가 되는 것 방지.

```python
        term3 = 0.5 * self.d0 * self.rho * self.s0 * self.A * (v_norm**3)
```
**항력(parasitic drag) 전력**: 속도의 세제곱에 비례. 고속에서 지배적.

---

## 1-4. UAVTrackingEnv 클래스 — 메인 환경 (148~604줄)

### __init__() (151~218줄)

```python
    def __init__(self, num_uavs=5, num_targets=2, dt=1.0,
                 use_aai=True, aai_callback=None,
                 randomize_aai=False,
                 sigma_w_sq=5.0, log_episode=False,
                 num_critical_zones=2):
```
주요 파라미터:
- `num_uavs`: UAV 수 (기본 5)
- `num_targets`: 타겟 수 (기본 2)
- `dt=1.0`: 시간 스텝 (1초)
- `use_aai=True`: Heuristic AAI 사용 여부
- `aai_callback=None`: 외부(LLM) AAI 콜백. 있으면 heuristic 무시
- `randomize_aai=False`: 학습 시 True → domain randomization
- `sigma_w_sq=5.0`: 타겟 프로세스 잡음 분산
- `log_episode=False`: 시각화용 로그 저장 여부

```python
        all_zones = [
            np.array([200.0, 200.0], dtype=np.float32),
            np.array([-150.0, 300.0], dtype=np.float32),
            np.array([300.0, -250.0], dtype=np.float32),
            np.array([-250.0, -200.0], dtype=np.float32),
        ]
        self.critical_zones = all_zones[:num_critical_zones]
```
위험구역 4개 정의. num_critical_zones=2면 앞 2개만 사용.  
타겟이 이 근처에 있으면 W_kt가 올라감 (우선 추적 대상).

```python
        self.snr_threshold_dB = 20.0
        self.snr_threshold = 10 ** (self.snr_threshold_dB / 10.0)
```
20 dB = linear 100.0. SNR이 이 값 이상이어야 탐지 성공(alpha=1).

```python
        self.R_c_threshold = 1.218e6
```
통신 속도 임계값 (bits/s). 이 이하면 통신 페널티 부과.

```python
        self.d_min = 5.0
```
UAV 간 최소 거리 5m. 이 이하면 충돌로 간주.

```python
        self.lam1 = 2.0    # F_kt 계수
        self.lam2 = 15.0   # prod_loss 계수 (swarm 전체 실패 페널티)
        self.lam3 = 0.5    # 충돌/경계 페널티
        self.lam4 = 0.0    # 통신 페널티 (삭제됨 — 항상 0)
        self.lam5 = 8.0    # untracked 페널티 (어떤 UAV도 미탐지 시)
        self.lam6 = 8.0    # r_approach shaping 계수 (논문에 없음)
        self.lam7 = 15.0   # r_detect 양수 보상 계수 (논문에 없음)
```
보상 함수의 λ 가중치들. lam6/lam7은 논문 외 추가 항목.

```python
        self.sigma_r0_sq = 10.0
        self.sigma_theta0_sq = 1e-4
        self.sigma_r_sq_floor = 1e-3
        self.sigma_theta_sq_floor = 1e-7
```
EKF 측정 잡음 파라미터.  
- σ_r0² = 10: 기본 거리 측정 분산  
- σ_θ0² = 1e-4: 기본 각도 측정 분산  
- floor 값들: 수치 안정성용 하한선

```python
        self.uavs = [UAV(i, [0, 0], altitude=90.0) for i in range(self.num_uavs)]
        self.targets = [Target(i, [0, 0], [0, 0]) for i in range(self.num_targets)]
        self.assignment = {u: 0 for u in range(self.num_uavs)}
```
임시 초기화. 실제 초기화는 reset()에서 무작위로 수행.  
assignment: {UAV번호: 담당타겟번호}

```python
        dummy_local, dummy_global = self._get_obs()
        local_dim = len(dummy_local[0])   # 19 (num_targets=2일 때)
        global_dim = len(dummy_global)    # 31 (num_targets=2일 때)
```
관측 공간 크기를 자동 계산하기 위해 더미 관측 한 번 생성.

**local_dim 계산** (num_targets=2):
- uav_pos: 2
- energy: 1  
- per target: delta_q(2) + velocity(2) + W_kt(1) + eps_kt(1) = 6 × 2 = 12
- assignment one-hot: 2  
- **합계 = 17**

**global_dim 계산** (5 UAVs, 2 targets):
- per UAV: pos(2) + energy(1) = 3 × 5 = 15  
- per target: pos(2) + vel(2) + F_kt(1) + d_Z(1) + eps(1) + W(1) = 8 × 2 = 16  
- **합계 = 31**

```python
        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for _ in range(self.num_uavs)
        ]
```
연속 행동 공간: 각 UAV마다 (x방향, y방향) ∈ [-1, 1]²

---

### reset() (235~273줄)

```python
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.time_slot = 0
        self.bs_pos = np.array([0.0, 0.0], dtype=np.float32)
```
맵 중앙 (0,0)에 BS(기지국) 위치 고정.

```python
        # UAV를 타겟 기준 10~80m 반경에 무작위 배치 (30회 충돌 회피 재시도)
        for u_id, uav in enumerate(self.uavs):
            target = self.targets[self.assignment[u_id]]
            for _ in range(30):
                angle = np.random.uniform(0, 2 * np.pi)
                r = np.random.uniform(10.0, 80.0)
                pos = target.pos + r * np.array([np.cos(angle), np.sin(angle)])
                ...
```
UAV들을 담당 타겟 주위 10~80m 반경에 무작위 배치. 탐지 가능 거리(≈103m) 안에서 초기 탐지 보장.

```python
        self.targets = [
            Target(i,
                   [np.random.uniform(self.map_min*0.6, self.map_max*0.6), ...],
                   [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
            for i in range(self.num_targets)
        ]
```
타겟들을 맵의 60% 범위(±90m)에 무작위 배치, 초기 속도 ±2 m/s.

```python
        for target in self.targets:
            target.d_Z_kt = min(np.linalg.norm(target.pos - cz) for cz in self.critical_zones)
```
리셋 직후 각 타겟의 위험구역 거리 초기화.

```python
        self._invoke_aai()
```
AAI 호출 → W_kt, epsilon_kt, p_ut, assignment 설정.

```python
        obs = np.array(local_obs, dtype=np.float32)
        share_obs = np.array([global_state] * self.num_uavs, dtype=np.float32)
        obs = self._sanitize(obs)
        share_obs = self._sanitize(share_obs)
        return obs, share_obs
```
`obs`: (num_uavs, 17) — 각 UAV의 개별 관측  
`share_obs`: (num_uavs, 31) — 전역 상태를 모든 UAV에 복사 (centralized critic용)

---

### _sanitize() (275~277줄)

```python
    def _sanitize(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e4, 1e4)
```
NaN/Inf가 신경망에 들어가면 학습이 발산함. 0으로 치환하고 범위 제한.

---

### _invoke_aai() (279~299줄)

```python
    def _invoke_aai(self):
        if self.aai_callback is not None:
            self.aai_callback(self)
            self._build_assignment()
        elif self.use_aai:
            self._run_aai_heuristic()
        else:
            for t in self.targets:
                t.W_kt = 1.0
                t.epsilon_kt = 5.0
            for u in self.uavs:
                u.p_ut = 15.0
            self._build_assignment()
```
AAI 우선순위:
1. 외부 콜백(LLM) → 최우선
2. Heuristic AAI → use_aai=True일 때
3. 고정값(대조군) → use_aai=False일 때

```python
        if self.randomize_aai:
            for t in self.targets:
                t.W_kt = float(np.random.uniform(0.5, 3.5))
                t.epsilon_kt = float(np.random.uniform(1.0, 10.0))
            for u in self.uavs:
                u.p_ut = float(np.random.uniform(8.0, 30.0))
```
**Domain Randomization**: AAI가 먼저 assignment를 설정한 후, W/eps/p를 무작위로 덮어씀.  
MAPPO가 다양한 AAI 출력 범위에 적응하도록 함 (학습 시에만 사용).

---

### _build_assignment() (301~316줄)

```python
    def _build_assignment(self):
        unassigned = list(range(self.num_uavs))
        self.assignment = {}
        for t_idx, target in enumerate(self.targets):
            if not unassigned:
                break
            est_pos = target.S_global[:2]
            best_uav = min(unassigned,
                           key=lambda u: np.linalg.norm(self.uavs[u].pos - est_pos))
            self.assignment[best_uav] = t_idx
            unassigned.remove(best_uav)
```
각 타겟에 가장 가까운 UAV를 1:1로 먼저 배정. 타겟 수보다 UAV가 많으면...

```python
        for u_idx in unassigned:
            best_t = min(range(self.num_targets),
                         key=lambda t: np.linalg.norm(...))
            self.assignment[u_idx] = best_t
```
남은 UAV들은 가장 가까운 타겟에 추가 배정 (여러 UAV가 같은 타겟 담당 가능).

---

### _run_aai_heuristic() (318~342줄)

```python
    def _run_aai_heuristic(self):
        for target in self.targets:
            est_pos = target.S_global[:2]
            target.d_Z_kt = min(np.linalg.norm(est_pos - cz) for cz in self.critical_zones)
        for target in self.targets:
            if target.d_Z_kt < 100.0:
                target.W_kt = 3.0
                target.epsilon_kt = 2.0
            elif target.d_Z_kt < 200.0:
                target.W_kt = 2.0
                target.epsilon_kt = 5.0
            else:
                target.W_kt = 1.0
                target.epsilon_kt = 5.0
```
**Heuristic AAI 규칙**:
- 위험구역 100m 이내 → 최우선(W=3), 정밀 추적(ε=2)
- 100~200m → 보통 우선(W=2), 보통 정밀(ε=5)  
- 200m 초과 → 기본(W=1), 기본 정밀(ε=5)

```python
        for u_idx, uav in enumerate(self.uavs):
            target = self.targets[self.assignment[u_idx]]
            est_pos = target.S_global[:2]
            dist = np.linalg.norm(uav.pos - est_pos)
            if dist > 200.0:
                uav.p_ut = 25.0
            elif dist < 50.0:
                uav.p_ut = 10.0
            else:
                uav.p_ut = 15.0
```
**전력 조절**: 멀리 있으면 강한 레이더(25W), 가까이 있으면 절약(10W).

---

### _get_obs() (344~379줄)

```python
    def _get_obs(self):
        POS_SCALE = 150.0    # 위치 정규화 (맵 half-width)
        E_SCALE = 100000.0   # 에너지 정규화 (최대 에너지)
        F_SCALE = 1000.0     # PCRLB 정규화 (최대값)
        V_SCALE = 10.0       # 속도 정규화 (최대 속도)
```
신경망 입력을 [-1, 1] 또는 [0, 1] 범위로 정규화.

**전역 관측 (share_obs) 구성**:
```python
        for u in self.uavs:
            global_parts.extend(u.pos / POS_SCALE)   # 2: 위치
            global_parts.append(u.energy / E_SCALE)  # 1: 에너지
        for t in self.targets:
            global_parts.extend(t.S_global[:2] / POS_SCALE)  # 2: 추정 위치
            global_parts.extend(v_clipped / V_SCALE)          # 2: 추정 속도
            global_parts.append(min(t.F_kt, 1000.0) / F_SCALE) # 1: 추적 오차
            global_parts.append(t.d_Z_kt / POS_SCALE)          # 1: 위험구역 거리
            global_parts.append(t.epsilon_kt / 10.0)            # 1: 정확도 임계값
            global_parts.append(t.W_kt / 5.0)                   # 1: 우선순위 가중치
```

**지역 관측 (obs) 구성** (UAV별):
```python
        for u_idx, uav in enumerate(self.uavs):
            obs = list(uav.pos / POS_SCALE)  # 2: 내 위치
            obs.append(uav.energy / E_SCALE) # 1: 내 에너지
            for target in self.targets:
                delta_q = (target.S_global[:2] - uav.pos) / POS_SCALE  # 2: 타겟까지 방향/거리
                obs.extend(delta_q)
                obs.extend(v_clipped / V_SCALE)          # 2: 타겟 속도
                obs.append(target.W_kt / 5.0)            # 1: 타겟 우선순위
                obs.append(target.epsilon_kt / 10.0)     # 1: 타겟 정확도 임계값
            t_idx = self.assignment.get(u_idx, 0)
            for k in range(self.num_targets):
                obs.append(1.0 if k == t_idx else 0.0)  # 2: 내 담당 타겟 one-hot
```
`delta_q`가 핵심: 타겟 추정 위치 - 내 위치 = "어느 방향으로 가야 하나".

---

### step() (381~418줄)

```python
    def step(self, actions):
        self.time_slot += 1
        for target in self.targets:
            target.step(self.dt, Q=self.Q, ...)   # 1. 타겟 이동
        for i, uav in enumerate(self.uavs):
            hit = uav.apply_action(actions[i], ...)  # 2. UAV 이동
        self._update_local_tracking_and_energy()      # 3. 탐지 + EKF + 에너지 소모
        self._bs_fusion()                             # 4. BS에서 전역 추정 융합
        self._invoke_aai()                            # 5. AAI 업데이트
        n_collisions = self._count_collisions()       # 6. 충돌 체크
        team_reward = self._calculate_team_reward(...)  # 7. 보상 계산
```
**스텝 순서가 중요**:  
타겟 먼저 이동 → UAV 이동 → 탐지(현재 상태 기반) → BS 융합 → AAI 업데이트.

```python
        rewards = np.array([[team_reward]] * self.num_uavs, dtype=np.float32)
```
모든 UAV가 동일한 팀 보상을 받음 (cooperative MARL).

---

### _update_local_tracking_and_energy() (428~500줄)

```python
        for t in self.targets:
            t.measurement_info = np.zeros((4, 4), dtype=np.float32)
```
매 스텝 시작 시 측정 정보 초기화. 이번 스텝 탐지들을 새로 누적할 준비.

```python
            snr_sen, e_sen = uav.get_sensing_model(target.pos, self.dt)
            alpha_ukt = 1 if snr_sen >= self.snr_threshold else 0
```
**핵심 탐지 판정**: SNR ≥ 20 (13dB)이면 탐지 성공(α=1), 아니면 실패(α=0).  
탐지는 타겟의 **실제 위치(true position)**로 계산. (EKF 추정과 무관)

```python
            if alpha_ukt == 1:
                dx = target.pos[0] - uav.pos[0]
                dy = target.pos[1] - uav.pos[1]
                r_sq = dx**2 + dy**2
                r = math.sqrt(r_sq) if r_sq > 0 else 1e-6

                H = np.array([
                    [dx/r,    dy/r,    0, 0],
                    [-dy/r_sq, dx/r_sq, 0, 0]
                ], dtype=np.float32)
```
**측정 모델 야코비안 H** (극좌표 → 직교좌표 선형화).  
측정값 z = [거리, 각도] = [r, θ].  
상태 s = [x, y, vx, vy]에 대한 편미분.  
H[0] = ∂r/∂[x,y,vx,vy] = [dx/r, dy/r, 0, 0]  
H[1] = ∂θ/∂[x,y,vx,vy] = [-dy/r², dx/r², 0, 0]

```python
                sigma_r_sq = max(self.sigma_r0_sq / max(snr_sen, 1e-6), self.sigma_r_sq_floor)
                sigma_theta_sq = max(self.sigma_theta0_sq / max(snr_sen, 1e-6), ...)
                R_t = np.diag([sigma_r_sq, sigma_theta_sq]).astype(np.float32)
```
**측정 잡음 공분산 R**. SNR이 높을수록 잡음이 작아짐 (σ² ∝ 1/SNR).

```python
                target.measurement_info += H.T @ R_inv @ H
```
**측정 정보(Fisher Information) 누적**: 같은 타겟을 여러 UAV가 탐지하면 정보가 합산됨.  
BFIM 업데이트의 핵심 수식: J += H^T · R^{-1} · H

```python
                # EKF 예측 단계
                S_pred = self.F_mat @ target.S_global     # 상태 예측
                P_pred = self.F_mat @ target.P_global @ self.F_mat.T + self.Q  # 공분산 예측

                # 칼만 이득
                S_innov = H @ P_pred @ H.T + R_t
                K = P_pred @ H.T @ np.linalg.inv(S_innov + np.eye(2)*1e-6)

                # EKF 업데이트
                S_local = S_pred + K @ (z_ukt - pred_meas)  # 상태 업데이트
                P_local = (np.eye(4) - K @ H) @ P_pred       # 공분산 업데이트
```
**EKF(확장 칼만 필터) 로컬 업데이트**.  
각 UAV가 독립적으로 자기 탐지 결과로 로컬 추정값을 계산.  
이 추정값이 나중에 BS에서 융합됨.

```python
        for target in self.targets:
            prior_J = self.F_inv.T @ target.J_matrix @ self.F_inv + self.Q_inv
            target.J_matrix = prior_J + target.measurement_info
```
**BFIM 재귀 업데이트 (식 7)**:  
J_pred = [F · J_{t-1}^{-1} · F^T + Q]^{-1}  (Tichavsky 1998 표준)
J_t = J_pred + Σ_u(H^T·R^{-1}·H)  
`J_pred`가 예측(prediction) 단계의 BFIM.

```python
            PCRLB = np.linalg.inv(target.J_matrix)
            target.F_kt = float(min(np.trace(self.Lambda @ PCRLB @ self.Lambda.T), 1000.0))
```
**PCRLB 추적 오차 계산 (식 F_kt)**:  
F_kt = tr(Λ · J^{-1} · Λ^T)  
Λ = diag([1,1,0,0]) → 위치 성분만 측정. 1000으로 clipping.

---

### _bs_fusion() (502~549줄)

BS(기지국)에서 모든 UAV의 로컬 EKF 추정값을 융합하는 함수.

```python
        for t_idx, target in enumerate(self.targets):
            contributors = []
            for u_idx, uav in enumerate(self.uavs):
                est = uav.local_estimates.get(t_idx)
                if est is not None:
                    contributors.append(est)
            if len(contributors) == 0:
                target.S_global = self.F_mat @ target.S_global  # 순수 예측으로만 업데이트
                continue
```
이번 스텝에서 아무 UAV도 탐지 못 했으면 EKF 예측값으로만 전진.

```python
            omega = 1.0 / len(contributors)
            P_inv_sum = np.zeros((4, 4), dtype=np.float64)
            S_weighted = np.zeros(4, dtype=np.float64)
            for S_ut, P_ut in contributors:
                P_ut_inv = np.linalg.inv(P_ut64 + np.eye(4) * 1e-4)
                P_inv_sum += omega * P_ut_inv
                S_weighted += omega * (P_ut_inv @ S_ut)
```
**공분산 교차법(Covariance Intersection, CI)** 변형.  
각 UAV 추정의 역공분산으로 가중 평균. 정확한 UAV(P_inv 큰 것)에 더 많은 가중치.

```python
            P_tilde = np.linalg.inv(P_inv_sum + np.eye(4) * 1e-4)
            S_tilde = (P_tilde @ S_weighted).astype(np.float32)
```
융합된 로컬 추정값 S_tilde.

```python
            K_global = P_pred64 @ np.linalg.inv(P_pred64 + P_tilde64 + np.eye(4) * 1e-4)
            S_new = S_pred_global + K_global @ (S_tilde - S_pred_global)
```
전역 추정과 로컬 융합 추정을 다시 한번 칼만 업데이트.  
S_global이 이 결과로 업데이트됨.

---

### _calculate_team_reward() (551~578줄)

```python
        total_energy = sum(u.last_e_tot for u in self.uavs)
        r_energy = -total_energy / 100.0
```
**에너지 페널티**: 이번 스텝 총 소비 에너지 ÷ 100.

```python
        for t_idx, target in enumerate(self.targets):
            prod_loss = 1
            n_detected = 0
            for uav in self.uavs:
                a = uav.is_detected_per_target.get(t_idx, 0)
                prod_loss *= (1 - a)   # 아무도 탐지 안 하면 1, 한 명이라도 탐지하면 0
                n_detected += a
            r_tracking -= target.W_kt * (
                self.lam1 * (target.F_kt / 100.0) + self.lam2 * prod_loss
            )
```
**추적 보상** (식 32):  
- `target.F_kt / 100`: PCRLB 오차 (클수록 나쁨)  
- `prod_loss`: Π(1-α_u) = 0이면 누군가 탐지함, 1이면 아무도 탐지 못 함  
- W_kt로 타겟 우선순위 반영

```python
            if n_detected == 0:
                r_untracked -= self.lam5   # = -5
```
아무도 탐지 못 한 타겟마다 추가 -5 페널티.

```python
        r_collision = -self.lam3 * (n_collisions + boundary_violations)
```
UAV 간 충돌 + 경계 충돌 모두 -10씩 페널티.

```python
        for u in self.uavs:
            for t_idx, alpha in u.is_detected_per_target.items():
                if alpha == 1 and u.last_R_c < self.R_c_threshold:
                    r_comm -= self.lam4 * (self.R_c_threshold - u.last_R_c) / self.R_c_threshold
```
**통신 페널티**: 탐지는 했는데 BS까지 전송 속도가 부족하면 페널티.

---

### _log_step() 및 get_episode_log() (580~604줄)

```python
    def _log_step(self, team_reward, n_collisions):
        log = self._episode_log
        log['uav_pos'].append(...)        # UAV 위치들
        log['target_pos'].append(...)     # 타겟 실제 위치
        log['target_pos_est'].append(...) # 타겟 추정 위치
        log['energy_per_uav'].append(...) # UAV별 에너지 소모
        log['alpha'].append(alpha_mat)    # 탐지 행렬 (U×K)
        ...
```
`log_episode=True`로 설정하면 매 스텝 이 함수가 호출되어 모든 데이터 저장.

```python
    def get_episode_log(self):
        if self._episode_log is None:
            return None
        return {k: np.asarray(v) for k, v in self._episode_log.items()}
```
리스트를 numpy 배열로 변환하여 반환. plot_episode.py에서 사용.

---

# 2. uav_vec_env.py — 벡터화 래퍼

**역할**: 여러 환경 인스턴스를 동시에 실행하는 병렬 래퍼.  
학습 시 `n_rollout_threads=4`면 UAVTrackingEnv 4개를 동시에 돌림.

```python
class UAVDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
```
`env_fns`: 환경 생성 함수 목록. 각 함수를 호출해 독립적인 환경 인스턴스 생성.

```python
    def step_async(self, actions):
        self.actions = actions  # 행동 저장만 (아직 실행 안 함)

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
```
`step_async` + `step_wait` 패턴: 비동기 처리를 흉내내지만 실제로는 순차 실행.  
SubprocVecEnv로 바꾸면 진짜 병렬이 됨 (현재는 단순 sequential로 충분).

```python
        obs, share_obs, rews, dones, infos = map(np.array, zip(*results))
```
4개 환경 결과를 하나의 배열로 묶음.  
- obs: (4, 5, 17) → (n_envs, n_agents, obs_dim)  
- rews: (4, 5, 1)

```python
        for (i, done) in enumerate(dones):
            if np.all(done):
                o, s = self.envs[i].reset()
                obs[i] = o
                share_obs[i] = s
```
에피소드 종료(모든 UAV가 done=True)되면 자동 리셋.  
리셋 후 첫 관측을 즉시 반환.

---

# 3. uav_runner.py — 학습 루프

**역할**: MAPPO 학습의 실제 실행 루프. 환경과 상호작용하며 경험을 버퍼에 저장,  
주기적으로 정책 업데이트.

```python
class UAVRunner(Runner):
    def __init__(self, config):
        super(UAVRunner, self).__init__(config)
        self._save_run_meta()
```
`Runner` (base_runner.py)에서 버퍼, 정책, 트레이너, 로거 등 기본 설정 상속.

### _save_run_meta() (24~50줄)

```python
    def _save_run_meta(self):
        meta = {
            'obs_shape': list(self.envs.observation_space[0].shape),  # [17]
            'share_obs_shape': list(self.envs.share_observation_space[0].shape),  # [31]
            'action_shape': list(self.envs.action_space[0].shape),  # [2]
            ...
        }
        with open(os.path.join(save_dir, 'run_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
```
**평가 시 actor 재구성에 필요한 정보 저장**.  
나중에 evaluate_trained.py가 이 파일을 읽어서 정확한 크기로 신경망을 만듦.

### run() (52~98줄)

```python
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
```
총 에피소드 수 계산. 2M steps ÷ 100 steps ÷ 4 threads = 5,000 에피소드.

```python
        for episode in range(episodes):
            for step in range(self.episode_length):
                values, actions, ... = self.collect(step)  # 정책으로 행동 선택
                obs, share_obs, rewards, dones, infos = self.envs.step(actions_env)  # 환경 실행
                self.insert(data)  # 버퍼에 저장
            self.compute()   # 리턴 / 어드밴티지 계산
            train_infos = self.train()  # PPO 업데이트
```
**MAPPO 학습 루프 핵심**:  
collect → step → insert → (100 스텝 반복) → compute → train

### warmup() (100~108줄)

```python
    def warmup(self):
        obs, share_obs = self.envs.reset()
        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()
```
학습 시작 전 환경 리셋, 버퍼의 첫 번째 슬롯에 초기 관측 저장.

### collect() (110~130줄)

```python
    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(
                np.concatenate(self.buffer.share_obs[step]),  # critic 입력
                np.concatenate(self.buffer.obs[step]),        # actor 입력
                ...
            )
```
`@torch.no_grad()`: 행동 선택 시 그래디언트 불필요, 메모리/속도 최적화.  
`np.concatenate`: (n_threads, n_agents, dim) → (n_threads·n_agents, dim) 펼치기.

```python
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
```
다시 (n_threads, n_agents, dim) 형태로 되돌리기.

### insert() (132~144줄)

```python
        rnn_states[dones == True] = np.zeros(...)
```
에피소드가 끝난 환경의 RNN 상태를 0으로 초기화.  
(mappo 비재귀 모드에서는 항상 0이라 사실상 무의미하지만 형식적으로 유지)

### eval() (146~176줄)

```python
    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_obs, eval_share_obs = self.eval_envs.reset()
        for eval_step in range(self.episode_length):
            eval_action, eval_rnn_states = self.trainer.policy.act(...)
            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
        eval_average_episode_rewards = np.mean(...)
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
```
**훈련 중 성능 모니터링**. `--use_eval` 플래그로 활성화됨.  
터미널에 나오는 "eval average episode rewards is ~~"가 바로 이 출력.  
이 숫자가 올라가면 정책이 좋아지는 중.

---

# 4. train_uav.py — 학습 진입점

**역할**: 명령줄 인자 파싱 → 환경 생성 → Runner 실행.

```python
def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = UAVTrackingEnv(num_uavs=all_args.num_agents, num_targets=2,
                                 randomize_aai=True)  # domain randomization 활성화
            env.seed(all_args.seed + rank * 1000)  # 각 thread마다 다른 시드
            return env
        return init_env
    return UAVDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])
```
학습용 환경. `randomize_aai=True` → W/eps/p 무작위화 적용.

```python
def make_eval_env(all_args):
    ...
    env = UAVTrackingEnv(num_uavs=all_args.num_agents, num_targets=2)
    # randomize_aai 없음 → 안정적인 heuristic AAI로 평가 모니터링
```
평가용 환경. 무작위화 없이 안정적인 결과 모니터링.

```python
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to checkpoint dir to resume training from ...")
```
중단된 학습 이어서 하기. `run_full_pipeline.bat`이 자동으로 이 플래그 설정.

```python
    if all_args.resume_from is not None:
        run_dir = Path(all_args.resume_from)
        if run_dir.name == 'models':
            all_args.model_dir = str(run_dir)
            run_dir = run_dir.parent
```
`--resume_from .../run1/models` → `model_dir = .../run1/models`, `run_dir = .../run1/`.  
base_runner가 model_dir에서 actor/critic 가중치를 불러옴.

```python
    runner = UAVRunner(config)
    runner.run()
    done_flag = run_dir / 'training_done.txt'
    done_flag.write_text('Training completed successfully.\n')
```
학습 완료 후 플래그 파일 생성. bat 파일이 이 파일 존재 여부로 학습 완료 판단.

---

# 5. llm_aai.py — LLM 기반 AAI

**역할**: Claude/GPT/DeepSeek/Gemini 같은 LLM을 AAI로 사용하는 모듈.  
평가 시 heuristic 대신 LLM이 W_kt, epsilon_kt, p_ut를 결정.

## 전체 흐름

```
UAV 환경 step()
    → _invoke_aai()
        → aai_callback(env)   ← LLMAAI.callback(env)이 여기에 등록됨
            → (throttling 체크) → (캐시 체크) → LLM API 호출
                → JSON 파싱 → env.targets/uavs에 값 주입
                → (실패 시) → heuristic fallback
```

## 핵심 설계: Throttling + Caching

```python
        self.call_every = max(1, int(call_every))  # 기본 10
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._last_output: Optional[Dict[str, Any]] = None
        self._last_step: int = -10**9
```
- **Throttling**: 매 step 호출하면 100 step 에피소드에서 100번 API 호출 → 비용/시간 과다.  
  기본 10 step마다 1번 호출, 사이엔 마지막 결과 재사용.
- **Cache**: 같은 상황(30m 단위로 반올림)이 반복되면 저장된 결과 재사용.

## 클라이언트 초기화 (Provider별)

```python
        if provider == 'anthropic':
            self._client = anthropic.Anthropic(api_key=...)
        elif provider == 'openai':
            self._client = openai.OpenAI(api_key=...)
        elif provider == 'deepseek':
            self._client = openai.OpenAI(api_key=..., base_url='https://api.deepseek.com')
        elif provider == 'gemini':
            self._client = openai.OpenAI(api_key=..., 
                base_url='https://generativelanguage.googleapis.com/v1beta/openai/')
        elif provider == 'mock':
            pass  # LLM 없이 heuristic만 사용
```
DeepSeek와 Gemini는 OpenAI 호환 API를 제공하므로 openai SDK로 처리.  
base_url만 바꿔주면 됨.

## _call_llm_api() (295~334줄)

```python
                if self.provider == 'anthropic':
                    msg = self._client.messages.create(
                        model=self.model, max_tokens=1024,
                        system=system_prompt,
                        messages=[{"role": "user", "content": user_prompt}],
                    )
                    text = msg.content[0].text
                elif self.provider in ('openai', 'deepseek', 'gemini'):
                    kwargs = dict(model=self.model, messages=[...], timeout=self.timeout)
                    if self.provider in ('openai', 'deepseek'):
                        kwargs['response_format'] = {"type": "json_object"}
                    resp = self._client.chat.completions.create(**kwargs)
                    text = resp.choices[0].message.content
```
`response_format=json_object`: OpenAI/DeepSeek는 JSON 모드 강제.  
Gemini는 미지원이므로 system prompt에만 의존 → _parse_llm_response가 처리.

## _parse_llm_response() (336~394줄)

```python
        if text.startswith('```'):
            # ```json ... ``` 코드 블록 제거
            lines = text.split('\n')[1:]
            if lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)  # JSON 부분만 추출
```
LLM이 JSON을 마크다운 코드블록으로 감싸거나 주석을 달아도 처리 가능.

```python
        if len(targets) != expected_n_targets or len(uavs) != expected_n_uavs:
            return None
```
타겟/UAV 수가 환경과 맞지 않으면 파싱 실패. heuristic fallback으로 넘어감.

```python
        clean_targets.append({
            'W_kt': float(np.clip(W, 0.5, 5.0)),
            'epsilon_kt': float(np.clip(eps, 0.5, 20.0)),
        })
```
LLM이 범위를 벗어난 값을 출력해도 클리핑으로 안전하게 처리.

## callback() (404~474줄)

```python
    def callback(self, env):
        cur_dim = (env.num_uavs, env.num_targets)
        if getattr(self, '_last_dim', None) != cur_dim:
            self._cache.clear()
            ...
```
환경 크기가 바뀌면 캐시 무효화 (sweep 평가 시 UAV 수가 바뀔 때).

```python
        steps_since_last = env.time_slot - self._last_step
        if self._last_output is not None and steps_since_last < self.call_every:
            self.stats['n_throttled'] += 1
            self._apply_output(env, self._last_output)
            return
```
Throttling: 마지막 호출 후 call_every 스텝이 안 지났으면 이전 출력 재사용.

```python
        cache_key = _telemetry_cache_key(telemetry)
        if cache_key in self._cache:
            ...
```
Cache hit: 같은 상태(30m 그리드, 50m 단위 거리)이면 API 호출 없이 반환.

```python
        if output is None:
            output = heuristic_aai_output(env)
```
모든 게 실패하면 heuristic fallback. 평가가 멈추지 않음.

## default_callback() (483~499줄)

```python
def default_callback(env):
    global _default_instance
    if _default_instance is None:
        _default_instance = LLMAAI(
            model=os.environ.get('LLM_AAI_MODEL', 'claude-sonnet-4-6'),
            provider=os.environ.get('LLM_AAI_PROVIDER', 'anthropic'),
            call_every=int(os.environ.get('LLM_AAI_CALL_EVERY', '10')),
        )
    _default_instance.callback(env)
```
환경변수로 설정 변경 가능:
- `LLM_AAI_PROVIDER=deepseek` + `DEEPSEEK_API_KEY=sk-...` → DeepSeek 사용
- `LLM_AAI_PROVIDER=mock` → API 없이 테스트

---

# 6. run_evaluation.py — 에피소드 실행기

**역할**: 에피소드 실행 라이브러리. evaluate_trained.py가 이 함수들을 호출.

## run_episode() (83~127줄)

```python
def run_episode(env, policy_fn, log_episode=True):
    env.log_episode = log_episode
    obs, share_obs = env.reset()
    done = False
    while not done:
        action = policy_fn(obs)        # 정책으로 행동 선택
        obs, share_obs, reward, dones, infos = env.step(action)
        # 통계 누적
        energy_per_step.append(...)
        rewards.append(...)
        ...
        done = bool(dones[0])  # 첫 번째 UAV의 done (모두 동일)
```
에피소드 하나를 실행하고 통계를 반환.  
`dones[0]`만 확인해도 되는 이유: 모든 UAV가 time_slot >= max_steps에서 동시에 done.

## policy 함수들 (132~157줄)

```python
def policy_random(obs):
    n_uav = obs.shape[0]
    return np.random.uniform(-1, 1, (n_uav, 2)).astype(np.float32)
```
완전 랜덤 행동. 비교를 위한 하한선.

```python
def policy_naive_greedy_factory(env):
    def policy(obs):
        for u_idx, uav in enumerate(env.uavs):
            t_idx = env.assignment.get(u_idx, 0)
            target_est = env.targets[t_idx].S_global[:2]
            direction = target_est - uav.pos
            norm = np.linalg.norm(direction) + 1e-6
            actions[u_idx] = direction / norm
        return actions
    return policy
```
각 UAV가 담당 타겟 추정 위치를 향해 직진. 단순하지만 합리적인 baseline.  
Factory 패턴: env에 접근해야 assignment를 알 수 있어서 closure 사용.

```python
def policy_hover(obs):
    return np.zeros((n_uav, 2), dtype=np.float32)
```
움직이지 않는 정책. 이동 에너지 최소화의 이론적 하한.

## run_sweep() (162~208줄)

```python
def run_sweep(env_factory, policy_name, policy_factory, x_axis_label,
              x_axis_values, varying_arg, n_seeds=3, ...):
    for x_val in x_axis_values:  # ex: [3, 5, 7, 10]
        per_seed_stats = []
        for seed in range(n_seeds):
            for ep in range(n_episodes_per_seed):
                env = env_factory(x_val, seed)
                env.seed(seed * 1000 + ep)
                policy = policy_factory(env)
                res = run_episode(env, policy, log_episode=False)
                per_seed_stats.append(res)
        agg = {
            'energy_total_mean': float(np.mean([s['energy_total'] for s in per_seed_stats])),
            'F_kt_mean': ...,
            'F_kt_std': ...,   # 시드 간 분산 (에러바용)
            'detection_rate': ...,
            ...
        }
        fname = f'{policy_name}_{varying_arg}={x_val}.npz'
        np.savez(fname, **agg)
```
변수 하나(예: UAV 수)를 바꿔가며 조건별 평균/표준편차를 저장.  
저장된 .npz 파일이 나중에 plot_energy_curves.py의 입력이 됨.

## save_episode() (211~233줄)

```python
def save_episode(env, policy_fn, save_path, method_name, seed=42):
    env.seed(seed)
    env.log_episode = True
    res = run_episode(env, policy_fn, log_episode=True)
    log = res['log']
    out = {'method': method_name, 'seed': seed, 'num_uavs': env.num_uavs, ...}
    out.update(log)  # UAV 위치, 타겟 위치, 에너지 등 모든 시계열 포함
    np.savez(save_path, **out)
```
단일 에피소드의 상세 시계열 저장. plot_episode.py가 이를 읽어 멀티패널 시각화.

---

# 7. evaluate_trained.py — 평가 CLI

**역할**: 학습된 MAPPO actor를 불러와서 여러 조건에서 평가하는 메인 스크립트.

## load_mappo_actor() (104~155줄)

```python
def load_mappo_actor(checkpoint_dir, env, device='cpu'):
    meta, args_dict = _load_run_meta(checkpoint_dir)  # run_meta.json 로드
    ns = _build_args_namespace(args_dict, meta)        # args 복원
    actor = R_Actor(ns, obs_space, act_space, device=...)  # 신경망 재구성
    state_dict = torch.load('.../actor.pt', map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()
```
학습 시 저장한 run_meta.json으로 정확히 동일한 크기의 신경망을 만들고 가중치 로드.

```python
    @torch.no_grad()
    def policy_fn(obs):
        obs_t = torch.from_numpy(np.asarray(obs, ...))
        action, _, rnn_new = actor(obs_t, rnn_t, masks_t, deterministic=True)
        return action.cpu().numpy().astype(np.float32)
    return policy_fn, state
```
반환된 `policy_fn`은 run_evaluation.py의 policy_fn과 동일한 시그니처.  
`deterministic=True`: 평가 시 확률적 샘플링 대신 최빈값(mode) 선택.

## make_env() (194~224줄)

```python
def make_env(num_uavs, num_targets, baseline_name, aai_callback=None, ...):
    if baseline_name == 'mappo':
        use_aai = False  # 대조군: AAI 없음
    elif baseline_name == 'llm_aai':
        use_aai = True
        cb = aai_callback
    else:  # mappo+aai, naive_greedy, random, hover
        use_aai = True  # heuristic AAI 사용
        cb = None
```
baseline 이름에 따라 AAI 설정 자동 결정.

## Tau sweep 구현 (295~316줄)

```python
        def make_env_with_tau(tau_s_r):
            env = make_env(...)
            original_reset = env.reset
            def patched_reset(*a, **kw):
                res = original_reset(*a, **kw)
                for u in env.uavs:
                    u.tau_s_ratio = tau_s_r  # 리셋 후 새 UAV 객체들에 적용
                return res
            env.reset = patched_reset
            return env
```
Monkey-patching: reset()을 래핑해서 매번 tau_s_ratio를 설정.  
reset()에서 새 UAV 객체가 생성된 후 tau_s_ratio를 다시 설정해야 하므로 이 방법 사용.

---

# 8. test_llm_call.py — LLM 단독 테스트

**역할**: LLM AAI를 평가 파이프라인 없이 단독으로 테스트.

```python
    for _ in range(10):
        actions = np.random.uniform(-1, 1, (5, 2)).astype(np.float32)
        env.step(actions)
```
10스텝 랜덤 실행 후 상태가 초기화된 후라 더 현실적인 telemetry 생성.

```python
    print(SYSTEM_PROMPT)
    print(user_prompt)
    print(json.dumps(h_out, indent=2))  # heuristic 출력 (비교 기준)
    ...
    text = aai._call_llm_api(SYSTEM_PROMPT, user_prompt)
    print("Raw response:", text)
    parsed = aai._parse_llm_response(text, ...)
    print("Parsed output:", json.dumps(parsed, indent=2))
```
프롬프트, heuristic 기준값, LLM 원본 응답, 파싱 결과를 모두 출력.  
LLM이 이상한 출력을 내면 여기서 디버깅.

---

## 전체 데이터 흐름 요약

```
train_uav.py
    └─ UAVDummyVecEnv (4개 parallel)
         └─ UAVTrackingEnv × 4
              ├─ randomize_aai=True (domain randomization)
              └─ step() → reward → buffer
                            ↓
                       UAVRunner.run()
                            ↓
                       R_MAPPO.update()
                            ↓
                       actor.pt, critic.pt

run_full_pipeline.bat
    └─ evaluate_trained.py
         ├─ load_mappo_actor() → policy_fn
         ├─ make_env() → UAVTrackingEnv
         │    ├─ mappo+aai: use_aai=True, heuristic AAI
         │    ├─ mappo:     use_aai=False, 대조군
         │    └─ llm_aai:  aai_callback=LLMAAI.callback
         └─ run_sweep() / save_episode()
              └─ run_evaluation.py
                   └─ .npz 파일 저장
                         ↓
                   plot_energy_curves.py → 그래프
                   plot_episode.py → 에피소드 시각화
```

---

*생성일: 2026-05-11*
