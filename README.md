# 날파리 (Nalpari) — Multi-UAV Target Tracking with MAPPO + AAI

논문 *KJH___multi_UAV_search_trajectory* 구현 (Dec-POMDP MTT). MAPPO 정책 + Adaptive AI(AAI) 모듈 + (선택) LLM AAI 결합 구조.

---

## 1. 디렉토리 구조

```
nalpari/
├── onpolicy/                              # MAPPO 학습 코드 (수정됨)
│   ├── envs/uav/
│   │   ├── uav_env.py                     # ★ 환경 (dynamics + reward + reset 초기화)
│   │   ├── uav_vec_env.py                 # 벡터화 wrapper
│   │   └── __init__.py
│   ├── runner/shared/
│   │   ├── uav_runner.py                  # 학습 루프
│   │   └── base_runner.py                 # ★ save/restore (optimizer/value_norm 추가됨)
│   ├── scripts/train/
│   │   └── train_uav.py                   # 학습 entrypoint
│   ├── algorithms/r_mappo/                # MAPPO 트레이너 (수정 X)
│   └── utils/                             # buffer, valuenorm 등
│
├── tools/                                 # 평가 + 시각화
│   ├── run_evaluation.py                  # 평가 라이브러리 (run_episode, run_sweep, baseline policies)
│   ├── evaluate_trained.py                # 평가 CLI
│   ├── llm_aai.py                         # LLM AAI 모듈 (Anthropic / OpenAI / DeepSeek / Gemini)
│   ├── test_llm_call.py                   # LLM 단독 테스트
│   ├── plot_energy_curves.py              # Sweep 비교 그래프 (4-panel)
│   ├── plot_episode.py                    # 단일 episode 멀티 패널
│   └── plot_system_diagram.py             # 시스템 다이어그램
│
├── run_full_pipeline.bat                  # ★ Windows 통합 파이프라인 (train → eval → plot)
├── evaluation/nalpari_v1/                 # 평가 결과 (npz)
├── figures/nalpari_v1/                    # 시각화 출력 (png)
└── README.md                              # 이 문서
```

---

## 2. 핵심 환경: `onpolicy/envs/uav/uav_env.py`

### 2.1 클래스 구조

#### `Target` (line 21–58)
- `pos`, `velocity`: 실제 상태 (정책이 모르는 ground truth)
- `S_global` (4-vec): BS 융합 estimate `[x, y, vx, vy]`
- `P_global`: estimate 공분산
- `J_matrix`: Bayesian Fisher Information (paper 식 26)
- `F_kt`: paper 식 27 (trace of PCRLB)
- `W_kt, epsilon_kt`: AAI가 결정하는 priority weight, accuracy threshold
- `d_Z_kt`: 가장 가까운 risk zone까지 거리
- `step()`: 식 (1) 등속 모션 + 프로세스 노이즈 + 경계 반사

#### `UAV` (line 60–145)
- `pos, energy, p_ut`: 위치, 잔여 에너지, 송신 전력
- `local_estimates[t_idx]`: 자체 EKF estimate (BS 융합 전)
- `is_detected_per_target`: 이번 step의 α 값 (0/1)
- `get_sensing_model()`: 거리⁻⁴ SNR 계산 + 센싱 에너지
- `get_comm_model()`: BS로 통신 시 Shannon rate + 에너지
- `get_mobility_energy()`: 이동에 따른 에너지

#### `UAVTrackingEnv` (line 148–) — gym.Env
주요 인자:
- `num_uavs=5`, `num_targets=2`, `dt=1.0`, `max_steps=150`
- `use_aai=True` → heuristic AAI 활성화
- `aai_callback=None` → 평가 시 LLM AAI callback 주입 hook
- `randomize_aai=False` → 논문 설계 그대로
- `sigma_w_sq=0.1` → 프로세스 노이즈 분산 (sweep_noise에서 가변)
- `num_critical_zones=2` → 위험구역 개수

### 2.2 초기화 (`reset`, line 238–) — UAV를 타겟 근처에 배치

타겟이 너무 멀어 처음부터 detection 실패하면 EKF가 작동 불가능하다.
한 번 놓치면 위치 정보 자체가 없어 복원 불가능하므로, 초기 detection을 강제로 보장한다.

```python
# 1) 타겟 랜덤 생성 (맵 60% 범위, 초기 속도 ±2 m/s)
# 2) UAV를 round-robin으로 타겟에 할당
#    U=5, T=2: uavs_per_target = {0: [u0, u2, u4], 1: [u1, u3]}
# 3) 각 UAV: 담당 타겟 기준 10~50m 반경 랜덤 배치
#    - 30회 재시도로 d_min=5m 충돌 회피
```

UAV 고도 H=80m. 탐지 가능 수평 거리 ≈ 52m이므로 50m 이내 배치 시 초기 탐지 보장 (slant ≈ 94m < 104m 임계).

### 2.3 Reward 가중치

[uav_env.py:187-193](onpolicy/envs/uav/uav_env.py#L187-L193)
```python
self.lam1 = 2.0    # F_kt 계수
self.lam2 = 15.0   # prod_loss 계수 (swarm 전체 실패 페널티)
self.lam3 = 0.5    # 충돌/경계 페널티
self.lam4 = 0.0    # 통신 페널티 (삭제됨)
self.lam5 = 8.0    # untracked 추가 페널티
self.lam6 = 8.0    # r_approach shaping 계수 (논문에 없음)
self.lam7 = 15.0   # r_detect 양수 보상 계수 (논문에 없음)
```

lam6/lam7은 논문 원본에 없는 추가 항목 — 재탐색 gradient와 탐지 성공 양수 보상을 제공한다 (섹션 2.4 참고).

### 2.4 `_calculate_team_reward` (line 594–) — 항목별 의미

```python
r_energy = -total_energy / 100.0
```
모든 UAV의 이번 step 총 에너지 소비. 보통 -1 ~ -3/step.

```python
for t_idx, target in enumerate(self.targets):
    prod_loss = 1
    n_detected = 0
    for uav in self.uavs:
        a = uav.is_detected_per_target.get(t_idx, 0)
        prod_loss *= (1 - a)
        n_detected += a
    r_tracking -= target.W_kt * (
        self.lam1 * (target.F_kt / 100.0) + self.lam2 * prod_loss
    )
    if n_detected == 0:
        r_untracked -= self.lam5   # 완전 미탐지: -8 per target
    else:
        r_detect += self.lam7      # 탐지 성공: +15 per target
```
- `prod_loss=1` (아무도 탐지 못함) → W×lam2 페널티 + r_untracked
- `prod_loss=0` (한 대 이상 탐지) → F_kt 항만 + r_detect

```python
r_collision = -self.lam3 * (n_collisions + boundary_violations)
```
충돌/경계 위반 1회당 -0.5.

```python
r_comm = 0.0  # lam4=0이므로 항상 0 (통신 페널티 삭제됨)
```

```python
r_approach = 0.0
for t_idx, target in enumerate(self.targets):
    min_dist = min(np.linalg.norm(uav.pos - target.pos) for uav in self.uavs)
    r_approach += max(0.0, 1.0 - min_dist / 200.0) * self.lam6
```
각 타겟에 가장 가까운 UAV 거리 기반 shaping. 200m 이내에 접근할수록 +보상.  
탐지가 끊겼을 때 UAV가 타겟 방향으로 이동하도록 유도하는 재탐색 gradient.

총합: `r_energy + r_tracking + r_untracked + r_detect + r_collision + r_comm + r_approach`

### 2.5 Observation (line 373–410)

#### Local obs (per UAV): `3 + 8T` 차원
- UAV 위치 (2), 에너지 (1)
- 각 타겟별: delta (2) + vel (2) + W (1) + eps (1) + alpha (1) = 7 × T
- assignment one-hot (T)

T=2일 때 **19차원**. **U와 무관** → 평가 시 다른 U에서도 actor 실행 가능.  
`alpha`: 직전 step에서 이 UAV가 해당 타겟을 탐지했는지 (0/1). 탐지 가능 범위 여부를 직접 피드백.

#### Share obs (global state): `3U + 8T` 차원
Centralized critic 입력. U=5, T=2일 때 31차원.

### 2.6 PCRLB / F_kt 계산 (line 495–502)

```python
P_post_prev = np.linalg.inv(target.J_matrix)
P_prior     = F · P_post_prev · Fᵀ + Q
prior_J     = inv(P_prior)
target.J_matrix = prior_J + target.measurement_info
PCRLB = inv(target.J_matrix)
target.F_kt = trace(Λ · PCRLB · Λᵀ)
```

표준 PCRLB 재귀 사용 (자세한 사항은 섹션 12 참고).

### 2.7 AAI 헬퍼 (line 279–344)

- `_run_aai_heuristic`: `d_Z`에 따라 W, eps, p_ut를 규칙 기반 결정
- `_invoke_aai`: callback 있으면 그것 사용 (LLM), 없고 `use_aai=True`면 heuristic, 둘 다 X면 고정값
- `randomize_aai=True`면 위 결정을 덮어쓰고 `np.random.uniform`으로 재설정 (학습에는 False)

### 2.8 `step()` 흐름 (line 383–418)

1. `time_slot += 1`
2. Target 동역학 갱신
3. UAV 행동 적용 + 경계 클리핑
4. `_update_local_tracking_and_energy`: 각 UAV sensing/EKF/에너지 갱신
5. `_bs_fusion`: 모든 UAV local estimate → BS에서 융합
6. `_invoke_aai`: AAI 다시 호출
7. `_count_collisions`
8. `_calculate_team_reward`
9. obs/share_obs/reward/dones/infos 반환

---

## 3. 학습 코드

### 3.1 `train_uav.py` — entrypoint
[onpolicy/scripts/train/train_uav.py](onpolicy/scripts/train/train_uav.py)

- `make_train_env`: `UAVTrackingEnv(num_uavs=N, num_targets=2, randomize_aai=False)` 생성
- `--resume_from path/to/models`: 체크포인트 로드해서 학습 재개
  - actor/critic + optimizer + value_normalizer 모두 복원

### 3.2 `uav_runner.py` — 학습 루프

```python
episodes = int(num_env_steps) // episode_length // n_rollout_threads
```
NUM_ENV_STEPS=50000000, ep_len=150, n_threads=4 → **83333 에피소드**.

매 episode 후 모델 저장 (save_interval=1, 덮어쓰기).

### 3.3 `base_runner.py:save/restore`
[onpolicy/runner/shared/base_runner.py](onpolicy/runner/shared/base_runner.py)

`save()`가 저장하는 5개 파일:
- `actor.pt`, `critic.pt`: 네트워크 가중치
- `actor_optim.pt`, `critic_optim.pt`: Adam optimizer state
- `value_normalizer.pt`: ValueNorm running stats

이전에는 actor/critic만 저장해 resume 시 Adam momentum과 value norm이 0으로 리셋되어 학습이 불안정했다. 5개 모두 복원하면 진짜 이어서 학습 가능하다.

---

## 4. 평가 코드

### 4.1 `tools/run_evaluation.py`

#### `run_episode(env, policy_fn, log_episode=True)` (line 82–)
한 에피소드 실행 + 메트릭 계산.

| 변수 | 의미 |
|---|---|
| `reward_mean` | episode 총합 (sum) — 훈련 metric과 비교 가능 |
| `reward_per_step` | per-step 평균 |
| `detection_rate` | alpha=1 비율 |
| `collision_count` | 누적 충돌 수 |
| `untracked_count` | n_det == 0인 step·타겟 수 |
| `F_kt_mean` | paper 공식 (saturate되어 0 근처) |
| `tracking_err_mean` | 실제 ‖S_est - S_true‖ — **paper F_kt의 한계 보완** |
| `tracking_err_detected_mean` | 탐지된 step만 한정 |

#### `run_sweep(env_factory, policy_factory, x_axis_values, ...)`
x축 값마다 n_seeds × n_episodes 평가, npz 저장. obs shape mismatch 시 해당 x_val 스킵.

#### Baseline policies
- `policy_random`: 액션 -1~1 uniform
- `policy_hover`: 0 액션
- `policy_naive_greedy_factory(env)`: 각 UAV가 자기 할당 타겟의 추정 위치로 단위벡터

### 4.2 `tools/evaluate_trained.py`

#### `load_mappo_actor(checkpoint_dir, env, device)`
1. `run_meta.json` + `all_args.pkl` 로드 → R_Actor 재구성
2. obs shape mismatch면 `ValueError` raise
3. `actor.load_state_dict(...)` → eval 모드
4. `policy_fn(obs) → action`: deterministic forward pass

#### `make_env(...)`
baseline마다 use_aai / aai_callback / randomize_aai 자동 결정:
- `mappo`: use_aai=False
- `mappo+aai`: use_aai=True (heuristic)
- `llm_aai`: use_aai=True, callback=LLM
- `naive_greedy/random/hover`: use_aai=True

CLI `--randomize_aai`로 분포 명시 (기본 False).

### 4.3 `tools/llm_aai.py`

#### `LLMAAI` 클래스
- `provider`: `'anthropic' / 'openai' / 'deepseek' / 'gemini' / 'mock'`
- `model`: 모델 ID
- `call_every=10`: 매 K step마다 LLM 호출, 사이엔 last output 재사용
- 캐싱: 반올림된 telemetry hash로 동일 상황 재사용
- Fallback: API/JSON 실패 시 heuristic으로

#### `default_callback(env)`: 환경변수 기반 instance 자동 사용
- `LLM_AAI_PROVIDER`, `LLM_AAI_MODEL`, `LLM_AAI_CALL_EVERY`, `LLM_AAI_VERBOSE`

---

## 5. 시각화

### 5.1 `plot_energy_curves.py` — sweep 비교 (4-panel)
1. **Total energy**: 메서드별 episode 총 에너지
2. **Detection rate + collision** (twin axis)
3. **Estimation error**: 실제 ‖S_est-S_true‖ — F_kt 대신
4. **Per-step energy**: 시간축 에너지 곡선

### 5.2 `plot_episode.py` — 단일 episode 멀티 패널
- Top-down trajectory
- Energy timeline
- F_kt / AAI 시계열 / Detection α 히트맵
- 5개 time snapshot

### 5.3 `plot_system_diagram.py` — 시스템 구조도

---

## 6. 통합 파이프라인: `run_full_pipeline.bat`

### 6.1 상단 config
```bat
set EXP_NAME=nalpari_v1
set NUM_AGENTS=5
set NUM_ENV_STEPS=50000000   :: 83333 에피소드
set N_ROLLOUT=4
set N_SEEDS=3
set N_EPISODES=3
set KMP_DUPLICATE_LIB_OK=TRUE
set PYTHONPATH=%cd%
```

### 6.2 PPO 옵션
```bat
set "STABLE_OPTS=--lr 1e-4 --entropy_coef 0.05 --ppo_epoch 5 --num_mini_batch 4"
```
run4에서 lr=1e-4, ppo_epoch=5 조합이 21% 달성하여 확정.

### 6.3 흐름
1. 체크포인트 검출: `latest/models/actor.pt` + `training_done.txt` 존재 여부로 학습/평가/archive 결정
2. 메뉴 (choice 1/2):
   - `[1]` 학습 재개 또는 평가만
   - `[2]` 기존 결과 archive 후 fresh 학습
3. 학습: `train_uav.py + STABLE_OPTS`
4. 평가: 6개 baseline (mappo+aai, mappo, naive_greedy, random, hover, llm_aai)
5. 결과 병합: `evaluation/{exp}/_combined/sweep_*/`
6. 시각화: system_diagram, energy_curves, episode

### 6.4 LLM 키 체크
ANTHROPIC / GEMINI / OPENAI / DEEPSEEK 중 하나라도 정의되어 있으면 llm_aai 평가 실행.

---

## 7. 시행착오 history

### 7.1 Reward 함수 변천 (uav_env.py)

| 버전 | lam1 | lam2 | lam3 | lam4 | lam5 | lam6 | lam7 | det% | 비고 |
|---|---|---|---|---|---|---|---|---|---|
| run1 | 1 | 10 | 1 | 1 | 5 | 5 | - | 8.1% | obs=17, lr=5e-4 |
| run2 | 1 | 10 | 1 | 1 | 5 | 5 | - | 13.8% | obs=19 (alpha 추가) |
| run3 | 1 | 10 | 1 | 1 | 5 | 5 | - | 12.1% | per-target approach |
| run4 | 1 | 10 | 1 | 1 | 5 | 5 | - | 21.0% | lr=1e-4, ppo_epoch=5 |
| run1* | 1 | 10 | 0.5 | 1 | 5 | 5 | 20 | 28.6% | 2M steps, lam7 신규 |
| **현재** | **2** | **15** | **0.5** | **0** | **8** | **8** | **15** | ≥70% 목표 | lam4 삭제 |

### 7.2 PPO 하이퍼파라미터 변천

| 버전 | lr | ppo_epoch | entropy_coef | max_grad_norm |
|---|---|---|---|---|
| Original | 5e-4 | 15 | 0.01 | 10 |
| run1~3 | 5e-4 | 15 | 0.05 | 10 |
| run4 시도 | 1e-4 | 5 | 0.05 | 10 |
| **현재** | **1e-4** | **5** | **0.05** | **10** |

run4에서 lr=1e-4 + ppo_epoch=5 조합이 21% 달성 → 확정.

### 7.3 초기화 변천 (reset)

| 버전 | 설명 |
|---|---|
| Original | UAV: (-50, 50) 랜덤, Target: (-300, 300) 랜덤 → 평균 300m+ 떨어져 초기 detection 실패 |
| **현재** | UAV를 타겟 근처(반경 8m + 1대는 바로 위)에 round-robin 배치 → 초기 detection 보장 |

### 7.4 기타 인프라
- Phase 1: 평가 reward sum 통일, randomize_aai CLI, obs shape mismatch skip, tracking_err metric 추가
- Phase 3: optimizer/value_normalizer save/restore
- Phase 4: train randomize_aai=False

---

## 8. 학습 모니터링

### 8.1 tensorboard
```cmd
tensorboard --logdir onpolicy\scripts\results\UAV\uav_tracking\mappo\nalpari_v1\latest\logs
```

### 8.2 정상 학습 신호
- `average_episode_rewards`: 우상향
- `dist_entropy`: **2~4 유지** (5 초과 = 폭주, 0 미만 = 붕괴)
- `value_loss`: 점진적 감소
- `eval_average_episode_rewards`: train과 같은 방향

### 8.3 진단
| 증상 | 가능한 원인/대처 |
|---|---|
| entropy 폭주 (>5) | lr 낮춤 또는 entropy_coef 낮춤 |
| entropy 붕괴 (<0) | entropy_coef 더 올림, reward 신호 강화 필요 |
| reward 정체 | lam 재조정 또는 학습 더 길게 |
| reward 떨어지다 회복 안 됨 | best 시점 checkpoint 별도 저장 메커니즘 필요 |

### 8.4 절전 방지 (Windows)
```cmd
powercfg /change standby-timeout-ac 0
powercfg /change hibernate-timeout-ac 0
```

---

## 9. 평가 결과 해석

### 핵심 비교 지표 (`evaluation/nalpari_v1/_combined/sweep_*/`)
- `reward_mean`: episode 총합 (훈련 metric과 직접 비교 가능)
- `detection_rate`: alpha=1 비율 (높을수록 좋음)
- `collision_count`: episode당 평균 충돌 (낮을수록 좋음)
- `energy_total_mean`: episode당 총 에너지 (효율성)
- `tracking_err_mean`: **실제 추정 오차 m 단위** (F_kt 대체 보조 metric)
- `F_kt_mean`: paper 공식 (saturation으로 ≈0 평탄, 변별력 X)

### 논문 claim 검증 기준
MAPPO+AAI가 naive_greedy 대비:
- 충돌 90% 이상 감소 → 목표
- 탐지율 유사 (>90%) → 학습 후 검증
- 에너지 유사 또는 우수 → 학습 후 검증

---

## 10. 알려진 이슈

### 10.1 F_kt 공식 수정
초기 구현에서 F_kt가 모든 baseline에서 ≈0으로 saturate되어 변별력이 없었음 → 표준 PCRLB 공식으로 수정. 자세한 내용은 섹션 12 참고. 보조 metric인 `tracking_err_mean`은 계속 같이 측정한다.

### 10.2 sweep_target T≠2 skip
학습은 T=2 고정이라 obs_shape=19. T=1 (obs=11) / T=3 (obs=27) 환경에선 actor 로드 실패. MAPPO 기반 baseline 자동 SKIP. naive/random/hover는 모든 T에서 평가.

### 10.3 초기화 의존성
새 reset 로직은 UAV를 타겟 근처에 강제 배치한다. 만약 베이스라인(naive)이 너무 잘 나오면 이 강제 배치 때문일 수 있다. 비교 의도가 "초기 detection 보장 상태에서 누가 잘 추적하는가"인지 명확히 해야 한다.

---

## 11. 빠른 실행

### 학습 + 평가 + 시각화 통합
```cmd
run_full_pipeline.bat
```
메뉴: `2` (fresh) 또는 `1` (skip training)

### 평가만 (학습된 모델로)
```cmd
run_full_pipeline.bat
:: 메뉴에서 1 선택
```

### LLM AAI 평가
```cmd
:: Gemini 예시
set GEMINI_API_KEY=AIza...
set LLM_AAI_PROVIDER=gemini
set LLM_AAI_MODEL=gemini-2.0-flash
run_full_pipeline.bat
```

---

## 12. 논문에서 벗어난 부분 — F_kt 계산식

논문 식 (26)을 그대로 구현했을 때 모든 baseline과 모든 시점에서 `F_kt ≈ 0`으로 saturate되어 추적 정확도를 비교하는 metric으로 작동하지 않았다. 이를 해결하기 위해 표준 PCRLB 재귀 형태로 다음과 같이 수정해서 구현했다.

**논문 식 (26):**
```
J(S_{k,t}) = [F⁻¹]ᵀ J(S_{k,t-1}) F⁻¹ + Q⁻¹ + Σ α HᵀR⁻¹H
```

**구현 (표준 PCRLB):**
```
J(S_{k,t}|k-1) = [F · J(S_{k,t-1})⁻¹ · Fᵀ + Q]⁻¹
J(S_{k,t})    = J(S_{k,t}|k-1) + Σ α HᵀR⁻¹H
F_kt          = trace(Λ · J(S_{k,t})⁻¹ · Λᵀ)
```

Λ = diag([1,1,0,0]) — 위치 성분만 (속도 오차 제외). 이 형태는 측정이 들어오면 J가 증가(불확실성 감소), 측정이 없으면 J가 감소(불확실성 증가)하여 정책별 추적 성능 차이가 F_kt에 반영된다.
