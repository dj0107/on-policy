# 날파리 (Nalpari) — 통합 파이프라인

**현재 진행 상태**: Step 4 (코드 재점검) → Step 2 (시각화) → Step 3 (대조군) → Step 1 (LLM 도입) **모두 완료**.

학습은 GPU 환경에서 직접 돌려야 합니다. 평가/시각화는 학습된 체크포인트만 있으면 즉시 동작.

---

## 디렉토리 구조

```
onpolicy/                       # 기존 MAPPO 코드 + 수정 사항
  envs/uav/uav_env.py           # ★ Step 4: 재점검된 환경
  runner/shared/uav_runner.py   # ★ run_meta.json/all_args.pkl 자동 저장 추가
  scripts/train/train_uav.py    # 그대로 사용

tools/                          # ★ 신규: 평가 + 시각화 도구
  run_evaluation.py             # baseline 정책 + episode runner (내부 라이브러리)
  evaluate_trained.py           # CLI: 학습된 모델로 sweep 평가
  llm_aai.py                    # ★ Step 1: LLM-based AAI 모듈
  test_llm_call.py              # LLM prompt/response 단독 검증
  plot_energy_curves.py         # ★ Step 2: 회의 요구 그래프 (y=energy)
  plot_episode.py               # ★ Step 2: 단일 episode 멀티패널 (교수님 피드백)
  plot_system_diagram.py        # ★ Step 2: 시스템 작동 그림

run_full_pipeline.sh            # 학습 → 평가 → 시각화 한 번에
figures/                        # 예시 출력
```

---

## Step 4: 코드 재점검 — 핵심 변경사항

### A. 논문 정합성
- SNR 임계값 `13 dB` 명시적 선형 변환 (`10**(13/10) ≈ 19.95`).
- 식 (28b)(28c) 강제: 타겟 경계 반사 + UAV 경계 클리핑.
- 식 (28f) `Σ_u α ≥ 1` 위반 페널티 신설 (`λ5=5`).
- α=0일 때 통신 에너지 1%만 소비 ("absence signal").

### B. 학습 안정성
- 보상 가중치 재조정: `λ3` 100→10 (충돌이 학습 신호 압도하던 문제 해결, std 21.8→2.2).
- 측정 잡음 분산 floor: `σ_r²_floor=1e-3`, `σ_θ²_floor=1e-7`. BFIM 폭발 방지.
- F_kt cap (1000), 보상 cap ([-1000, 1000]), 타겟 속도 cap (8 m/s).

### C. 새 기능 (`UAVTrackingEnv` 추가 인자)
- `use_aai=False`: 대조군용 (W=1, ε=5, p=15 고정)
- `aai_callback`: 평가 시 LLM AAI 주입 hook
- `randomize_aai`: 학습 시 W/ε/p 랜덤 → robust policy
- `log_episode=True`: 시각화/평가용 14개 항목 시계열 로깅

### 학습 전략 답변
**MAPPO만으로 학습 → 평가 시 AAI 추가**가 정답입니다.
- AAI 출력은 정책 입장에선 외생 파라미터(observation 일부).
- 학습 시 `randomize_aai=True`로 다양한 (W,ε,p) 조합 노출 → robust policy.
- 평가 시 LLM이 합리적 값을 골라줘도 OOD 문제 X (학습 분포 내).
- AAI를 학습 루프에 넣으면 LLM 호출 비용으로 sample efficiency 박살남.

---

## 사용법

### 1. 학습 (GPU 권장, 수 시간~수 일)

```bash
cd onpolicy/scripts/train
python train_uav.py \
    --env_name UAV \
    --scenario_name uav_tracking \
    --algorithm_name mappo \
    --experiment_name nalpari_v1 \
    --num_agents 5 \
    --num_env_steps 2000000 \
    --n_rollout_threads 4 \
    --use_eval
```

체크포인트가 다음 위치에 저장됨:
```
onpolicy/scripts/results/UAV/uav_tracking/mappo/nalpari_v1/run1/models/
    actor.pt            # MAPPO actor weights
    critic.pt
    run_meta.json       # ★ obs_dim/action_dim 등 (평가 시 사용)
    all_args.pkl        # ★ 학습 args 전체 (정확한 R_Actor 복원용)
```

**학습 측 추가 사항**: `uav_runner._save_run_meta()`가 학습 시작 시 자동으로 메타데이터를 저장. 이게 있어야 평가 측에서 actor를 정확히 재구성할 수 있음.

### 2. 평가 (학습된 정책 + 4가지 baseline)

#### 2-a. 학습된 MAPPO + heuristic AAI (메인 결과)
```bash
python tools/evaluate_trained.py \
    --checkpoint_dir path/to/run1/models \
    --baseline mappo+aai \
    --out_dir evaluation/mappo_aai \
    --sweep_uav 3 5 7 10 \
    --sweep_target 1 2 3 4 \
    --sweep_noise 1 5 20 50 \
    --sweep_tau 0.3 0.5 0.7 \
    --save_episode --episode_seeds 42 100 200 \
    --n_seeds 3 --n_episodes 3
```

#### 2-b. 학습된 MAPPO + AAI 무사용 (대조군 — AAI 효과 측정)
```bash
python tools/evaluate_trained.py \
    --checkpoint_dir path/to/run1/models \
    --baseline mappo \
    --out_dir evaluation/mappo_only \
    [...sweep options 동일...]
```

#### 2-c. Naive greedy / Random / Hover (학습 X 베이스라인)
```bash
python tools/evaluate_trained.py --baseline naive_greedy --out_dir evaluation/naive [...]
python tools/evaluate_trained.py --baseline random       --out_dir evaluation/random [...]
python tools/evaluate_trained.py --baseline hover        --out_dir evaluation/hover  [...]
```

#### 2-d. 학습된 MAPPO + LLM AAI (Step 1 결과)
```bash
export ANTHROPIC_API_KEY=sk-ant-...

python tools/evaluate_trained.py \
    --checkpoint_dir path/to/run1/models \
    --baseline llm_aai \
    --aai_callback tools.llm_aai:default_callback \
    --out_dir evaluation/llm_aai \
    [...]
```

LLM AAI 환경 변수:
- `LLM_AAI_MODEL` (default: `claude-sonnet-4-5`)
- `LLM_AAI_PROVIDER` (anthropic / openai / mock)
- `LLM_AAI_CALL_EVERY` (default 10 — 매 K step마다 LLM 호출. 사이엔 cache+last reuse)
- `LLM_AAI_VERBOSE=1`로 디버그 출력

### 3. 시각화

#### 3-a. 시스템 작동 그림 (회의 요구사항)
```bash
python tools/plot_system_diagram.py --out_dir figures/
```
→ `system_architecture.png` (3-layer 계층 구조)
→ `timestep_flow.png` (1 time slot 7-step 절차)

#### 3-b. Sweep 그래프 (회의 요구사항: y=에너지, x=조건)
여러 method 결과를 한 디렉토리에 모아서:
```bash
mkdir -p combined/{sweep_uav,sweep_target,sweep_noise,sweep_tau}
for sweep in sweep_uav sweep_target sweep_noise sweep_tau; do
    cp evaluation/mappo_aai/$sweep/*.npz combined/$sweep/ 2>/dev/null
    cp evaluation/mappo_only/$sweep/*.npz combined/$sweep/ 2>/dev/null
    cp evaluation/naive/$sweep/*.npz combined/$sweep/ 2>/dev/null
    cp evaluation/random/$sweep/*.npz combined/$sweep/ 2>/dev/null
    cp evaluation/llm_aai/$sweep/*.npz combined/$sweep/ 2>/dev/null
done

python tools/plot_energy_curves.py --results_dir combined --out_dir figures/
```
→ `energy_overview.png` (4 sweep 한눈에)
→ `sweep_uav_detailed.png` 등 (각 sweep × 3 패널: total energy / detection+collision / per-step curve)

#### 3-c. 단일 episode 시각화 (교수님 피드백: realistic setting 시각적 표현)
```bash
python tools/plot_episode.py \
    --episode_npz evaluation/mappo_aai/episodes/mappo+aai_seed42.npz \
    --out_path figures/episode_mappo_aai_seed42.png
```
→ Top-down trajectory (UAV/타겟/위험구역/BS, fade-in)
+ Energy timeline (UAV별 stacked)
+ F_kt timeline + AAI 시계열 (W/p) + Detection α 히트맵
+ 5 time-lapse snapshots

### 4. 한 번에 돌리기
```bash
chmod +x run_full_pipeline.sh
./run_full_pipeline.sh
```

---

## 결정 대기 사항

회의에서 나온 x축 후보 중 **noise power와 sensing/comm 비율**은 현재 setting (안테나 게인 매우 강함, detection rate 항상 100%) 에서 그래프가 평평하게 나올 가능성이 높음. 의미 있는 변동을 만들려면:
- 옵션 A: 맵 1500~2000m로 확장
- 옵션 B: 안테나 게인 ↓
- 옵션 C: 현재 setting 유지 (에너지 절약+충돌 회피만 학습 신호)

신동주님 결정 후 환경 생성 인자만 바꾸면 됨 (코드 수정 X).

---

## 알려진 이슈 / TODO

- **검증된 흐름**: random/naive/hover/mock-mappo/llm_aai-mock 모두 평가 → 시각화까지 동작 확인.
- **검증 안 된 부분**: 실제 GPU 학습은 이 환경에서 못 돌림. 학습 코드 자체는 기존 MAPPO 그대로(검증된 코드)이고, 새로 추가한 부분은 `_save_run_meta` 한 메서드뿐.
- **LLM 실제 호출**: ANTHROPIC_API_KEY가 환경에 없어서 실제 API 호출 검증 못함. mock 모드에서는 throttling/cache/fallback 흐름 모두 동작 확인.
- **eps_kt 사용처**: 현재 reward에는 직접 들어가지 않음 (식 (32) 참고). 논문도 eps_kt는 제약 (28h)에서만 사용. obs에는 들어가서 정책이 인지함.

## 다음 작업 (선순위 순)

1. **학습 setting 결정** (질문에 대한 답 후) — 안테나 게인/맵 크기 등
2. **학습 1회 돌리기** (GPU) → checkpoint 확보
3. **5가지 baseline 평가** (`run_full_pipeline.sh`)
4. **회의/지도교수 보고용 figure 정리** (`figures/`)
5. (선택) **LLM AAI vs heuristic AAI 비교** — Step 1의 정량적 의의 측정
