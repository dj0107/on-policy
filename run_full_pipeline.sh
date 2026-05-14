#!/usr/bin/env bash
# run_full_pipeline.sh
# ====================
# 학습 → 평가 → 시각화 전체 파이프라인 예시.
# 각 단계는 독립 실행 가능 (학습 후 며칠 뒤 평가만 다시 돌려도 OK).
#
# 사용 전:
#   - 학습 환경에 conda env 활성화 (environment.yaml 참고)
#   - 평가 시점에 ANTHROPIC_API_KEY 설정 (LLM AAI 사용 시)
#
# Usage:
#   chmod +x run_full_pipeline.sh
#   ./run_full_pipeline.sh

set -e

# ============================================================================
# 설정
# ============================================================================
EXP_NAME="nalpari_v1"
NUM_AGENTS=5
NUM_ENV_STEPS=2000000
N_ROLLOUT=4

# 결과물 위치
TRAIN_RESULTS="onpolicy/scripts/results/UAV/uav_tracking/mappo/${EXP_NAME}"
CHECKPOINT_DIR="${TRAIN_RESULTS}/run1/models"
EVAL_OUT="evaluation/${EXP_NAME}"
FIG_OUT="figures/${EXP_NAME}"

# Sweep condition (현재 setting에선 noise/tau는 평평할 가능성)
SWEEP_UAV="3 5 7 10"
SWEEP_TARGET="1 2 3 4"
SWEEP_NOISE="1 5 20 50"
SWEEP_TAU="0.3 0.5 0.7"

EPISODE_SEEDS="42 100 200"
N_SEEDS=3
N_EPISODES=3


# ============================================================================
# 1. 학습 (시간이 많이 걸림 - GPU 권장)
# ============================================================================
if [ ! -f "${CHECKPOINT_DIR}/actor.pt" ]; then
    echo "[1/4] Training MAPPO..."
    cd onpolicy/scripts/train
    python train_uav.py \
        --env_name UAV \
        --scenario_name uav_tracking \
        --algorithm_name mappo \
        --experiment_name "${EXP_NAME}" \
        --num_agents ${NUM_AGENTS} \
        --num_env_steps ${NUM_ENV_STEPS} \
        --n_rollout_threads ${N_ROLLOUT} \
        --episode_length 100 \
        --use_eval
    cd ../../..
else
    echo "[1/4] Skipping training (checkpoint exists)."
fi

# ============================================================================
# 2. 평가 — 5가지 baseline (학습된 정책 + LLM 변형 포함)
# ============================================================================
echo "[2/4] Evaluating baselines..."

# 2-a. 학습된 MAPPO + heuristic AAI
python tools/evaluate_trained.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --baseline mappo+aai \
    --out_dir "${EVAL_OUT}/mappo_aai" \
    --sweep_uav ${SWEEP_UAV} \
    --sweep_target ${SWEEP_TARGET} \
    --sweep_noise ${SWEEP_NOISE} \
    --sweep_tau ${SWEEP_TAU} \
    --save_episode --episode_seeds ${EPISODE_SEEDS} \
    --n_seeds ${N_SEEDS} --n_episodes ${N_EPISODES}

# 2-b. 학습된 MAPPO + AAI 무사용 (대조군)
python tools/evaluate_trained.py \
    --checkpoint_dir "${CHECKPOINT_DIR}" \
    --baseline mappo \
    --out_dir "${EVAL_OUT}/mappo_only" \
    --sweep_uav ${SWEEP_UAV} \
    --sweep_target ${SWEEP_TARGET} \
    --sweep_noise ${SWEEP_NOISE} \
    --sweep_tau ${SWEEP_TAU} \
    --save_episode --episode_seeds ${EPISODE_SEEDS} \
    --n_seeds ${N_SEEDS} --n_episodes ${N_EPISODES}

# 2-c. Naive greedy (학습 X)
python tools/evaluate_trained.py \
    --baseline naive_greedy \
    --out_dir "${EVAL_OUT}/naive" \
    --sweep_uav ${SWEEP_UAV} \
    --sweep_target ${SWEEP_TARGET} \
    --sweep_noise ${SWEEP_NOISE} \
    --sweep_tau ${SWEEP_TAU} \
    --save_episode --episode_seeds ${EPISODE_SEEDS} \
    --n_seeds ${N_SEEDS} --n_episodes ${N_EPISODES}

# 2-d. Random (학습 X)
python tools/evaluate_trained.py \
    --baseline random \
    --out_dir "${EVAL_OUT}/random" \
    --sweep_uav ${SWEEP_UAV} \
    --sweep_target ${SWEEP_TARGET} \
    --sweep_noise ${SWEEP_NOISE} \
    --sweep_tau ${SWEEP_TAU} \
    --save_episode --episode_seeds ${EPISODE_SEEDS} \
    --n_seeds ${N_SEEDS} --n_episodes ${N_EPISODES}

# 2-e. (선택) 학습된 MAPPO + LLM AAI
if [ -n "${ANTHROPIC_API_KEY}" ]; then
    python tools/evaluate_trained.py \
        --checkpoint_dir "${CHECKPOINT_DIR}" \
        --baseline llm_aai \
        --aai_callback tools.llm_aai:default_callback \
        --out_dir "${EVAL_OUT}/llm_aai" \
        --sweep_uav ${SWEEP_UAV} \
        --sweep_target ${SWEEP_TARGET} \
        --save_episode --episode_seeds ${EPISODE_SEEDS} \
        --n_seeds ${N_SEEDS} --n_episodes ${N_EPISODES}
else
    echo "  [info] ANTHROPIC_API_KEY not set; skipping LLM AAI eval."
fi

# ============================================================================
# 3. 결과 합치기 (한 sweep dir에 method별 npz 모두 모음)
# ============================================================================
echo "[3/4] Merging results..."
COMBINED="${EVAL_OUT}/_combined"
for sweep_name in sweep_uav sweep_target sweep_noise sweep_tau; do
    mkdir -p "${COMBINED}/${sweep_name}"
    for variant in mappo_aai mappo_only naive random llm_aai; do
        src="${EVAL_OUT}/${variant}/${sweep_name}"
        if [ -d "${src}" ]; then
            cp ${src}/*.npz "${COMBINED}/${sweep_name}/" 2>/dev/null || true
        fi
    done
done

# ============================================================================
# 4. 시각화
# ============================================================================
echo "[4/4] Generating figures..."
mkdir -p "${FIG_OUT}"

# 4-a. 시스템 다이어그램 (한 번만 그리면 됨)
python tools/plot_system_diagram.py --out_dir "${FIG_OUT}"

# 4-b. Sweep 그래프
python tools/plot_energy_curves.py \
    --results_dir "${COMBINED}" \
    --out_dir "${FIG_OUT}"

# 4-c. 단일 episode 시각화 (각 method 대표 episode)
for variant in mappo_aai mappo_only naive llm_aai; do
    ep_dir="${EVAL_OUT}/${variant}/episodes"
    if [ -d "${ep_dir}" ]; then
        for ep_file in ${ep_dir}/*.npz; do
            [ -f "${ep_file}" ] || continue
            base=$(basename "${ep_file}" .npz)
            python tools/plot_episode.py \
                --episode_npz "${ep_file}" \
                --out_path "${FIG_OUT}/episode_${variant}_${base}.png"
        done
    fi
done

echo ""
echo "[done] All artifacts:"
echo "  Training:   ${TRAIN_RESULTS}"
echo "  Evaluation: ${EVAL_OUT}"
echo "  Figures:    ${FIG_OUT}"
