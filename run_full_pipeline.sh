#!/usr/bin/env bash
# run_full_pipeline.sh
# NALPARI: train -> eval -> visualize (Linux)
# Usage: bash run_full_pipeline.sh
# Prerequisites:
#   conda activate marl
#   export ANTHROPIC_API_KEY=sk-ant-...  (optional, for LLM AAI)

set -e
cd "$(dirname "$0")"

# ============================================================
# Config  (edit here)
# ============================================================
EXP_NAME="nalpari_v1"
NUM_AGENTS=5
NUM_ENV_STEPS=5000000
N_ROLLOUT=16          # 128코어 / RTX 3090 기준 최적값
N_SEEDS=3
N_EPISODES=3

export PYTHONPATH="$(pwd)"
export KMP_DUPLICATE_LIB_OK=TRUE
export CUDA_VISIBLE_DEVICES=0

# Result paths
RESULTS_ROOT="onpolicy/scripts/results"
TRAIN_RESULTS="${RESULTS_ROOT}/UAV/uav_tracking/mappo/${EXP_NAME}"
CHECKPOINT_DIR="${TRAIN_RESULTS}/latest/models"
TRAIN_DONE="${TRAIN_RESULTS}/latest/training_done.txt"
EVAL_OUT="evaluation/${EXP_NAME}"
FIG_OUT="figures/${EXP_NAME}"

# Sweep settings
SWEEP_UAV="3 5 7 10"
SWEEP_TARGET="1 2 3 4"
SWEEP_NOISE="1 5 20 50"
SWEEP_TAU="0.3 0.5 0.7"
EPISODE_SEEDS="42 100 200"

# PPO 안정화 옵션
STABLE_OPTS="--lr 1e-4 --entropy_coef 0.05 --ppo_epoch 5 --num_mini_batch 4"

# ============================================================
# 유틸
# ============================================================
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}$*${NC}"; }
warn()  { echo -e "${YELLOW}$*${NC}"; }
error() { echo -e "${RED}$*${NC}"; }

# ============================================================
# 0. Checkpoint detection
# ============================================================
echo ""
echo "============================================================"
echo "  NALPARI Full Pipeline"
echo "  Experiment: ${EXP_NAME}"
echo "============================================================"
echo ""

if [ ! -f "${CHECKPOINT_DIR}/actor.pt" ]; then
    info "[check] No training checkpoint found. Fresh start required."
    MENU_SKIP=1
    DO_FRESH=1
else
    if [ -f "${TRAIN_DONE}" ]; then
        info "[check] Training checkpoint found AND training is complete."
    else
        warn "[check] Training checkpoint found but training is NOT complete."
    fi

    echo ""
    echo "---"
    echo "What would you like to do?"
    if [ -f "${TRAIN_DONE}" ]; then
        echo "  [1] Skip training (already done), go to evaluation"
    else
        echo "  [1] Resume training from checkpoint"
    fi
    echo "  [2] Archive existing results and start fresh"
    echo "---"
    read -rp "Enter choice (1 or 2): " MENU_CHOICE

    if [ "${MENU_CHOICE}" = "2" ]; then
        DO_FRESH=1
        DO_ARCHIVE=1
    else
        DO_FRESH=0
        DO_ARCHIVE=0
    fi
fi

# ============================================================
# Archive
# ============================================================
if [ "${DO_ARCHIVE}" = "1" ]; then
    info "[archive] Archiving existing results..."
    DATESTAMP=$(date +%Y-%m-%d)
    ARCHIVE_BASE="_archive/${EXP_NAME}_${DATESTAMP}"
    ARCHIVE_DIR="${ARCHIVE_BASE}"
    IDX=1
    while [ -d "${ARCHIVE_DIR}" ]; do
        ARCHIVE_DIR="${ARCHIVE_BASE}_${IDX}"
        IDX=$((IDX + 1))
    done
    mkdir -p "_archive"
    [ -d "${TRAIN_RESULTS}" ] && mv "${TRAIN_RESULTS}" "${ARCHIVE_DIR}" \
        && info "[archive] Training results -> ${ARCHIVE_DIR}"
    [ -d "${EVAL_OUT}" ] && mv "${EVAL_OUT}" "${ARCHIVE_DIR}/evaluation" 2>/dev/null || true
    [ -d "${FIG_OUT}" ]  && mv "${FIG_OUT}"  "${ARCHIVE_DIR}/figures"    2>/dev/null || true
    info "[archive] Done."
fi

# ============================================================
# 1. Training
# ============================================================
if [ -f "${TRAIN_DONE}" ] && [ "${DO_FRESH}" != "1" ]; then
    info "[1/4] Skipping training (already complete)."
else
    COMMON_TRAIN_OPTS="--env_name UAV \
        --scenario_name uav_tracking \
        --algorithm_name mappo \
        --experiment_name ${EXP_NAME} \
        --num_agents ${NUM_AGENTS} \
        --num_env_steps ${NUM_ENV_STEPS} \
        --n_rollout_threads ${N_ROLLOUT} \
        --episode_length 150 \
        --use_eval \
        ${STABLE_OPTS}"

    if [ "${DO_FRESH}" = "1" ]; then
        info "[1/4] Starting fresh training... (this may take hours)"
        TRAIN_OPTS="${COMMON_TRAIN_OPTS}"
    else
        info "[1/4] Resuming training from checkpoint..."
        TRAIN_OPTS="${COMMON_TRAIN_OPTS} --resume_from ${CHECKPOINT_DIR}"
    fi

    python onpolicy/scripts/train/train_uav.py ${TRAIN_OPTS}
    if [ $? -ne 0 ]; then
        error "[ERROR] Training failed."
        exit 1
    fi
    info "[1/4] Training completed."
fi

# ============================================================
# 2. Evaluation
# ============================================================
info ""
info "[2/4] Evaluating baselines..."

run_eval() {
    local BASELINE=$1
    local OUT_DIR=$2
    shift 2
    info "  [eval] ${BASELINE}"
    python tools/evaluate_trained.py \
        --baseline "${BASELINE}" \
        --out_dir "${OUT_DIR}" \
        "$@" \
        --sweep_uav ${SWEEP_UAV} \
        --sweep_target ${SWEEP_TARGET} \
        --sweep_noise ${SWEEP_NOISE} \
        --sweep_tau ${SWEEP_TAU} \
        --save_episode \
        --episode_seeds ${EPISODE_SEEDS} \
        --n_seeds ${N_SEEDS} \
        --n_episodes ${N_EPISODES}
    if [ $? -ne 0 ]; then
        error "[ERROR] ${BASELINE} eval failed."
        exit 1
    fi
}

CKPT_OPTS="--checkpoint_dir ${CHECKPOINT_DIR}"

run_eval mappo+aai   "${EVAL_OUT}/mappo_aai"   ${CKPT_OPTS}
run_eval mappo       "${EVAL_OUT}/mappo_only"   ${CKPT_OPTS}
run_eval naive_greedy "${EVAL_OUT}/naive"
run_eval random      "${EVAL_OUT}/random"
run_eval hover       "${EVAL_OUT}/hover"

# LLM AAI (API 키 있을 때만)
LLM_KEY_SET=0
[ -n "${ANTHROPIC_API_KEY}" ] && LLM_KEY_SET=1
[ -n "${OPENAI_API_KEY}" ]    && LLM_KEY_SET=1
[ -n "${DEEPSEEK_API_KEY}" ]  && LLM_KEY_SET=1
[ -n "${GEMINI_API_KEY}" ]    && LLM_KEY_SET=1

if [ "${LLM_KEY_SET}" = "0" ]; then
    warn "  [SKIP] llm_aai - no LLM API key set."
else
    # provider 자동 감지 (우선순위: deepseek > anthropic > openai > gemini)
    if   [ -n "${DEEPSEEK_API_KEY}" ];  then LLM_AAI_PROVIDER=deepseek;   LLM_AAI_MODEL=deepseek-chat
    elif [ -n "${ANTHROPIC_API_KEY}" ]; then LLM_AAI_PROVIDER=anthropic;  LLM_AAI_MODEL=claude-sonnet-4-6
    elif [ -n "${OPENAI_API_KEY}" ];    then LLM_AAI_PROVIDER=openai;      LLM_AAI_MODEL=gpt-4o
    elif [ -n "${GEMINI_API_KEY}" ];    then LLM_AAI_PROVIDER=gemini;      LLM_AAI_MODEL=gemini-2.0-flash
    fi
    export LLM_AAI_PROVIDER LLM_AAI_MODEL
    info "  [LLM] provider=${LLM_AAI_PROVIDER}  model=${LLM_AAI_MODEL}"
    run_eval llm_aai "${EVAL_OUT}/llm_aai" \
        ${CKPT_OPTS} \
        --aai_callback tools.llm_aai:default_callback
fi

# ============================================================
# 3. Merge results
# ============================================================
info ""
info "[3/4] Merging results..."
COMBINED="${EVAL_OUT}/_combined"
for SWEEP in sweep_uav sweep_target sweep_noise sweep_tau; do
    mkdir -p "${COMBINED}/${SWEEP}"
    for VARIANT in mappo_aai mappo_only naive random hover llm_aai; do
        SRC="${EVAL_OUT}/${VARIANT}/${SWEEP}"
        [ -d "${SRC}" ] && cp "${SRC}"/*.npz "${COMBINED}/${SWEEP}/" 2>/dev/null || true
    done
done

# ============================================================
# 4. Visualization
# ============================================================
info ""
info "[4/4] Generating figures..."
mkdir -p "${FIG_OUT}"

python tools/plot_system_diagram.py --out_dir "${FIG_OUT}"
python tools/plot_energy_curves.py --results_dir "${COMBINED}" --out_dir "${FIG_OUT}"

for VARIANT in mappo_aai mappo_only naive llm_aai; do
    EP_DIR="${EVAL_OUT}/${VARIANT}/episodes"
    if [ -d "${EP_DIR}" ]; then
        for EP_FILE in "${EP_DIR}"/*.npz; do
            [ -f "${EP_FILE}" ] || continue
            BASE=$(basename "${EP_FILE}" .npz)
            python tools/plot_episode.py \
                --episode_npz "${EP_FILE}" \
                --out_path "${FIG_OUT}/episode_${VARIANT}_${BASE}.png"
        done
    fi
done

# ============================================================
# Done
# ============================================================
echo ""
info "[done] All artifacts:"
info "  Training  : ${TRAIN_RESULTS}"
info "  Evaluation: ${EVAL_OUT}"
info "  Figures   : ${FIG_OUT}"