#!/bin/sh
env="uav"
num_uavs=3
num_targets=2
algo="mappo"
exp="test_run_01"
seed_max=1

echo "환경: ${env}, 알고리즘: ${algo}, 실험명: ${exp}, 최대 시드: ${seed_max}"

for seed in `seq ${seed_max}`;
do
    echo "현재 Seed: ${seed} 훈련 시작..."
    
    # GPU가 여러 대라면 CUDA_VISIBLE_DEVICES를 조절하세요 (0번 GPU 사용)
    CUDA_VISIBLE_DEVICES=0 python train_uav.py \
    --env_name ${env} \
    --algorithm_name ${algo} \
    --experiment_name ${exp} \
    --num_agents ${num_uavs} \
    --num_env_steps 5000000 \
    --episode_length 100 \
    --n_rollout_threads 7 \
    --ppo_epoch 15 \
    --lr 5e-4 \
    --critic_lr 5e-4 \
    --use_recurrent_policy \
    --seed ${seed}
done