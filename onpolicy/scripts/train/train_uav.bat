@echo off
set ENV=UAV
set NUM_UAVS=5
set NUM_TARGETS=2
set ALGO=mappo
set EXP=test_run_01
set SEED=1

echo 환경: %ENV%, 알고리즘: %ALGO%, 실험명: %EXP%

:: on-policy 루트에서 PYTHONPATH 설정
:: 이 bat 파일은 on-policy\onpolicy\scripts\train\ 에 위치
cd /d %~dp0
cd ..\..\..
set PYTHONPATH=%CD%
cd onpolicy\scripts\train

python train_uav.py ^
    --env_name %ENV% ^
    --scenario_name uav_tracking ^
    --algorithm_name %ALGO% ^
    --experiment_name %EXP% ^
    --num_agents %NUM_UAVS% ^
    --num_env_steps 5000000 ^
    --episode_length 100 ^
    --n_rollout_threads 4 ^
    --ppo_epoch 15 ^
    --lr 5e-4 ^
    --critic_lr 5e-4 ^
    --seed %SEED% ^
    --use_wandb

pause
