@echo off
set ENV=UAV
set ALGO=mappo
set EXP=test_run_01
set SEED=1

cd /d %~dp0
cd ..\..\..
set PYTHONPATH=%CD%
cd onpolicy\scripts\train

python train_uav.py --env_name %ENV% --scenario_name uav_tracking --algorithm_name %ALGO% --experiment_name %EXP% --num_agents 5 --num_env_steps 5000000 --episode_length 100 --n_rollout_threads 4 --ppo_epoch 15 --lr 5e-4 --critic_lr 5e-4 --seed %SEED% --use_wandb

pause
