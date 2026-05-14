@echo off
set KMP_DUPLICATE_LIB_OK=TRUE
cd /d %~dp0
cd ..\..\..
set PYTHONPATH=%CD%
cd onpolicy\scripts\train
 
python train_uav.py --env_name UAV --scenario_name uav_tracking --algorithm_name mappo --experiment_name test_run_01 --num_agents 5 --num_env_steps 5000000 --episode_length 100 --n_rollout_threads 4 --ppo_epoch 15 --lr 5e-4 --critic_lr 5e-4 --seed 1 --use_wandb
 
pause