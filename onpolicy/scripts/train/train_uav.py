#!/usr/bin/env python
import sys
import os
import socket
import setproctitle
import numpy as np
from pathlib import Path
import torch

# 프로젝트 루트를 sys.path에 추가 (배치 실행 시 PYTHONPATH 문제 해결)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from onpolicy.config import get_config
from onpolicy.envs.uav.uav_env import UAVTrackingEnv
from onpolicy.envs.uav.uav_vec_env import UAVDummyVecEnv
from onpolicy.runner.shared.uav_runner import UAVRunner

"""Train script for UAV Tracking."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = UAVTrackingEnv(num_uavs=all_args.num_agents, num_targets=2,
                                 randomize_aai=True)  # AAI domain randomization → LLM 값 범위에 일반화
            env.seed(all_args.seed + rank * 1000)
            return env
        return init_env
    # 일단 UAVDummyVecEnv만 지원 (SubprocVec은 multiprocessing 호환 추후 작업)
    return UAVDummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env = UAVTrackingEnv(num_uavs=all_args.num_agents, num_targets=2)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env
        return init_env
    return UAVDummyVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument('--scenario_name', type=str,
                        default='uav_tracking', help="Which scenario to run on")
    parser.add_argument('--num_agents', type=int,
                        default=5, help="number of UAV agents")
    parser.add_argument('--resume_from', type=str, default=None,
                        help="Path to checkpoint dir to resume training from "
                             "(e.g. results/UAV/uav_tracking/mappo/nalpari_v1/run1/models)")
    all_args = parser.parse_known_args(args)[0]
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        all_args.use_recurrent_policy = True
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "mappo":
        all_args.use_recurrent_policy = False
        all_args.use_naive_recurrent_policy = False
    elif all_args.algorithm_name == "ippo":
        all_args.use_centralized_V = False
    else:
        raise NotImplementedError

    # cuda
    if all_args.cuda and torch.cuda.is_available():
        print("choose to use gpu...")
        device = torch.device("cuda:0")
        torch.set_num_threads(all_args.n_training_threads)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    else:
        print("choose to use cpu...")
        device = torch.device("cpu")
        torch.set_num_threads(all_args.n_training_threads)

    # run dir
    if all_args.resume_from is not None:
        # resume: keep the same run directory
        run_dir = Path(all_args.resume_from)
        # resume_from points to .../models/, so run_dir is its parent
        if run_dir.name == 'models':
            all_args.model_dir = str(run_dir)
            run_dir = run_dir.parent
        else:
            all_args.model_dir = str(run_dir / 'models')
        print(f"[resume] Using existing run directory: {run_dir}")
        print(f"[resume] Will restore from: {all_args.model_dir}")
    else:
        run_dir = Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results") \
                  / all_args.env_name / all_args.scenario_name / all_args.algorithm_name / all_args.experiment_name
        if not run_dir.exists():
            os.makedirs(str(run_dir))

        if all_args.use_wandb:
            import wandb
            run = wandb.init(config=all_args,
                             project=all_args.env_name,
                             entity=all_args.user_name,
                             notes=socket.gethostname(),
                             name=str(all_args.algorithm_name) + "_" +
                                  str(all_args.experiment_name) +
                                  "_seed" + str(all_args.seed),
                             group=all_args.scenario_name,
                             dir=str(run_dir),
                             job_type="training",
                             reinit=True)
        else:
            if not run_dir.exists():
                curr_run = 'run1'
            else:
                exst_run_nums = [int(str(folder.name).split('run')[1])
                                 for folder in run_dir.iterdir()
                                 if str(folder.name).startswith('run')]
                curr_run = 'run1' if len(exst_run_nums) == 0 else 'run%i' % (max(exst_run_nums) + 1)
            run_dir = run_dir / curr_run
            if not run_dir.exists():
                os.makedirs(str(run_dir))

    setproctitle.setproctitle(str(all_args.algorithm_name) + "-" +
                              str(all_args.env_name) + "-" +
                              str(all_args.experiment_name) + "@" +
                              str(all_args.user_name))

    # seed
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    np.random.seed(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir
    }

    runner = UAVRunner(config)
    runner.run()

    # 학습 완료 플래그 저장
    done_flag = run_dir / 'training_done.txt'
    done_flag.write_text('Training completed successfully.\n')

    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    if all_args.use_wandb:
        run.finish()
    else:
        runner.writter.export_scalars_to_json(str(runner.log_dir + '/summary.json'))
        runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
