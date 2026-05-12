"""
evaluate_trained.py
===================
학습이 끝난 MAPPO 체크포인트(actor.pt + run_meta.json)를 받아 다양한 condition에서
평가 + 단일 에피소드 상세 로그 저장.

대조군 (Step 3):
    --baseline mappo+aai     : 학습된 MAPPO 정책 + heuristic AAI
    --baseline mappo         : 학습된 MAPPO 정책 + AAI 무사용 (대조군)
    --baseline naive_greedy  : 학습 X, 직진 정책
    --baseline random        : 학습 X, 랜덤 정책
    --baseline hover         : 학습 X, 정지 정책
    --baseline llm_aai       : 학습된 MAPPO 정책 + LLM-based AAI (Step 1)

LLM AAI 주입:
    --aai_callback module:func    예) tools.llm_aai:gpt4_aai_callback
    callback 시그니처: callback(env) → None (env.targets, env.uavs를 직접 수정)

체크포인트 디렉토리 구조 (학습 측 uav_runner.py가 자동 생성):
    /path/to/run1/models/
        actor.pt
        critic.pt
        run_meta.json     # obs_dim, action_dim 등
        all_args.pkl      # 전체 args (정확한 R_Actor 복원에 필요)

Usage:
    python evaluate_trained.py \\
        --checkpoint_dir /path/to/run1/models \\
        --baseline mappo+aai \\
        --out_dir results/mappo_aai \\
        --sweep_uav 3 5 7 10 \\
        --sweep_target 1 2 3 4 \\
        --sweep_noise 1 5 20 50 \\
        --sweep_tau 0.3 0.5 0.7 \\
        --save_episode --episode_seeds 42 100 \\
        --n_seeds 3 --n_episodes 3
"""
import argparse
import os
import sys
import json
import pickle
import importlib
import argparse as _argparse_module
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from onpolicy.envs.uav.uav_env import UAVTrackingEnv
from tools.run_evaluation import (
    run_episode, save_episode, run_sweep,
    policy_random, policy_hover, policy_naive_greedy_factory,
)


# ============================================================================
# 학습된 MAPPO actor 로딩
# ============================================================================
def _load_run_meta(checkpoint_dir):
    """run_meta.json + all_args.pkl 로드"""
    meta_path = os.path.join(checkpoint_dir, 'run_meta.json')
    args_path = os.path.join(checkpoint_dir, 'all_args.pkl')
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(
            f'run_meta.json not found in {checkpoint_dir}. '
            f'학습 시 uav_runner._save_run_meta()가 호출되었는지 확인하세요.'
        )
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    args_dict = None
    if os.path.isfile(args_path):
        with open(args_path, 'rb') as f:
            args_dict = pickle.load(f)
    return meta, args_dict


def _build_args_namespace(args_dict, meta):
    """pickle된 args가 없으면 meta로부터 R_Actor 생성에 최소한 필요한 args 재구성"""
    if args_dict is not None:
        ns = _argparse_module.Namespace(**args_dict)
        return ns
    # fallback: 학습 시 기본값으로 가정
    ns = _argparse_module.Namespace(
        hidden_size=meta.get('hidden_size', 64),
        recurrent_N=meta.get('recurrent_N', 1),
        use_orthogonal=True,
        use_policy_active_masks=True,
        use_naive_recurrent_policy=meta.get('use_naive_recurrent_policy', False),
        use_recurrent_policy=meta.get('use_recurrent_policy', False),
        gain=0.01,
        algorithm_name=meta.get('algorithm_name', 'mappo'),
        # MLPBase에 필요한 기본 args (config.py 기본값)
        layer_N=1,
        use_ReLU=True,
        use_feature_normalization=True,
        stacked_frames=1,
        use_stacked_frames=False,
        # ACTLayer
        gain_continuous=0.01,
    )
    return ns


def load_mappo_actor(checkpoint_dir, env, device='cpu'):
    """
    학습된 actor를 정확한 dim으로 재구성하고 weight 로드.
    Returns: policy_fn(obs) → action  (numpy array, shape (U, action_dim))
    """
    import torch
    from onpolicy.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor

    meta, args_dict = _load_run_meta(checkpoint_dir)
    ns = _build_args_namespace(args_dict, meta)

    # env가 학습 시와 obs/action shape이 동일해야 함
    obs_space = env.observation_space[0]
    act_space = env.action_space[0]
    expected_obs = tuple(meta['obs_shape'])
    expected_act = tuple(meta['action_shape'])
    if tuple(obs_space.shape) != expected_obs:
        raise ValueError(
            f'obs shape mismatch: env={obs_space.shape}, checkpoint={expected_obs}.'
        )
    if tuple(act_space.shape) != expected_act:
        print(f'[WARN] action shape mismatch: env={act_space.shape}, '
              f'checkpoint={expected_act}.')

    actor = R_Actor(ns, obs_space, act_space, device=torch.device(device))
    state_dict = torch.load(os.path.join(checkpoint_dir, 'actor.pt'),
                            map_location=device)
    actor.load_state_dict(state_dict)
    actor.eval()

    use_recurrent = (meta.get('use_recurrent_policy', False) or
                     meta.get('use_naive_recurrent_policy', False))
    hidden_size = meta.get('hidden_size', 64)
    recurrent_N = meta.get('recurrent_N', 1)
    num_uavs = env.num_uavs

    # rnn states / mask는 호출자가 episode 시작 시 0으로 초기화
    state = {
        'rnn_states': np.zeros((num_uavs, recurrent_N, hidden_size), dtype=np.float32),
        'masks': np.ones((num_uavs, 1), dtype=np.float32),
    }

    @torch.no_grad()
    def policy_fn(obs):
        obs_t = torch.from_numpy(np.asarray(obs, dtype=np.float32)).to(device)
        rnn_t = torch.from_numpy(state['rnn_states']).to(device)
        masks_t = torch.from_numpy(state['masks']).to(device)
        action, _, rnn_new = actor(obs_t, rnn_t, masks_t, deterministic=True)
        state['rnn_states'] = rnn_new.cpu().numpy()
        # action: tensor (U, action_dim). Box 공간이라 그대로 numpy로
        return action.cpu().numpy().astype(np.float32)

    return policy_fn, state


# ============================================================================
# LLM-based AAI callback 로더
# ============================================================================
def load_aai_callback(spec):
    """spec: 'module.path:func_name' → callback 함수 반환"""
    if spec is None:
        return None
    if ':' not in spec:
        raise ValueError(f'aai_callback spec must be "module:func", got {spec}')
    module_path, func_name = spec.split(':', 1)
    mod = importlib.import_module(module_path)
    return getattr(mod, func_name)


# ============================================================================
# Baseline policy factory
# ============================================================================
def get_policy_factory(baseline_name, checkpoint_dir=None, device='cpu'):
    """baseline_name → policy_factory(env) → policy_fn"""
    if baseline_name == 'naive_greedy':
        return lambda env: policy_naive_greedy_factory(env)
    if baseline_name == 'random':
        return lambda env: policy_random
    if baseline_name == 'hover':
        return lambda env: policy_hover
    if baseline_name in ('mappo', 'mappo+aai', 'llm_aai'):
        if checkpoint_dir is None:
            raise ValueError(f'{baseline_name} requires --checkpoint_dir')

        def factory(env):
            try:
                policy_fn, _ = load_mappo_actor(checkpoint_dir, env, device=device)
                return policy_fn
            except ValueError as e:
                print(f'[SKIP] {e}')
                return None
        return factory
    raise ValueError(f'unknown baseline: {baseline_name}')


def make_env(num_uavs, num_targets, baseline_name, aai_callback=None,
             sigma_w_sq=5.0, randomize_aai=False, **extra):
    """
    baseline에 맞춰 use_aai/aai_callback을 결정.

    mappo+aai     → use_aai=True, aai_callback=None  (heuristic AAI)
    mappo         → use_aai=False, aai_callback=None  (대조군: AAI 무사용)
    naive_greedy  → use_aai=True (assignment 사용)
    llm_aai       → aai_callback=user-provided  (env 내부 use_aai 무시되고 callback 사용)

    randomize_aai: 훈련 시에는 True였음 (도메인 랜덤화). 평가는 기본 False로 고정 시나리오 테스트.
    extra: 추가 환경 인자 (sigma_w_sq, num_critical_zones 등)
    """
    if baseline_name == 'mappo':
        use_aai = False
        cb = None
    elif baseline_name == 'llm_aai':
        use_aai = True
        cb = aai_callback
        if cb is None:
            raise ValueError('llm_aai baseline requires --aai_callback')
    else:
        # mappo+aai, naive_greedy, random, hover 모두 heuristic AAI 사용
        use_aai = True
        cb = None

    return UAVTrackingEnv(
        num_uavs=num_uavs, num_targets=num_targets,
        use_aai=use_aai, aai_callback=cb,
        sigma_w_sq=sigma_w_sq,
        randomize_aai=randomize_aai,
        **extra
    )


# ============================================================================
# Sweep runner
# ============================================================================
def run_full_evaluation(args):
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    aai_callback = load_aai_callback(args.aai_callback)
    policy_factory = get_policy_factory(args.baseline, args.checkpoint_dir, args.device)

    # 학습 시 env로 actor를 미리 한 번 만들어 dim 검증 (빠른 실패)
    if args.baseline in ('mappo', 'mappo+aai', 'llm_aai'):
        sanity_env = make_env(args.default_uav, args.default_target,
                              args.baseline, aai_callback=aai_callback)
        try:
            _fn = policy_factory(sanity_env)
            print(f'[ok] checkpoint loaded; obs_shape={sanity_env.observation_space[0].shape}')
        except Exception as e:
            print(f'[fatal] checkpoint load failed: {e}')
            return

    common_kwargs = {'aai_callback': aai_callback, 'randomize_aai': args.randomize_aai}

    # ---- Sweep: num_uavs ----
    if args.sweep_uav:
        run_sweep(
            env_factory=lambda x, s: make_env(int(x), args.default_target,
                                              args.baseline, **common_kwargs),
            policy_name=args.baseline,
            policy_factory=policy_factory,
            x_axis_label='Number of UAVs',
            x_axis_values=args.sweep_uav,
            varying_arg='U',
            n_seeds=args.n_seeds, n_episodes_per_seed=args.n_episodes,
            save_dir=os.path.join(out_dir, 'sweep_uav'),
        )

    # ---- Sweep: num_targets ----
    if args.sweep_target:
        run_sweep(
            env_factory=lambda x, s: make_env(args.default_uav, int(x),
                                              args.baseline, **common_kwargs),
            policy_name=args.baseline,
            policy_factory=policy_factory,
            x_axis_label='Number of targets',
            x_axis_values=args.sweep_target,
            varying_arg='K',
            n_seeds=args.n_seeds, n_episodes_per_seed=args.n_episodes,
            save_dir=os.path.join(out_dir, 'sweep_target'),
        )

    # ---- Sweep: process noise σ_w² ----
    if args.sweep_noise:
        run_sweep(
            env_factory=lambda x, s: make_env(args.default_uav, args.default_target,
                                              args.baseline,
                                              sigma_w_sq=float(x),
                                              **common_kwargs),
            policy_name=args.baseline,
            policy_factory=policy_factory,
            x_axis_label=r'Process noise variance $\sigma_w^2$',
            x_axis_values=args.sweep_noise,
            varying_arg='sigmaW',
            n_seeds=args.n_seeds, n_episodes_per_seed=args.n_episodes,
            save_dir=os.path.join(out_dir, 'sweep_noise'),
        )

    # ---- Sweep: tau_s_ratio ----
    if args.sweep_tau:
        def make_env_with_tau(tau_s_r):
            env = make_env(args.default_uav, args.default_target,
                           args.baseline, **common_kwargs)
            original_reset = env.reset
            def patched_reset(*a, **kw):
                res = original_reset(*a, **kw)
                for u in env.uavs:
                    u.tau_s_ratio = tau_s_r
                return res
            env.reset = patched_reset
            return env
        run_sweep(
            env_factory=lambda x, s: make_env_with_tau(float(x)),
            policy_name=args.baseline,
            policy_factory=policy_factory,
            x_axis_label=r'Sensing time ratio $\tau_s/\delta$',
            x_axis_values=args.sweep_tau,
            varying_arg='tauS',
            n_seeds=args.n_seeds, n_episodes_per_seed=args.n_episodes,
            save_dir=os.path.join(out_dir, 'sweep_tau'),
        )

    # ---- Single episodes for visualization ----
    if args.save_episode:
        ep_dir = os.path.join(out_dir, 'episodes')
        for seed in (args.episode_seeds or [42]):
            env = make_env(args.default_uav, args.default_target,
                           args.baseline, **common_kwargs)
            policy = policy_factory(env)
            fname = f'{args.baseline}_seed{seed}.npz'
            save_episode(env, policy, os.path.join(ep_dir, fname),
                         method_name=args.baseline, seed=seed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', type=str, default=None,
                        help='actor.pt + run_meta.json이 있는 디렉토리')
    parser.add_argument('--out_dir', type=str, required=True)
    parser.add_argument('--baseline', type=str, required=True,
                        choices=['mappo+aai', 'mappo', 'naive_greedy',
                                 'random', 'hover', 'llm_aai'])
    parser.add_argument('--aai_callback', type=str, default=None,
                        help='LLM AAI callback. 형식: "module.path:func_name"')
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--n_seeds', type=int, default=3)
    parser.add_argument('--n_episodes', type=int, default=3)
    parser.add_argument('--default_uav', type=int, default=5,
                        help='target/noise/tau sweep 시 고정할 UAV 수')
    parser.add_argument('--default_target', type=int, default=2,
                        help='uav/noise/tau sweep 시 고정할 target 수')
    parser.add_argument('--sweep_uav', type=int, nargs='*', default=None)
    parser.add_argument('--sweep_target', type=int, nargs='*', default=None)
    parser.add_argument('--sweep_noise', type=float, nargs='*', default=None)
    parser.add_argument('--sweep_tau', type=float, nargs='*', default=None)
    parser.add_argument('--save_episode', action='store_true')
    parser.add_argument('--episode_seeds', type=int, nargs='*', default=None)
    parser.add_argument('--randomize_aai', action='store_true',
                        help='훈련처럼 W/eps/p_ut를 매 step 랜덤화. 기본 False (고정 시나리오 평가).')
    args = parser.parse_args()

    run_full_evaluation(args)
    print('\n[done] all evaluations finished.')


if __name__ == '__main__':
    main()
