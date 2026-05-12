"""
결과 파일 포맷 명세 + 더미 데이터 생성기
========================================

목적:
- 학습/평가가 끝나면 결과 npz 파일을 한 디렉토리에 모음
- 시각화 스크립트는 이 디렉토리만 보고 그래프를 그림 (학습 중 상호작용 X)

디렉토리 구조:
    results/
        sweep_uav/
            condition_U3K2.npz   # 1개 condition = 여러 seed의 평균/std
            condition_U5K2.npz
            condition_U7K2.npz
            condition_U10K2.npz
        sweep_target/
            condition_K1.npz
            ...
        sweep_noise/
            ...
        sweep_tau/
            ...
        episodes/                 # 단일 에피소드 상세 로그 (시각화용)
            mappo_aai_seed42.npz
            mappo_no_aai_seed42.npz
            naive_seed42.npz

각 condition_*.npz 파일에 들어가는 키:
    # 메타데이터
    method        : str    'mappo+aai' / 'mappo' / 'naive_greedy' / 'random'
    condition     : str    'U=3, K=2'
    x_axis_label  : str    'Number of UAVs'
    x_axis_value  : float  3.0
    n_episodes    : int    각 seed마다 episode 수
    n_seeds       : int
    
    # 통계 (seed에 대한 평균/표준편차)
    energy_total_mean      : float  episode당 total energy의 mean
    energy_total_std       : float
    energy_per_step_mean   : np.ndarray (T,)  스텝별 평균 (모든 seed/episode)
    energy_per_step_std    : np.ndarray (T,)
    F_kt_mean              : float  tracking 정확도 (모든 시점/타겟 평균)
    F_kt_std               : float
    detection_rate         : float  α의 평균
    collision_count        : float  episode당 평균 충돌 수
    untracked_count        : float  episode당 평균 미탐지 횟수 (식 28f 위반)
    reward_mean            : float
    reward_std             : float

각 episode_*.npz 파일에 들어가는 키 (단일 에피소드 시각화용):
    method         : str
    seed           : int
    num_uavs       : int
    num_targets    : int
    map_min, map_max, dt, max_steps : 환경 메타
    critical_zones : np.ndarray (Z, 2)
    bs_pos         : np.ndarray (2,)
    
    # 시계열 (T = max_steps + 1)
    uav_pos        : np.ndarray (T, U, 2)
    target_pos     : np.ndarray (T, K, 2)
    target_pos_est : np.ndarray (T, K, 2)
    energy_per_uav : np.ndarray (T, U)
    energy_total   : np.ndarray (T,)
    F_kt           : np.ndarray (T, K)
    alpha          : np.ndarray (T, U, K)
    p_ut           : np.ndarray (T, U)
    W_kt           : np.ndarray (T, K)
    eps_kt         : np.ndarray (T, K)
    d_Z_kt         : np.ndarray (T, K)
    team_reward    : np.ndarray (T,)
    collisions     : np.ndarray (T,)
    assignment     : np.ndarray (T, U)
"""

import sys
import os
import numpy as np
from onpolicy.envs.uav.uav_env import UAVTrackingEnv


def run_episode(env, policy_fn, log_episode=True):
    """
    한 에피소드 실행. policy_fn(obs) -> action_array (U, 2) 반환.
    """
    env.log_episode = log_episode
    obs, share_obs = env.reset()
    
    energy_per_step = []
    rewards = []
    detect_total, detect_count = 0, 0
    collisions_total = 0
    untracked_total = 0
    F_kt_total = []
    
    done = False
    while not done:
        action = policy_fn(obs)
        obs, share_obs, reward, dones, infos = env.step(action)
        energy_per_step.append(sum(u.last_e_tot for u in env.uavs))
        rewards.append(infos[0]['team_reward'])
        for u in env.uavs:
            for k, alp in u.is_detected_per_target.items():
                detect_total += alp
                detect_count += 1
        collisions_total += infos[0]['n_collisions']
        # 미탐지 카운트
        for k_idx in range(env.num_targets):
            n_det = sum(u.is_detected_per_target.get(k_idx, 0) for u in env.uavs)
            if n_det == 0:
                untracked_total += 1
        F_kt_total.append([t.F_kt for t in env.targets])
        done = bool(dones[0])
    
    out = {
        'energy_per_step': np.array(energy_per_step),
        'energy_total': float(np.sum(energy_per_step)),
        'reward_mean': float(np.mean(rewards)),
        'detection_rate': float(detect_total / max(detect_count, 1)),
        'collision_count': int(collisions_total),
        'untracked_count': int(untracked_total),
        'F_kt_mean': float(np.mean(F_kt_total)),
    }
    if log_episode:
        out['log'] = env.get_episode_log()
    return out


# ---------------- Baseline policies (학습 전용 더미) ----------------

def policy_random(obs):
    n_uav = obs.shape[0]
    return np.random.uniform(-1, 1, (n_uav, 2)).astype(np.float32)


def policy_naive_greedy_factory(env):
    """
    Naive 베이스라인: 각 UAV가 자기 assignment 타겟의 추정 위치를 향해 직진.
    학습 없이도 합리적 + 비교용 lower bound.
    """
    def policy(obs):
        actions = np.zeros((env.num_uavs, 2), dtype=np.float32)
        for u_idx, uav in enumerate(env.uavs):
            t_idx = env.assignment.get(u_idx, 0)
            target_est = env.targets[t_idx].S_global[:2]
            direction = target_est - uav.pos
            norm = np.linalg.norm(direction) + 1e-6
            actions[u_idx] = direction / norm  # 단위벡터 (env에서 v_max*dt 곱함)
        return actions
    return policy


def policy_hover(obs):
    """움직이지 않는 정책 (이론적 minimum-mobility)"""
    n_uav = obs.shape[0]
    return np.zeros((n_uav, 2), dtype=np.float32)


# ---------------- Sweep runner ----------------

def run_sweep(env_factory, policy_name, policy_factory, x_axis_label,
              x_axis_values, varying_arg, n_seeds=3, n_episodes_per_seed=3,
              save_dir='/home/claude/results_dummy/sweep'):
    """
    한 축(예: num_uavs)을 변동시키며 condition별 결과를 저장.
    policy_factory(env) -> policy_fn
    env_factory(x_value, seed) -> env
    """
    os.makedirs(save_dir, exist_ok=True)
    for x_val in x_axis_values:
        per_seed_stats = []
        T = None
        all_e_per_step = []
        for seed in range(n_seeds):
            for ep in range(n_episodes_per_seed):
                env = env_factory(x_val, seed * 1000 + ep)
                env.seed(seed * 1000 + ep)
                policy = policy_factory(env)
                res = run_episode(env, policy, log_episode=False)
                per_seed_stats.append(res)
                all_e_per_step.append(res['energy_per_step'])
                if T is None:
                    T = len(res['energy_per_step'])
        
        all_e = np.array(all_e_per_step)  # (n_seeds*n_eps, T)
        agg = {
            'method': policy_name,
            'condition': f'{varying_arg}={x_val}',
            'x_axis_label': x_axis_label,
            'x_axis_value': float(x_val),
            'n_episodes': len(per_seed_stats),
            'n_seeds': n_seeds,
            'energy_total_mean': float(np.mean([s['energy_total'] for s in per_seed_stats])),
            'energy_total_std':  float(np.std([s['energy_total'] for s in per_seed_stats])),
            'energy_per_step_mean': all_e.mean(axis=0),
            'energy_per_step_std':  all_e.std(axis=0),
            'F_kt_mean': float(np.mean([s['F_kt_mean'] for s in per_seed_stats])),
            'F_kt_std':  float(np.std([s['F_kt_mean'] for s in per_seed_stats])),
            'detection_rate': float(np.mean([s['detection_rate'] for s in per_seed_stats])),
            'collision_count': float(np.mean([s['collision_count'] for s in per_seed_stats])),
            'untracked_count': float(np.mean([s['untracked_count'] for s in per_seed_stats])),
            'reward_mean': float(np.mean([s['reward_mean'] for s in per_seed_stats])),
            'reward_std': float(np.std([s['reward_mean'] for s in per_seed_stats])),
        }
        fname = os.path.join(save_dir, f'{policy_name}_{varying_arg}={x_val}.npz')
        np.savez(fname, **agg)
        print(f'[saved] {fname}: E_total={agg["energy_total_mean"]:.1f}±{agg["energy_total_std"]:.1f}, '
              f'det={agg["detection_rate"]*100:.1f}%, col={agg["collision_count"]:.1f}')


def save_episode(env, policy_fn, save_path, method_name, seed=42):
    """단일 episode 상세 로그 저장 (시각화용)"""
    env.seed(seed)
    env.log_episode = True
    res = run_episode(env, policy_fn, log_episode=True)
    log = res['log']
    out = {
        'method': method_name,
        'seed': int(seed),
        'num_uavs': int(env.num_uavs),
        'num_targets': int(env.num_targets),
        'map_min': float(env.map_min),
        'map_max': float(env.map_max),
        'dt': float(env.dt),
        'max_steps': int(env.max_steps),
        'critical_zones': np.array(env.critical_zones),
        'bs_pos': env.bs_pos.copy(),
    }
    out.update(log)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez(save_path, **out)
    print(f'[episode saved] {save_path}: E_total={res["energy_total"]:.1f}, '
          f'det={res["detection_rate"]*100:.1f}%')


# ---------------- 더미 데이터 생성 (시각화 검증용) ----------------

def generate_dummy_results():
    """학습이 끝나지 않아도 시각화 파이프라인을 돌릴 수 있도록 더미 결과 생성"""
    base_dir = '/home/claude/results_dummy'
    
    # SWEEP 1: number of UAVs
    print('\n=== Sweep: num_uavs ===')
    for policy_name, policy_factory in [
        ('random', lambda env: policy_random),
        ('naive_greedy', lambda env: policy_naive_greedy_factory(env)),
    ]:
        run_sweep(
            env_factory=lambda x, s: UAVTrackingEnv(num_uavs=int(x), num_targets=2, use_aai=True),
            policy_name=policy_name,
            policy_factory=policy_factory,
            x_axis_label='Number of UAVs',
            x_axis_values=[3, 5, 7, 10],
            varying_arg='U',
            n_seeds=2, n_episodes_per_seed=2,
            save_dir=os.path.join(base_dir, 'sweep_uav'),
        )
    
    # SWEEP 2: number of targets
    print('\n=== Sweep: num_targets ===')
    for policy_name, policy_factory in [
        ('random', lambda env: policy_random),
        ('naive_greedy', lambda env: policy_naive_greedy_factory(env)),
    ]:
        run_sweep(
            env_factory=lambda x, s: UAVTrackingEnv(num_uavs=5, num_targets=int(x), use_aai=True),
            policy_name=policy_name,
            policy_factory=policy_factory,
            x_axis_label='Number of Targets',
            x_axis_values=[1, 2, 3, 4],
            varying_arg='K',
            n_seeds=2, n_episodes_per_seed=2,
            save_dir=os.path.join(base_dir, 'sweep_target'),
        )
    
    # SWEEP 3: noise power (process noise)
    print('\n=== Sweep: sigma_w_sq ===')
    for policy_name, policy_factory in [
        ('random', lambda env: policy_random),
        ('naive_greedy', lambda env: policy_naive_greedy_factory(env)),
    ]:
        run_sweep(
            env_factory=lambda x, s: UAVTrackingEnv(num_uavs=5, num_targets=2,
                                                    sigma_w_sq=float(x), use_aai=True),
            policy_name=policy_name,
            policy_factory=policy_factory,
            x_axis_label='Process noise variance σ²_w',
            x_axis_values=[1.0, 5.0, 20.0, 50.0],
            varying_arg='sigmaW',
            n_seeds=2, n_episodes_per_seed=2,
            save_dir=os.path.join(base_dir, 'sweep_noise'),
        )
    
    # SWEEP 4: tau_s ratio (sensing/comm 비율)
    print('\n=== Sweep: tau_s_ratio ===')
    def make_env_with_tau(tau_s_r):
        env = UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=True)
        # 환경 reset 안에서 만들어지는 UAV에도 tau_s_ratio 적용
        original_reset = env.reset
        def patched_reset(*args, **kwargs):
            res = original_reset(*args, **kwargs)
            for u in env.uavs:
                u.tau_s_ratio = tau_s_r
            return res
        env.reset = patched_reset
        return env
    for policy_name, policy_factory in [
        ('random', lambda env: policy_random),
        ('naive_greedy', lambda env: policy_naive_greedy_factory(env)),
    ]:
        run_sweep(
            env_factory=lambda x, s: make_env_with_tau(float(x)),
            policy_name=policy_name,
            policy_factory=policy_factory,
            x_axis_label='Sensing time ratio τ_s/δ',
            x_axis_values=[0.3, 0.5, 0.7],
            varying_arg='tauS',
            n_seeds=2, n_episodes_per_seed=2,
            save_dir=os.path.join(base_dir, 'sweep_tau'),
        )
    
    # 단일 에피소드 상세 로그 (시각화용)
    print('\n=== Single episode logs ===')
    save_episode(UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=True),
                 policy_naive_greedy_factory(UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=True)),
                 os.path.join(base_dir, 'episodes', 'naive_greedy_aai.npz'),
                 method_name='naive_greedy+AAI', seed=42)
    save_episode(UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=False),
                 policy_naive_greedy_factory(UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=False)),
                 os.path.join(base_dir, 'episodes', 'naive_greedy_no_aai.npz'),
                 method_name='naive_greedy_NO_AAI', seed=42)
    
    # 추가: 적은 UAV로 어려운 시나리오 (시각화 흥미용)
    save_episode(UAVTrackingEnv(num_uavs=3, num_targets=4, use_aai=True),
                 policy_naive_greedy_factory(UAVTrackingEnv(num_uavs=3, num_targets=4, use_aai=True)),
                 os.path.join(base_dir, 'episodes', 'naive_3uav_4target.npz'),
                 method_name='naive_3uav_4target', seed=42)


if __name__ == '__main__':
    generate_dummy_results()
    print('\nAll dummy results saved.')
