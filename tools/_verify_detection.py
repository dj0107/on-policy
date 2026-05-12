import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from onpolicy.envs.uav.uav_env import UAVTrackingEnv
from tools.run_evaluation import policy_naive_greedy_factory, policy_random, policy_hover, run_episode
import numpy as np

uav_sample = UAVTrackingEnv(num_uavs=5, num_targets=2).uavs[0]
print(f'G_t={uav_sample.G_t}, G_r={uav_sample.G_r}')
print()

N_SEEDS = 8

for label, make_policy in [
    ('naive_greedy', lambda env: policy_naive_greedy_factory(env)),
    ('random',       lambda env: policy_random),
    ('hover',        lambda env: policy_hover),
]:
    results = []
    for seed in range(N_SEEDS):
        env = UAVTrackingEnv(num_uavs=5, num_targets=2, use_aai=True)
        env.seed(seed * 100)
        policy = make_policy(env)
        res = run_episode(env, policy, log_episode=False)
        results.append(res)
    print(f'[{label}]')
    print(f'  detection_rate:  {np.mean([r["detection_rate"] for r in results])*100:.1f}%')
    print(f'  energy_total:    {np.mean([r["energy_total"] for r in results]):.0f}')
    print(f'  collision_count: {np.mean([r["collision_count"] for r in results]):.1f}')
    print(f'  untracked_count: {np.mean([r["untracked_count"] for r in results]):.1f}')
    print()
