from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import os

base = r'c:\Users\dongj\Desktop\nalpari\onpolicy\scripts\results\UAV\uav_tracking\mappo\nalpari_v1\run1\logs'
for sub in ['average_episode_rewards', 'eval_average_episode_rewards', 'value_loss', 'policy_loss', 'dist_entropy']:
    p = os.path.join(base, sub, sub)
    if not os.path.isdir(p):
        continue
    ea = EventAccumulator(p)
    ea.Reload()
    for tag in ea.Tags().get('scalars', []):
        evs = ea.Scalars(tag)
        if not evs:
            continue
        first = evs[0]
        last = evs[-1]
        best = max(evs, key=lambda e: e.value)
        worst = min(evs, key=lambda e: e.value)
        print(f"[{sub}/{tag}]", flush=True)
        print(f"  n={len(evs)}, first step={first.step:>9} val={first.value:>12.3f}", flush=True)
        print(f"  last step={last.step:>9} val={last.value:>12.3f}", flush=True)
        print(f"  best step={best.step:>9} val={best.value:>12.3f}", flush=True)
        print(f"  worst step={worst.step:>9} val={worst.value:>12.3f}", flush=True)
