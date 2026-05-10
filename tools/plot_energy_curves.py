"""
plot_energy_curves.py
=====================
Sweep 결과를 읽어 y=총에너지(또는 step당 에너지), x=조건변수 그래프 생성.

회의 요구사항:
- y축은 에너지 (total)
- x축 후보: number of UAVs, number of targets, noise power, sensing/comm time ratio

각 sweep 디렉토리(results/sweep_uav, sweep_target, ...)에서 condition별 .npz를
읽어 method별 곡선 비교 그래프를 그림.

Usage:
    python plot_energy_curves.py --results_dir /path/to/results --out_dir /path/to/figs
"""
import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------- 그래프 스타일 (논문/슬라이드 톤) ----------
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'axes.linewidth': 1.0,
    'axes.grid': True,
    'grid.alpha': 0.25,
    'grid.linestyle': '--',
    'legend.fontsize': 10,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

# 메서드별 색상 (일관된 톤)
METHOD_STYLE = {
    'mappo+aai':    {'color': '#1f5fa8', 'marker': 'o', 'linestyle': '-',  'label': 'MAPPO + AAI'},
    'mappo':        {'color': '#3b8edb', 'marker': 's', 'linestyle': '-',  'label': 'MAPPO (no AAI)'},
    'naive_greedy': {'color': '#e07a3c', 'marker': '^', 'linestyle': '--', 'label': 'Naive greedy'},
    'random':       {'color': '#888888', 'marker': 'x', 'linestyle': ':',  'label': 'Random'},
    'hover':        {'color': '#9c27b0', 'marker': 'D', 'linestyle': '-.', 'label': 'Hover (no movement)'},
}

# Sweep dir → (x label for plot, plot title)
SWEEP_INFO = {
    'sweep_uav':    ('Number of UAVs',                    'Energy vs. fleet size'),
    'sweep_target': ('Number of targets',                 'Energy vs. number of targets'),
    'sweep_noise':  (r'Process noise variance $\sigma_w^2$', 'Energy vs. process noise'),
    'sweep_tau':    (r'Sensing time ratio $\tau_s/\delta$', 'Energy vs. sensing/comm split'),
}


def load_sweep_dir(sweep_dir):
    """디렉토리에서 모든 condition npz를 읽어 method별로 묶음."""
    results_by_method = {}
    if not os.path.isdir(sweep_dir):
        return results_by_method
    for fname in sorted(os.listdir(sweep_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(sweep_dir, fname)
        d = np.load(path, allow_pickle=True)
        method = str(d['method'])
        results_by_method.setdefault(method, []).append({
            'x': float(d['x_axis_value']),
            'condition': str(d['condition']),
            'x_label': str(d['x_axis_label']),
            'energy_total_mean': float(d['energy_total_mean']),
            'energy_total_std':  float(d['energy_total_std']),
            'energy_per_step_mean': d['energy_per_step_mean'],
            'energy_per_step_std':  d['energy_per_step_std'],
            'detection_rate':   float(d['detection_rate']),
            'collision_count':  float(d['collision_count']),
            'untracked_count':  float(d['untracked_count']),
            'F_kt_mean':        float(d['F_kt_mean']),
            'reward_mean':      float(d['reward_mean']),
        })
    # x로 정렬
    for m in results_by_method:
        results_by_method[m].sort(key=lambda r: r['x'])
    return results_by_method


def plot_sweep_comparison(sweep_dir, out_path, x_label=None, title=None):
    """y=총에너지, x=조건. 메서드별 곡선 + 보조 패널 (탐지율/충돌)."""
    data = load_sweep_dir(sweep_dir)
    if not data:
        print(f'[skip] no data in {sweep_dir}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    # 패널 1: total energy
    ax = axes[0]
    for method, runs in data.items():
        style = METHOD_STYLE.get(method, {'color': 'k', 'marker': 'o',
                                          'linestyle': '-', 'label': method})
        xs = [r['x'] for r in runs]
        ys = [r['energy_total_mean'] for r in runs]
        es = [r['energy_total_std'] for r in runs]
        ax.errorbar(xs, ys, yerr=es,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    capsize=3, markersize=7, linewidth=1.5)
    ax.set_xlabel(x_label or runs[0]['x_label'])
    ax.set_ylabel('Total energy per episode  [J]')
    ax.set_title('Total energy', fontweight='semibold')
    ax.legend(loc='best')

    # 패널 2: detection rate + collision  (twin axis)
    ax = axes[1]
    ax2 = ax.twinx()
    for method, runs in data.items():
        style = METHOD_STYLE.get(method, {'color': 'k', 'marker': 'o',
                                          'linestyle': '-', 'label': method})
        xs = [r['x'] for r in runs]
        det = [r['detection_rate'] * 100 for r in runs]
        col = [r['collision_count'] for r in runs]
        ax.plot(xs, det, color=style['color'], marker=style['marker'],
                linestyle='-', label=style['label'] + ' (det.)',
                markersize=6, linewidth=1.5)
        ax2.plot(xs, col, color=style['color'], marker=style['marker'],
                 linestyle=':', alpha=0.55, markersize=5, linewidth=1.0)
    ax.set_xlabel(x_label or runs[0]['x_label'])
    ax.set_ylabel('Detection rate  [%]', color='#222')
    ax2.set_ylabel('Avg. collisions / episode', color='#666')
    ax.set_ylim(-5, 105)
    ax.set_title('Tracking quality', fontweight='semibold')
    ax2.tick_params(axis='y', colors='#666')
    ax.legend(loc='lower left', fontsize=9)

    # 패널 3: per-step energy curve (한 condition, 모든 method 비교)
    ax = axes[2]
    if len(data):
        # 중간 condition을 골라 보여줌
        any_method = next(iter(data))
        n_cond = len(data[any_method])
        mid_idx = n_cond // 2
        mid_x = data[any_method][mid_idx]['x']
        for method, runs in data.items():
            run = next((r for r in runs if abs(r['x'] - mid_x) < 1e-9), None)
            if run is None:
                continue
            style = METHOD_STYLE.get(method, {'color': 'k', 'marker': 'o',
                                              'linestyle': '-', 'label': method})
            mean = run['energy_per_step_mean']
            std = run['energy_per_step_std']
            ts = np.arange(len(mean))
            ax.plot(ts, mean, color=style['color'], linestyle=style['linestyle'],
                    label=style['label'], linewidth=1.5)
            ax.fill_between(ts, mean - std, mean + std,
                            color=style['color'], alpha=0.15)
        ax.set_xlabel('Time step  $t$')
        ax.set_ylabel('Energy / step  [J]')
        ax.set_title(f'Per-step energy (at {x_label}={mid_x})', fontweight='semibold')
        ax.legend(loc='best', fontsize=9)

    fig.suptitle(title or os.path.basename(sweep_dir),
                 fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def plot_overview(results_dir, out_path):
    """모든 sweep을 한 figure에 모아서 (y=energy)만 보여주는 요약."""
    n = len(SWEEP_INFO)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.8))
    if n == 1:
        axes = [axes]

    for ax, (sweep_name, (xlbl, ttl)) in zip(axes, SWEEP_INFO.items()):
        sweep_dir = os.path.join(results_dir, sweep_name)
        data = load_sweep_dir(sweep_dir)
        if not data:
            ax.text(0.5, 0.5, f'(no data:\n{sweep_name})',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#999', fontsize=11)
            ax.set_xlabel(xlbl)
            ax.set_ylabel('Total energy [J]')
            ax.set_title(ttl, fontweight='semibold')
            continue
        for method, runs in data.items():
            style = METHOD_STYLE.get(method, {'color': 'k', 'marker': 'o',
                                              'linestyle': '-', 'label': method})
            xs = [r['x'] for r in runs]
            ys = [r['energy_total_mean'] for r in runs]
            es = [r['energy_total_std'] for r in runs]
            ax.errorbar(xs, ys, yerr=es,
                        color=style['color'], marker=style['marker'],
                        linestyle=style['linestyle'], label=style['label'],
                        capsize=3, markersize=6, linewidth=1.4)
        ax.set_xlabel(xlbl)
        ax.set_ylabel('Total energy [J]')
        ax.set_title(ttl, fontweight='semibold')
        ax.legend(loc='best', fontsize=9)

    fig.suptitle('Energy comparison across experimental conditions',
                 fontsize=14, fontweight='bold', y=1.03)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.results_dir, 'figures')
    os.makedirs(out_dir, exist_ok=True)

    # 각 sweep마다 상세 figure (3 panels)
    for sweep_name, (xlbl, ttl) in SWEEP_INFO.items():
        sweep_dir = os.path.join(args.results_dir, sweep_name)
        out_path = os.path.join(out_dir, f'{sweep_name}_detailed.png')
        plot_sweep_comparison(sweep_dir, out_path, x_label=xlbl, title=ttl)

    # 모두 모은 overview (energy only)
    plot_overview(args.results_dir, os.path.join(out_dir, 'energy_overview.png'))


if __name__ == '__main__':
    main()
