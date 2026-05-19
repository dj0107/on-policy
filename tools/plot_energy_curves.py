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

# 메서드별 색상/스타일 (논문 그래프 기준)
METHOD_STYLE = {
    'mappo+aai':    {'color': '#1f5fa8', 'marker': 'o', 'linestyle': '-',  'label': 'MAPPO + AAI (Ours)', 'lw': 2.5, 'ms': 9},
    'mappo':        {'color': '#3b8edb', 'marker': 's', 'linestyle': '-',  'label': 'MAPPO (no AAI)',      'lw': 1.5, 'ms': 7},
    'naive_greedy': {'color': '#e07a3c', 'marker': '^', 'linestyle': '--', 'label': 'Naive greedy',        'lw': 1.7, 'ms': 7},
    'random':       {'color': '#888888', 'marker': 'x', 'linestyle': ':',  'label': 'Random',              'lw': 1.5, 'ms': 7},
    'hover':        {'color': '#9c27b0', 'marker': 'D', 'linestyle': '-.', 'label': 'Hover (static)',      'lw': 1.7, 'ms': 7},
}
# 핵심 3개 비교군만 표시 (논문 주장에 집중)
SHOW_METHODS = ('mappo+aai', 'naive_greedy', 'hover')

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
            'tracking_err_mean':           float(d['tracking_err_mean']) if 'tracking_err_mean' in d.files else float('nan'),
            'tracking_err_detected_mean':  float(d['tracking_err_detected_mean']) if 'tracking_err_detected_mean' in d.files else float('nan'),
        })
    # x로 정렬
    for m in results_by_method:
        results_by_method[m].sort(key=lambda r: r['x'])
    return results_by_method


def _shared_legend(fig, axes, bottom_margin=0.10):
    """모든 axes에서 핵심 메서드 handles를 모아 figure 하단에 공유 범례 배치."""
    seen, handles, labels = set(), [], []
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in seen:
                seen.add(l)
                handles.append(h)
                labels.append(l)
    fig.legend(handles, labels,
               loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, 0.0),
               frameon=True, framealpha=0.92,
               fontsize=10, markerscale=1.1)
    fig.tight_layout(rect=[0, bottom_margin, 1, 1])


def plot_sweep_comparison(sweep_dir, out_path, x_label=None, title=None):
    """3개 핵심 패널: 총에너지 / 탐지율 / PCRLB 추적 불확실성."""
    data = {m: v for m, v in load_sweep_dir(sweep_dir).items() if m in SHOW_METHODS}
    if not data:
        print(f'[skip] no data in {sweep_dir}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    any_runs = next(iter(data.values()))
    xlbl = x_label or any_runs[0]['x_label']

    # 패널 1: 총 에너지 (±std error bar)
    ax = axes[0]
    for method in SHOW_METHODS:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        runs = data[method]
        xs = [r['x'] for r in runs]
        ys = [r['energy_total_mean'] for r in runs]
        es = [r['energy_total_std'] for r in runs]
        ax.errorbar(xs, ys, yerr=es,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    capsize=4, markersize=style['ms'], linewidth=style['lw'])
    ax.set_xlabel(xlbl)
    ax.set_ylabel('Total energy per episode  [J]')
    ax.set_title('Total energy', fontweight='semibold')

    # 패널 2: 탐지율
    ax = axes[1]
    for method in SHOW_METHODS:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        runs = data[method]
        xs = [r['x'] for r in runs]
        det = [r['detection_rate'] * 100 for r in runs]
        ax.plot(xs, det,
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], label=style['label'],
                markersize=style['ms'], linewidth=style['lw'])
    ax.set_xlabel(xlbl)
    ax.set_ylabel('Detection rate  [%]')
    ax.set_ylim(-5, 105)
    ax.set_title('Target detection rate', fontweight='semibold')

    # 패널 3: PCRLB 추적 불확실성 F_kt (↓ better)
    ax = axes[2]
    for method in SHOW_METHODS:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        runs = data[method]
        xs = [r['x'] for r in runs]
        ys = [r['F_kt_mean'] for r in runs]
        ax.plot(xs, ys,
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], label=style['label'],
                markersize=style['ms'], linewidth=style['lw'])
    ax.set_xlabel(xlbl)
    ax.set_ylabel(r'Mean $\bar{F}_{kt}$  (↓ better)')
    ax.set_title('PCRLB tracking uncertainty', fontweight='semibold')

    fig.suptitle(title or os.path.basename(sweep_dir),
                 fontsize=14, fontweight='bold')
    _shared_legend(fig, axes, bottom_margin=0.12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def plot_overview(results_dir, out_path):
    """모든 sweep을 한 figure에 모아서 (y=energy) 요약 — 핵심 3개 메서드."""
    n = len(SWEEP_INFO)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0))
    if n == 1:
        axes = [axes]

    for ax, (sweep_name, (xlbl, ttl)) in zip(axes, SWEEP_INFO.items()):
        sweep_dir = os.path.join(results_dir, sweep_name)
        data = {m: v for m, v in load_sweep_dir(sweep_dir).items() if m in SHOW_METHODS}
        if not data:
            ax.text(0.5, 0.5, f'(no data:\n{sweep_name})',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#999', fontsize=11)
            ax.set_xlabel(xlbl)
            ax.set_ylabel('Total energy [J]')
            ax.set_title(ttl, fontweight='semibold')
            continue
        for method in SHOW_METHODS:
            if method not in data:
                continue
            style = METHOD_STYLE[method]
            runs = data[method]
            xs = [r['x'] for r in runs]
            ys = [r['energy_total_mean'] for r in runs]
            es = [r['energy_total_std'] for r in runs]
            ax.errorbar(xs, ys, yerr=es,
                        color=style['color'], marker=style['marker'],
                        linestyle=style['linestyle'], label=style['label'],
                        capsize=4, markersize=style['ms'], linewidth=style['lw'])
        ax.set_xlabel(xlbl)
        ax.set_ylabel('Total energy [J]')
        ax.set_title(ttl, fontweight='semibold')

    fig.suptitle('Energy comparison across experimental conditions',
                 fontsize=14, fontweight='bold')
    _shared_legend(fig, axes, bottom_margin=0.14)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def plot_fkt_overview(results_dir, out_path):
    """모든 sweep에서 PCRLB F_kt 비교 — 핵심 3개 메서드."""
    n = len(SWEEP_INFO)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5.0))
    if n == 1:
        axes = [axes]

    for ax, (sweep_name, (xlbl, ttl)) in zip(axes, SWEEP_INFO.items()):
        sweep_dir = os.path.join(results_dir, sweep_name)
        data = {m: v for m, v in load_sweep_dir(sweep_dir).items() if m in SHOW_METHODS}
        if not data:
            ax.text(0.5, 0.5, f'(no data:\n{sweep_name})',
                    ha='center', va='center', transform=ax.transAxes,
                    color='#999', fontsize=11)
            ax.set_xlabel(xlbl)
            ax.set_ylabel(r'Mean $\bar{F}_{kt}$  (↓ better)')
            ax.set_title(ttl, fontweight='semibold')
            continue
        for method in SHOW_METHODS:
            if method not in data:
                continue
            style = METHOD_STYLE[method]
            runs = data[method]
            xs = [r['x'] for r in runs]
            ys = [r['F_kt_mean'] for r in runs]
            ax.plot(xs, ys, color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], label=style['label'],
                    markersize=style['ms'], linewidth=style['lw'])
        ax.set_xlabel(xlbl)
        ax.set_ylabel(r'Mean $\bar{F}_{kt}$  (↓ better)')
        ax.set_title(ttl, fontweight='semibold')

    fig.suptitle(r'PCRLB tracking uncertainty $\bar{F}_{kt}$ across conditions',
                 fontsize=14, fontweight='bold')
    _shared_legend(fig, axes, bottom_margin=0.14)
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

    # F_kt tracking quality overview
    plot_fkt_overview(args.results_dir, os.path.join(out_dir, 'fkt_overview.png'))


if __name__ == '__main__':
    main()
