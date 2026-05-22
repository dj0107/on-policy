"""
generate_final_figures.py
=========================
최종 발표용 그래프를 final_figure/ 폴더에 생성.

수정 사항:
1. Episode seed42 그래프: snapshot 패널에 UAV 궤적 선 추가 + 메인 맵 UAV 선 강화
2. Sweep 그래프: detection rate / PCRLB y축을 데이터 범위에 맞게 자동 스케일 (두 직선이 더 멀어 보이도록)
"""

import os
import shutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ─── 경로 설정 ───────────────────────────────────────────────────────────────
ARCHIVE_FIGURES = r"c:\Users\dongj\Documents\GitHub\on-policy\_archive\nalpari_v1_2026-05-21\figures"
ARCHIVE_EVAL    = r"c:\Users\dongj\Documents\GitHub\on-policy\_archive\nalpari_v1_2026-05-21\evaluation"
OUT_DIR         = r"c:\Users\dongj\Documents\GitHub\on-policy\final_figure"

os.makedirs(OUT_DIR, exist_ok=True)

# ─── 공통 스타일 ──────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'axes.linewidth': 1.0,
    'axes.grid': False,
    'legend.fontsize': 9,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

UAV_CMAP    = plt.cm.viridis
TARGET_CMAP = plt.cm.plasma
RISK_COLOR  = '#c62828'
BS_COLOR    = '#2e7d32'


def _make_uav_colors(n):
    return [UAV_CMAP(i / max(n - 1, 1)) for i in range(n)]


def _make_target_colors(k):
    return [TARGET_CMAP(0.15 + 0.55 * i / max(k - 1, 1)) for i in range(k)]


# ═══════════════════════════════════════════════════════════════════════════════
#  1. Episode 그래프 (UAV 경로 강화)
# ═══════════════════════════════════════════════════════════════════════════════

def _draw_map_layout(ax, log, env_meta):
    map_min = float(env_meta['map_min'])
    map_max = float(env_meta['map_max'])
    bs_pos  = env_meta['bs_pos']
    risk_zones = env_meta['critical_zones']
    num_uavs   = int(env_meta['num_uavs'])
    num_targets = int(env_meta['num_targets'])

    uav_pos = log['uav_pos']
    tgt_pos = log['target_pos']
    tgt_est = log['target_pos_est']

    # 위험 구역
    for cz in risk_zones:
        for r, alpha in [(120.0, 0.06), (80.0, 0.08), (40.0, 0.13)]:
            ax.add_patch(plt.Circle(cz, r, facecolor=RISK_COLOR, alpha=alpha, linewidth=0))
        ax.add_patch(plt.Circle(cz, 25, facecolor=RISK_COLOR, alpha=0.45,
                                linewidth=1.2, edgecolor=RISK_COLOR))
        ax.text(cz[0], cz[1] + 35, 'risk\nzone', ha='center', va='bottom',
                fontsize=8, color=RISK_COLOR, fontweight='semibold')

    # Base Station
    ax.plot(*bs_pos, marker='s', color=BS_COLOR, markersize=10,
            markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    ax.text(bs_pos[0], bs_pos[1] - 35, 'BS', ha='center', va='top',
            fontsize=9, color=BS_COLOR, fontweight='bold')

    # ★ UAV 궤적: 선을 두껍게 + 시작 위치 표시
    uav_colors = _make_uav_colors(num_uavs)
    T = uav_pos.shape[0]
    for u in range(num_uavs):
        traj = uav_pos[:, u, :]
        # 시간에 따라 alpha를 점점 진하게 (과거→희미, 현재→진함)
        n_seg = min(T - 1, 30)
        seg_size = max(1, (T - 1) // n_seg)
        for s in range(0, T - 1, seg_size):
            e = min(s + seg_size + 1, T)
            frac = s / max(T - 1, 1)
            a = 0.35 + 0.65 * frac  # 0.35 → 1.0 으로 점점 진해짐
            ax.plot(traj[s:e, 0], traj[s:e, 1], '-',
                    color=uav_colors[u], linewidth=2.0, alpha=a, zorder=3)
        # 시작점 (작은 원)
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=uav_colors[u],
                markersize=5, markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.6, zorder=4)
        # 끝점 (삼각형)
        ax.plot(traj[-1, 0], traj[-1, 1], '^', color=uav_colors[u],
                markersize=11, markeredgecolor='white', markeredgewidth=1.2,
                zorder=6, label=f'UAV {u}')

    # 타겟 궤적
    tgt_colors = _make_target_colors(num_targets)
    for k in range(num_targets):
        traj     = tgt_pos[:, k, :]
        traj_est = tgt_est[:, k, :]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=tgt_colors[k],
                linewidth=2.0, alpha=0.85, zorder=3, label=f'Target {k} (true)')
        ax.plot(traj_est[:, 0], traj_est[:, 1], ':', color=tgt_colors[k],
                linewidth=1.3, alpha=0.65, zorder=3)
        ax.plot(traj[0, 0], traj[0, 1], 's', color=tgt_colors[k],
                markersize=8, markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.7, zorder=4)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=tgt_colors[k],
                markersize=14, markeredgecolor='white', markeredgewidth=1.2, zorder=6)

    ax.set_xlim(map_min - 30, map_max + 30)
    ax.set_ylim(map_min - 30, map_max + 30)
    ax.set_aspect('equal')
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')
    ax.add_patch(plt.Rectangle((map_min, map_min), map_max - map_min, map_max - map_min,
                                fill=False, edgecolor='#444', linewidth=1.2,
                                linestyle='-', zorder=1))

    handles = [
        Line2D([0], [0], marker='^', color='#1f5fa8', markerfacecolor='#1f5fa8',
               markersize=10, linewidth=2.0, label='UAV (final)'),
        Line2D([0], [0], color='#1f5fa8', linewidth=2.0, alpha=0.8, label='UAV (path)'),
        Line2D([0], [0], color=tgt_colors[0], linewidth=2, label='Target (true)'),
        Line2D([0], [0], color=tgt_colors[0], linewidth=1.3,
               linestyle=':', label='Target (BS estimate)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BS_COLOR,
               markersize=9, label='Base station'),
        mpatches.Patch(color=RISK_COLOR, alpha=0.45, label='Risk zone'),
    ]
    ax.legend(handles=handles,
              bbox_to_anchor=(0.0, -0.12, 1.0, 0.08),
              loc='upper left', mode='expand',
              ncol=3, fontsize=7.5, framealpha=0.92, borderaxespad=0)


def _draw_energy_timeline(ax, log):
    e_per_uav = log['energy_per_uav']
    e_total   = log['energy_total']
    T, U = e_per_uav.shape
    ts = np.arange(T)
    uav_colors = _make_uav_colors(U)
    bottom = np.zeros(T)
    for u in range(U):
        ax.fill_between(ts, bottom, bottom + e_per_uav[:, u],
                        color=uav_colors[u], alpha=0.7, label=f'UAV {u}', linewidth=0)
        bottom += e_per_uav[:, u]
    ax.plot(ts, e_total, color='black', linewidth=1.6, label='total')
    ax.set_xlabel('Time step  $t$')
    ax.set_ylabel('Energy / step  [J]')
    ax.set_title('Per-step energy breakdown', fontweight='semibold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2)


def _draw_F_kt_timeline(ax, log, env_meta):
    F = log['F_kt']
    T, K = F.shape
    ts = np.arange(T)
    quality = np.maximum(0.0, (1.0 - np.clip(F, 0.0, 1000.0) / 1000.0) * 100.0)
    tgt_colors = _make_target_colors(K)
    for k in range(K):
        ax.plot(ts, quality[:, k], color=tgt_colors[k], linewidth=1.6, label=f'Target {k}')
    ax.set_xlabel('Time step  $t$')
    ax.set_ylabel('Tracking quality  [%]')
    ax.set_title('Tracking quality', fontweight='semibold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)


def _draw_detection_heatmap(ax, log, env_meta):
    A = log['alpha']
    T, U, K = A.shape
    heat = A.transpose(1, 2, 0).reshape(U * K, T)
    ax.imshow(heat, aspect='auto', cmap='Greens', interpolation='nearest', vmin=0, vmax=1)
    yticks, ylabels = [], []
    for u in range(U):
        for k in range(K):
            yticks.append(u * K + k)
            ylabels.append(f'U{u}-T{k}')
    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel('Time step')
    ax.set_title(r'Detection $\alpha_{u,k,t}$', fontweight='semibold')


def _draw_snapshots(fig, log, env_meta, n_snaps=5):
    """★ 수정: 각 스냅샷에 UAV 궤적 trailing 선 추가"""
    map_min    = float(env_meta['map_min'])
    map_max    = float(env_meta['map_max'])
    bs_pos     = env_meta['bs_pos']
    risk_zones = env_meta['critical_zones']
    num_uavs   = int(env_meta['num_uavs'])
    num_targets = int(env_meta['num_targets'])

    uav_pos = log['uav_pos']
    tgt_pos = log['target_pos']
    alpha   = log['alpha']
    T = uav_pos.shape[0]

    snap_ts = np.linspace(0, T - 1, n_snaps).astype(int)
    uav_colors = _make_uav_colors(num_uavs)
    tgt_colors = _make_target_colors(num_targets)

    TRAIL_LEN = 25  # 스냅샷에서 보여줄 최근 궤적 길이

    gs = fig.add_gridspec(nrows=1, ncols=n_snaps, top=0.18, bottom=0.02,
                          left=0.05, right=0.97, wspace=0.15)
    for i, t in enumerate(snap_ts):
        ax = fig.add_subplot(gs[0, i])
        for cz in risk_zones:
            ax.add_patch(plt.Circle(cz, 25, facecolor=RISK_COLOR, alpha=0.5, linewidth=0))
        ax.plot(*bs_pos, 's', color=BS_COLOR, markersize=6, zorder=4)

        # ★ UAV: trailing 경로선 + 현재 위치 마커
        for u in range(num_uavs):
            t_start = max(0, t - TRAIL_LEN)
            trail = uav_pos[t_start:t + 1, u, :]
            if len(trail) > 1:
                # alpha를 점점 진하게
                n = len(trail) - 1
                for s in range(n):
                    a = 0.25 + 0.75 * (s / max(n - 1, 1))
                    ax.plot(trail[s:s+2, 0], trail[s:s+2, 1], '-',
                            color=uav_colors[u], linewidth=1.0, alpha=a, zorder=3)
            ax.plot(uav_pos[t, u, 0], uav_pos[t, u, 1], '^',
                    color=uav_colors[u], markersize=6,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)

        # 타겟
        for k in range(num_targets):
            ax.plot(tgt_pos[t, k, 0], tgt_pos[t, k, 1], '*',
                    color=tgt_colors[k], markersize=10,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            for u in range(num_uavs):
                if alpha[t, u, k] > 0.5:
                    ax.plot([uav_pos[t, u, 0], tgt_pos[t, k, 0]],
                            [uav_pos[t, u, 1], tgt_pos[t, k, 1]],
                            '-', color=tgt_colors[k], alpha=0.35, linewidth=0.8, zorder=2)

        ax.set_xlim(map_min, map_max)
        ax.set_ylim(map_min, map_max)
        ax.set_aspect('equal')
        ax.set_title(f't = {t}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#888')


def plot_episode(npz_path, out_path):
    d = np.load(npz_path, allow_pickle=True)
    env_meta = {
        'map_min': float(d['map_min']),
        'map_max': float(d['map_max']),
        'bs_pos': d['bs_pos'],
        'critical_zones': d['critical_zones'],
        'num_uavs': int(d['num_uavs']),
        'num_targets': int(d['num_targets']),
        'dt': float(d['dt']),
        'max_steps': int(d['max_steps']),
    }
    log = {
        'uav_pos':        d['uav_pos'],
        'target_pos':     d['target_pos'],
        'target_pos_est': d['target_pos_est'],
        'energy_per_uav': d['energy_per_uav'],
        'energy_total':   d['energy_total'],
        'F_kt':           d['F_kt'],
        'alpha':          d['alpha'],
        'p_ut':           d['p_ut'],
        'W_kt':           d['W_kt'],
        'eps_kt':         d['eps_kt'],
        'team_reward':    d['team_reward'],
        'collisions':     d['collisions'],
    }
    method = str(d['method'])
    seed   = int(d['seed'])

    fig = plt.figure(figsize=(13.0, 10.5))
    gs_top = fig.add_gridspec(nrows=2, ncols=3, top=0.95, bottom=0.27,
                              left=0.05, right=0.97,
                              hspace=0.42, wspace=0.32,
                              width_ratios=[1.4, 1, 1])

    ax_map = fig.add_subplot(gs_top[:, 0])
    _draw_map_layout(ax_map, log, env_meta)
    ax_map.set_title('Top-down trajectory  (UAVs · targets · risk zones · BS)',
                     fontsize=10.5, fontweight='semibold')

    title_str = (f"Episode visualization · method={method} · seed={seed}    "
                 f"|    U={env_meta['num_uavs']}, K={env_meta['num_targets']}, "
                 f"map=[{env_meta['map_min']:.0f}, {env_meta['map_max']:.0f}] m, "
                 f"δ={env_meta['dt']}s, T={env_meta['max_steps']}")
    fig.suptitle(title_str, fontsize=11.5, fontweight='bold', y=0.99)

    ax_e = fig.add_subplot(gs_top[0, 1])
    _draw_energy_timeline(ax_e, log)
    ax_F = fig.add_subplot(gs_top[1, 1])
    _draw_F_kt_timeline(ax_F, log, env_meta)

    ax_alpha = fig.add_subplot(gs_top[:, 2])
    _draw_detection_heatmap(ax_alpha, log, env_meta)

    _draw_snapshots(fig, log, env_meta, n_snaps=5)
    fig.text(0.5, 0.205, 'Time-lapse snapshots',
             ha='center', va='bottom', fontsize=10.5,
             fontweight='semibold', color='#444')

    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  2. Sweep 그래프 (y축 자동 스케일)
# ═══════════════════════════════════════════════════════════════════════════════

plt.rcParams.update({
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

METHOD_STYLE = {
    'mappo+aai':    {'color': '#1f5fa8', 'marker': 'o', 'linestyle': '-',  'label': 'MAPPO + AAI (Ours)', 'lw': 2.5, 'ms': 9},
    'mappo':        {'color': '#3b8edb', 'marker': 's', 'linestyle': '-',  'label': 'MAPPO (no AAI)',      'lw': 1.5, 'ms': 7},
    'naive_greedy': {'color': '#e07a3c', 'marker': '^', 'linestyle': '--', 'label': 'Naive greedy',        'lw': 1.7, 'ms': 7},
    'random':       {'color': '#888888', 'marker': 'x', 'linestyle': ':',  'label': 'Random',              'lw': 1.5, 'ms': 7},
    'hover':        {'color': '#9c27b0', 'marker': 'D', 'linestyle': '-.', 'label': 'Hover (static)',      'lw': 1.7, 'ms': 7},
}
SHOW_METHODS = ('mappo+aai', 'naive_greedy', 'hover')

SWEEP_INFO = {
    'sweep_uav':    ('Number of UAVs',                         'Energy vs. fleet size'),
    'sweep_target': ('Number of targets',                      'Energy vs. number of targets'),
    'sweep_noise':  (r'Process noise variance $\sigma_w^2$',   'Energy vs. process noise'),
    'sweep_tau':    (r'Sensing time ratio $\tau_s/\delta$',    'Energy vs. sensing/comm split'),
}


def load_sweep_dir(sweep_dir):
    results = {}
    if not os.path.isdir(sweep_dir):
        return results
    for fname in sorted(os.listdir(sweep_dir)):
        if not fname.endswith('.npz'):
            continue
        path = os.path.join(sweep_dir, fname)
        d = np.load(path, allow_pickle=True)
        method = str(d['method'])
        results.setdefault(method, []).append({
            'x':                   float(d['x_axis_value']),
            'x_label':             str(d['x_axis_label']),
            'energy_total_mean':   float(d['energy_total_mean']),
            'energy_total_std':    float(d['energy_total_std']),
            'detection_rate':      float(d['detection_rate']),
            'F_kt_mean':           float(d['F_kt_mean']),
        })
    for m in results:
        results[m].sort(key=lambda r: r['x'])
    return results


def _tight_ylim(ax, pad_frac=0.12, min_span=None):
    """현재 axes의 데이터 범위를 읽어 여백을 추가한 ylim 설정."""
    ax.relim()
    ax.autoscale_view()
    ylo, yhi = ax.get_ylim()
    span = yhi - ylo
    if min_span is not None and span < min_span:
        mid = (ylo + yhi) / 2
        ylo = mid - min_span / 2
        yhi = mid + min_span / 2
        span = min_span
    ax.set_ylim(ylo - pad_frac * span, yhi + pad_frac * span)


def _shared_legend(fig, axes, bottom_margin=0.10):
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


def plot_sweep(sweep_dir, out_path, x_label=None, title=None):
    """★ 수정: detection rate / PCRLB y축 자동 스케일로 두 직선이 더 잘 보이도록."""
    data = {m: v for m, v in load_sweep_dir(sweep_dir).items() if m in SHOW_METHODS}
    if not data:
        print(f'[skip] no data in {sweep_dir}')
        return

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))
    any_runs = next(iter(data.values()))
    xlbl = x_label or any_runs[0]['x_label']

    # 패널 1: 총 에너지
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

    # ★ 패널 2: 탐지율 — y축 자동 스케일 (데이터 범위에 맞게 확대)
    ax = axes[1]
    det_vals = []
    for method in SHOW_METHODS:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        runs = data[method]
        xs  = [r['x'] for r in runs]
        det = [r['detection_rate'] * 100 for r in runs]
        det_vals.extend(det)
        ax.plot(xs, det,
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], label=style['label'],
                markersize=style['ms'], linewidth=style['lw'])
    ax.set_xlabel(xlbl)
    ax.set_ylabel('Detection rate  [%]')
    ax.set_title('Target detection rate', fontweight='semibold')
    # ★ 핵심 변경: 고정 ylim 제거, 데이터 범위 기반 스케일
    if det_vals:
        lo, hi = min(det_vals), max(det_vals)
        span = max(hi - lo, 5.0)   # 최소 5% span 확보
        pad  = span * 0.20
        ax.set_ylim(max(0, lo - pad), min(100, hi + pad))

    # ★ 패널 3: PCRLB — y축 자동 스케일
    ax = axes[2]
    fkt_vals = []
    for method in SHOW_METHODS:
        if method not in data:
            continue
        style = METHOD_STYLE[method]
        runs = data[method]
        xs = [r['x'] for r in runs]
        ys = [r['F_kt_mean'] for r in runs]
        fkt_vals.extend(ys)
        ax.plot(xs, ys,
                color=style['color'], marker=style['marker'],
                linestyle=style['linestyle'], label=style['label'],
                markersize=style['ms'], linewidth=style['lw'])
    ax.set_xlabel(xlbl)
    ax.set_ylabel(r'Mean $\bar{F}_{kt}$  (↓ better)')
    ax.set_title('PCRLB tracking uncertainty', fontweight='semibold')
    # ★ 핵심 변경: 자동 스케일 (hover가 포함되면 너무 넓어지므로 상한 클리핑)
    if fkt_vals:
        lo, hi = min(fkt_vals), max(fkt_vals)
        span = max(hi - lo, 2.0)
        pad  = span * 0.15
        ax.set_ylim(max(0, lo - pad), hi + pad)

    fig.suptitle(title or os.path.basename(sweep_dir), fontsize=14, fontweight='bold')
    _shared_legend(fig, axes, bottom_margin=0.12)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


# ═══════════════════════════════════════════════════════════════════════════════
#  main
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    # 1. 변경 없이 복사할 그래프
    COPY_FILES = [
        'energy_overview.png',
        'fkt_overview.png',
        'system_architecture.png',
        'timestep_flow.png',
        # seed100, seed200 에피소드 (seed42만 수정 대상이지만 전부 재생성해도 됨)
    ]
    for fname in COPY_FILES:
        src = os.path.join(ARCHIVE_FIGURES, fname)
        dst = os.path.join(OUT_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            print(f'[copy]  {fname}')

    # 2. Episode 그래프 전체 재생성 (UAV 경로 개선 적용)
    METHOD_DIRS = {
        'mappo_aai':  'mappo+aai',
        'mappo_only': 'mappo',
        'naive':      'naive_greedy',
        'llm_aai':    'llm_aai',
    }
    SEEDS = [42, 100, 200]
    for dir_name, method_key in METHOD_DIRS.items():
        ep_dir = os.path.join(ARCHIVE_EVAL, dir_name, 'episodes')
        if not os.path.isdir(ep_dir):
            print(f'[warn] episode dir not found: {ep_dir}')
            continue
        for seed in SEEDS:
            npz_fname = f'{method_key}_seed{seed}.npz'
            npz_path  = os.path.join(ep_dir, npz_fname)
            if not os.path.exists(npz_path):
                print(f'[warn] not found: {npz_path}')
                continue
            # 출력 파일명 (기존 archive 파일명과 동일하게)
            out_fname = f'episode_{dir_name}_{method_key}_seed{seed}.png'
            out_path  = os.path.join(OUT_DIR, out_fname)
            plot_episode(npz_path, out_path)

    # 3. Sweep 그래프 재생성 (y축 자동 스케일 적용)
    SWEEP_COMBINED = os.path.join(ARCHIVE_EVAL, '_combined')
    for sweep_name, (xlbl, ttl) in SWEEP_INFO.items():
        sweep_dir = os.path.join(SWEEP_COMBINED, sweep_name)
        out_path  = os.path.join(OUT_DIR, f'{sweep_name}_detailed.png')
        plot_sweep(sweep_dir, out_path, x_label=xlbl, title=ttl)


if __name__ == '__main__':
    main()
