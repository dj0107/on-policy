"""
plot_episode.py
===============
단일 에피소드의 상세 로그(npz)를 읽어 multi-panel 시각화를 생성.

교수님 피드백 반영:
- "데이터를 그래프나 수치로만 보여주지 말고, 실험 환경 setting을 realistic하게
  설정한 상황을 시각적으로 보여주면 좋을 듯 싶네요"
→ 다음 패널을 한 figure에 모음:
   1. Top-down trajectory map  (UAV / target / risk zone / BS, 시간에 따라 fade)
   2. Energy timeline (per-UAV stack + total)
   3. Tracking accuracy F_kt timeline (per target)
   4. AAI parameter timeline (W_kt, p_ut, eps_kt) + detection α heatmap
   5. UAV-target distance & assignment bar
   6. Snapshot frames (at t=0, 25, 50, 75, 99)

Usage:
    python plot_episode.py --episode_npz /path/to/episode.npz --out_path /path/to/fig.png
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors

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

UAV_CMAP = plt.cm.viridis
TARGET_CMAP = plt.cm.plasma
RISK_COLOR = '#c62828'
BS_COLOR = '#2e7d32'


def _make_uav_colors(n):
    return [UAV_CMAP(i / max(n - 1, 1)) for i in range(n)]


def _make_target_colors(k):
    # plasma 0.0~0.7 사이 (너무 노란 끝부분 회피)
    return [TARGET_CMAP(0.15 + 0.55 * i / max(k - 1, 1)) for i in range(k)]


def _draw_map_layout(ax, log, env_meta, time_range=None):
    """Top-down trajectory + risk zones + BS. 한 ax 사용."""
    map_min = float(env_meta['map_min'])
    map_max = float(env_meta['map_max'])
    bs_pos = env_meta['bs_pos']
    risk_zones = env_meta['critical_zones']
    num_uavs = int(env_meta['num_uavs'])
    num_targets = int(env_meta['num_targets'])

    uav_pos = log['uav_pos']        # (T, U, 2)
    tgt_pos = log['target_pos']     # (T, K, 2)
    tgt_est = log['target_pos_est']

    if time_range is None:
        t0, t1 = 0, len(uav_pos)
    else:
        t0, t1 = time_range

    # 위험 구역 (회색→빨강 그라디언트로 영향권 표시)
    for cz in risk_zones:
        for r, alpha in [(120.0, 0.06), (80.0, 0.08), (40.0, 0.13)]:
            ax.add_patch(plt.Circle(cz, r, facecolor=RISK_COLOR,
                                    alpha=alpha, linewidth=0))
        ax.add_patch(plt.Circle(cz, 25, facecolor=RISK_COLOR,
                                alpha=0.45, linewidth=1.2,
                                edgecolor=RISK_COLOR))
        ax.text(cz[0], cz[1] + 35, 'risk\nzone', ha='center', va='bottom',
                fontsize=8, color=RISK_COLOR, fontweight='semibold')

    # Base Station
    ax.plot(*bs_pos, marker='s', color=BS_COLOR, markersize=10,
            markeredgecolor='white', markeredgewidth=1.2, zorder=5)
    ax.text(bs_pos[0], bs_pos[1] - 35, 'BS', ha='center', va='top',
            fontsize=9, color=BS_COLOR, fontweight='bold')

    # UAV 궤적: 시간에 따라 점점 진해지는 line (fade-in)
    uav_colors = _make_uav_colors(num_uavs)
    for u in range(num_uavs):
        traj = uav_pos[t0:t1, u, :]   # (T, 2)
        # LineCollection으로 시간에 따른 alpha 변화
        segs = np.stack([traj[:-1], traj[1:]], axis=1)  # (T-1, 2, 2)
        alphas = np.linspace(0.15, 0.95, len(segs))
        lc = LineCollection(segs, colors=[(*uav_colors[u][:3], a) for a in alphas],
                            linewidths=1.4, capstyle='round')
        ax.add_collection(lc)
        # 시작/끝 마커
        ax.plot(traj[0, 0], traj[0, 1], 'o', color=uav_colors[u],
                markersize=5, markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.7, zorder=4)
        ax.plot(traj[-1, 0], traj[-1, 1], '^', color=uav_colors[u],
                markersize=10, markeredgecolor='white', markeredgewidth=1.2,
                zorder=6, label=f'UAV {u}')

    # 타겟 궤적: 진한 곡선 + 추정치 (점선)
    tgt_colors = _make_target_colors(num_targets)
    for k in range(num_targets):
        traj = tgt_pos[t0:t1, k, :]
        traj_est = tgt_est[t0:t1, k, :]
        ax.plot(traj[:, 0], traj[:, 1], '-', color=tgt_colors[k],
                linewidth=2.0, alpha=0.85, zorder=3,
                label=f'Target {k} (true)')
        ax.plot(traj_est[:, 0], traj_est[:, 1], ':', color=tgt_colors[k],
                linewidth=1.3, alpha=0.65, zorder=3)
        # 시작/끝 마커
        ax.plot(traj[0, 0], traj[0, 1], 's', color=tgt_colors[k],
                markersize=8, markeredgecolor='white', markeredgewidth=0.8,
                alpha=0.7, zorder=4)
        ax.plot(traj[-1, 0], traj[-1, 1], '*', color=tgt_colors[k],
                markersize=14, markeredgecolor='white', markeredgewidth=1.2,
                zorder=6)

    ax.set_xlim(map_min - 30, map_max + 30)
    ax.set_ylim(map_min - 30, map_max + 30)
    ax.set_aspect('equal')
    ax.set_xlabel('x  [m]')
    ax.set_ylabel('y  [m]')

    # 맵 경계
    ax.add_patch(plt.Rectangle((map_min, map_min),
                                map_max - map_min, map_max - map_min,
                                fill=False, edgecolor='#444',
                                linewidth=1.2, linestyle='-', zorder=1))

    # 범례 — axes 바로 위(외부)에 가로 배치
    handles = [
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#1f5fa8',
               markersize=10, label='UAV (final)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#1f5fa8',
               markersize=6, alpha=0.7, label='UAV (start)'),
        Line2D([0], [0], color=tgt_colors[0], linewidth=2, label='Target (true)'),
        Line2D([0], [0], color=tgt_colors[0], linewidth=1.3,
               linestyle=':', label='Target (BS estimate)'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor=BS_COLOR,
               markersize=9, label='Base station'),
        mpatches.Patch(color=RISK_COLOR, alpha=0.45, label='Risk zone'),
    ]
    # 범례를 axes 외부 하단(x축 아래)에 배치 — 타이틀과 겹침 방지
    ax.legend(handles=handles,
              bbox_to_anchor=(0.0, -0.12, 1.0, 0.08),
              loc='upper left', mode='expand',
              ncol=3, fontsize=7.5, framealpha=0.92, borderaxespad=0)


def _draw_energy_timeline(ax, log):
    """Per-UAV stacked + total energy 시계열."""
    e_per_uav = log['energy_per_uav']     # (T, U)
    e_total = log['energy_total']          # (T,)
    T, U = e_per_uav.shape
    ts = np.arange(T)

    uav_colors = _make_uav_colors(U)
    bottom = np.zeros(T)
    for u in range(U):
        ax.fill_between(ts, bottom, bottom + e_per_uav[:, u],
                        color=uav_colors[u], alpha=0.7,
                        label=f'UAV {u}', linewidth=0)
        bottom += e_per_uav[:, u]
    ax.plot(ts, e_total, color='black', linewidth=1.6, label='total')
    ax.set_xlabel('Time step  $t$')
    ax.set_ylabel('Energy / step  [J]')
    ax.set_title('Per-step energy breakdown', fontweight='semibold')
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='upper right', fontsize=8, ncol=2)


def _draw_F_kt_timeline(ax, log, env_meta):
    """Tracking quality (F_kt → 높을수록 좋은 quality score로 변환)."""
    F = log['F_kt']  # (T, K)
    T, K = F.shape
    ts = np.arange(T)
    # log 정규화: quality = (1 - log10(1+F_kt)/log10(1001)) * 100
    # 선형 1000-cap 대비 중간 범위(10~100 m²)가 잘 보임
    quality = np.maximum(0.0,
        (1.0 - np.log10(1.0 + np.clip(F, 0.0, 1e6)) / np.log10(1001.0)) * 100.0
    )
    tgt_colors = _make_target_colors(K)
    for k in range(K):
        ax.plot(ts, quality[:, k], color=tgt_colors[k], linewidth=1.6,
                label=f'Target {k}')
    ax.set_xlabel('Time step  $t$')
    ax.set_ylabel('Tracking quality  [%]')
    ax.set_title('Tracking quality', fontweight='semibold')
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.legend(loc='lower right', fontsize=9)


def _draw_aai_panel(ax_W, ax_p, ax_alpha, log, env_meta):
    """AAI 출력 (W_kt, p_ut) + detection heatmap"""
    W = log['W_kt']     # (T, K)
    P = log['p_ut']     # (T, U)
    A = log['alpha']    # (T, U, K)
    T, U, K = A.shape
    ts = np.arange(T)

    tgt_colors = _make_target_colors(K)
    for k in range(K):
        ax_W.plot(ts, W[:, k], color=tgt_colors[k], linewidth=1.5,
                  label=f'Target {k}')
    ax_W.set_xlabel('Time step')
    ax_W.set_ylabel(r'$W_{k,t}$  (priority)')
    ax_W.set_title('AAI: target priority', fontweight='semibold')
    ax_W.legend(loc='best', fontsize=8)
    ax_W.grid(True, alpha=0.25, linestyle='--')

    uav_colors = _make_uav_colors(U)
    for u in range(U):
        ax_p.plot(ts, P[:, u], color=uav_colors[u], linewidth=1.4,
                  alpha=0.85, label=f'UAV {u}')
    ax_p.set_xlabel('Time step')
    ax_p.set_ylabel(r'$p_{u,t}$  [W]')
    ax_p.set_title('AAI: TX power per UAV', fontweight='semibold')
    ax_p.legend(loc='best', fontsize=8, ncol=2)
    ax_p.grid(True, alpha=0.25, linestyle='--')

    # 탐지 heatmap: 각 UAV가 각 타겟을 탐지한 시계열 (U*K rows × T cols)
    heat = A.transpose(1, 2, 0).reshape(U * K, T)  # (U*K, T)
    ax_alpha.imshow(heat, aspect='auto', cmap='Greens',
                    interpolation='nearest', vmin=0, vmax=1)
    yticks = []
    ylabels = []
    for u in range(U):
        for k in range(K):
            yticks.append(u * K + k)
            ylabels.append(f'U{u}-T{k}')
    ax_alpha.set_yticks(yticks)
    ax_alpha.set_yticklabels(ylabels, fontsize=7)
    ax_alpha.set_xlabel('Time step')
    ax_alpha.set_title(r'Detection $\alpha_{u,k,t}$', fontweight='semibold')


def _draw_snapshots(fig, log, env_meta, n_snaps=5):
    """시간축에서 균등 간격으로 snapshot frames 추가"""
    map_min = float(env_meta['map_min'])
    map_max = float(env_meta['map_max'])
    bs_pos = env_meta['bs_pos']
    risk_zones = env_meta['critical_zones']
    num_uavs = int(env_meta['num_uavs'])
    num_targets = int(env_meta['num_targets'])

    uav_pos = log['uav_pos']
    tgt_pos = log['target_pos']
    alpha = log['alpha']
    T = uav_pos.shape[0]

    snap_ts = np.linspace(0, T - 1, n_snaps).astype(int)
    uav_colors = _make_uav_colors(num_uavs)
    tgt_colors = _make_target_colors(num_targets)

    # bottom row of snapshots
    gs = fig.add_gridspec(nrows=1, ncols=n_snaps, top=0.18, bottom=0.02,
                          left=0.05, right=0.97, wspace=0.15)
    for i, t in enumerate(snap_ts):
        ax = fig.add_subplot(gs[0, i])
        # risk
        for cz in risk_zones:
            ax.add_patch(plt.Circle(cz, 25, facecolor=RISK_COLOR,
                                    alpha=0.5, linewidth=0))
        ax.plot(*bs_pos, 's', color=BS_COLOR, markersize=6, zorder=4)
        # uav
        for u in range(num_uavs):
            ax.plot(uav_pos[t, u, 0], uav_pos[t, u, 1], '^',
                    color=uav_colors[u], markersize=6,
                    markeredgecolor='white', markeredgewidth=0.5, zorder=5)
        # target
        for k in range(num_targets):
            ax.plot(tgt_pos[t, k, 0], tgt_pos[t, k, 1], '*',
                    color=tgt_colors[k], markersize=10,
                    markeredgecolor='white', markeredgewidth=0.6, zorder=5)
            # detected UAV ↔ target line
            for u in range(num_uavs):
                if alpha[t, u, k] > 0.5:
                    ax.plot([uav_pos[t, u, 0], tgt_pos[t, k, 0]],
                            [uav_pos[t, u, 1], tgt_pos[t, k, 1]],
                            '-', color=tgt_colors[k], alpha=0.35,
                            linewidth=0.8, zorder=2)
        ax.set_xlim(map_min, map_max)
        ax.set_ylim(map_min, map_max)
        ax.set_aspect('equal')
        ax.set_title(f't = {t}', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor('#888')


def plot_episode(npz_path, out_path, title=None, t_max=None):
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
    sl = slice(0, t_max + 1) if t_max is not None else slice(None)
    log = {
        'uav_pos': d['uav_pos'][sl],
        'target_pos': d['target_pos'][sl],
        'target_pos_est': d['target_pos_est'][sl],
        'energy_per_uav': d['energy_per_uav'][sl],
        'energy_total': d['energy_total'][sl],
        'F_kt': d['F_kt'][sl],
        'alpha': d['alpha'][sl],
        'p_ut': d['p_ut'][sl],
        'W_kt': d['W_kt'][sl],
        'eps_kt': d['eps_kt'][sl],
        'team_reward': d['team_reward'][sl],
        'collisions': d['collisions'][sl],
    }
    method = str(d['method'])
    seed = int(d['seed'])

    # 메인 figure
    fig = plt.figure(figsize=(15.5, 10.5))
    # 상단: 큰 trajectory map (왼쪽 절반) + 우측 상단 panel 4개
    gs_top = fig.add_gridspec(nrows=2, ncols=4, top=0.95, bottom=0.27,
                              left=0.05, right=0.97,
                              hspace=0.42, wspace=0.32,
                              width_ratios=[1.4, 1, 1, 1])

    ax_map = fig.add_subplot(gs_top[:, 0])
    _draw_map_layout(ax_map, log, env_meta)
    ax_map.set_title('Top-down trajectory  (UAVs · targets · risk zones · BS)',
                     fontsize=10.5, fontweight='semibold')

    t_shown = t_max if t_max is not None else env_meta['max_steps']
    title_str = title or (f"Episode visualization · method={method} · seed={seed}    "
                          f"|    U={env_meta['num_uavs']}, K={env_meta['num_targets']}, "
                          f"map=[{env_meta['map_min']:.0f}, {env_meta['map_max']:.0f}] m, "
                          f"δ={env_meta['dt']}s, T={t_shown}")
    fig.suptitle(title_str, fontsize=11.5, fontweight='bold', y=0.99)

    ax_e = fig.add_subplot(gs_top[0, 1])
    _draw_energy_timeline(ax_e, log)
    ax_F = fig.add_subplot(gs_top[1, 1])
    _draw_F_kt_timeline(ax_F, log, env_meta)

    ax_W = fig.add_subplot(gs_top[0, 2])
    ax_p = fig.add_subplot(gs_top[1, 2])
    ax_alpha = fig.add_subplot(gs_top[:, 3])
    _draw_aai_panel(ax_W, ax_p, ax_alpha, log, env_meta)

    # 하단: snapshot row
    _draw_snapshots(fig, log, env_meta, n_snaps=5)
    fig.text(0.5, 0.205, 'Time-lapse snapshots',
             ha='center', va='bottom', fontsize=10.5,
             fontweight='semibold', color='#444')

    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode_npz', type=str, required=True)
    parser.add_argument('--out_path', type=str, required=True)
    parser.add_argument('--title', type=str, default=None)
    parser.add_argument('--t_max', type=int, default=None,
                        help='Truncate episode to first t_max steps')
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.out_path) or '.', exist_ok=True)
    plot_episode(args.episode_npz, args.out_path, title=args.title, t_max=args.t_max)


if __name__ == '__main__':
    main()
