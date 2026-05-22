"""
_gen_priority_plots.py
======================
크리티컬존 접근 시 AAI 추적 우선도 변화 시각화 (MAPPO+AAI 버전).

출력: final_figure/aai_priority_seed{42,100,200}.png (파일 3개, 각 2개 subplot)

Subplot 1: 추적 우선도 W_kt vs 타임스텝
  - 두 타겟 색상 구분
  - 크리티컬존 진입 구간 (d_Z < 100m) 수직 음영 표시
  - 거리 임계값 기반 W_base 구간 점선 참조선

Subplot 2: 크리티컬존까지 거리 d_Z_kt vs 타임스텝
  - 두 타겟 색상 구분
  - d_Z=100, d_Z=200 수평 임계값선 (heuristic 계단 경계)
  - d_Z < 100 구간 음영
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

EVAL_DIR  = r"c:\Users\dongj\Documents\GitHub\on-policy\_archive\nalpari_v1_2026-05-21\evaluation\mappo_aai\episodes"
OUT_DIR   = r"c:\Users\dongj\Documents\GitHub\on-policy\final_figure"
SEEDS     = [42, 100, 200]

ZONE_RADIUS   = 25.0    # 크리티컬존 반경 (시각화 내부 진입 경계)
W_NEAR        = 100.0   # heuristic: d_Z < 100 → W_base = 1.5
W_MED         = 200.0   # heuristic: d_Z < 200 → W_base = 1.0
TARGET_COLORS = ['#1f5fa8', '#e07a3c']  # 파란계열, 주황계열

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'DejaVu Sans',
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'axes.linewidth': 1.0,
    'legend.fontsize': 10,
    'figure.dpi': 110,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})


def _shade_zones(ax, dZ, threshold, color, alpha, label=None):
    """dZ[:,k] < threshold 인 구간을 ax에 수직 음영 처리."""
    T = len(dZ)
    for k, c in enumerate(TARGET_COLORS[:dZ.shape[1]]):
        in_zone = dZ[:, k] < threshold
        # 연속 구간 찾기
        changes = np.diff(in_zone.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1
        if in_zone[0]:
            starts = np.concatenate([[0], starts])
        if in_zone[-1]:
            ends = np.concatenate([ends, [T]])
        for s, e in zip(starts, ends):
            ax.axvspan(s, e, color=c, alpha=alpha, linewidth=0)


def plot_priority(npz_path, out_path, seed):
    d = np.load(npz_path, allow_pickle=True)
    W   = d['W_kt']      # (T, K)
    dZ  = d['d_Z_kt']    # (T, K)
    T, K = W.shape
    ts = np.arange(T)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True,
                                    gridspec_kw={'hspace': 0.35})

    fig.suptitle(
        f'AAI Tracking Priority vs. Critical Zone Distance  ·  seed = {seed}  '
        f'(MAPPO + AAI)',
        fontsize=13, fontweight='bold'
    )

    # ── Subplot 1: W_kt 추적 우선도 ──────────────────────────────────────────
    # 크리티컬존 진입 구간 음영 (낮은 alpha로 배경처럼)
    for k in range(K):
        in_near = dZ[:, k] < W_NEAR
        changes = np.diff(in_near.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1
        if in_near[0]: starts = np.concatenate([[0], starts])
        if in_near[-1]: ends  = np.concatenate([ends, [T]])
        for s, e in zip(starts, ends):
            ax1.axvspan(s, e, color=TARGET_COLORS[k], alpha=0.10, linewidth=0)

    # 거리 임계값 기반 W_base 참조선
    ax1.axhline(1.5, color='#aaa', linewidth=0.9, linestyle=':', zorder=1,
                label='$W_{base}$ (d<100 m)')
    ax1.axhline(1.0, color='#bbb', linewidth=0.9, linestyle=':', zorder=1,
                label='$W_{base}$ (100≤d<200 m)')
    ax1.axhline(0.5, color='#ccc', linewidth=0.9, linestyle=':', zorder=1,
                label='$W_{base}$ (d≥200 m)')

    # W_kt 선
    for k in range(K):
        ax1.plot(ts, W[:, k], color=TARGET_COLORS[k],
                 linewidth=2.0, alpha=0.9,
                 label=f'Target {k}  $W_{{kt}}$')
        # 크리티컬존 진입 시점 수직 이벤트선
        entry_steps = np.where(
            (dZ[:-1, k] >= W_NEAR) & (dZ[1:, k] < W_NEAR)
        )[0] + 1
        for s in entry_steps:
            ax1.axvline(s, color=TARGET_COLORS[k], linewidth=1.2,
                        linestyle='--', alpha=0.7)
            ax1.annotate(f'T{k} enters\ncritical zone',
                         xy=(s, W[s, k]),
                         xytext=(s + 4, W[s, k] + 0.15),
                         fontsize=8.5, color=TARGET_COLORS[k],
                         arrowprops=dict(arrowstyle='->', color=TARGET_COLORS[k],
                                         lw=0.9))

    ax1.set_ylabel('Tracking priority weight  $W_{kt}$', fontsize=11)
    ax1.set_ylim(0.3, W.max() * 1.15 + 0.1)
    ax1.grid(True, alpha=0.25, linestyle='--')
    ax1.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
               fontsize=9, ncol=1, framealpha=0.9, borderaxespad=0)
    ax1.set_title('Tracking priority weight $W_{kt}$  (↑ = more tracking resources by AAI)',
                  fontsize=11)

    # ── Subplot 2: d_Z_kt 거리 ───────────────────────────────────────────────
    # d_Z < W_NEAR 구간 음영
    for k in range(K):
        in_near = dZ[:, k] < W_NEAR
        changes = np.diff(in_near.astype(int))
        starts  = np.where(changes == 1)[0] + 1
        ends    = np.where(changes == -1)[0] + 1
        if in_near[0]: starts = np.concatenate([[0], starts])
        if in_near[-1]: ends  = np.concatenate([ends, [T]])
        for s, e in zip(starts, ends):
            ax2.axvspan(s, e, color=TARGET_COLORS[k], alpha=0.10, linewidth=0)

    # heuristic 임계값 수평선
    ax2.axhline(W_NEAR, color='#e53935', linewidth=1.3, linestyle='--',
                alpha=0.7, label=f'd = {W_NEAR:.0f} m  (W↑ threshold)')
    ax2.axhline(W_MED,  color='#fb8c00', linewidth=1.0, linestyle='--',
                alpha=0.6, label=f'd = {W_MED:.0f} m  (medium priority)')

    for k in range(K):
        ax2.plot(ts, dZ[:, k], color=TARGET_COLORS[k],
                 linewidth=2.0, alpha=0.9,
                 label=f'Target {k}  $d_{{Z,kt}}$')

    ax2.set_xlabel('Time step  $t$', fontsize=11)
    ax2.set_ylabel('Distance to critical zone  [m]', fontsize=11)
    ax2.set_title('Distance to critical zone $d_{Z,kt}$', fontsize=11)
    ax2.set_ylim(bottom=0)
    ax2.grid(True, alpha=0.25, linestyle='--')
    ax2.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0),
               fontsize=9, ncol=1, framealpha=0.9, borderaxespad=0)

    ax2.set_xlim(0, T - 1)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    for seed in SEEDS:
        npz_path = os.path.join(EVAL_DIR, f'mappo+aai_seed{seed}.npz')
        out_path = os.path.join(OUT_DIR, f'aai_priority_seed{seed}.png')
        if not os.path.exists(npz_path):
            print(f'[warn] not found: {npz_path}')
            continue
        plot_priority(npz_path, out_path, seed)


if __name__ == '__main__':
    main()
