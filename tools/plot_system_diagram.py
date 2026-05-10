"""
plot_system_diagram.py
======================
시스템 작동 방식을 그림으로 표현 (회의 요구사항).

생성 그림:
- Figure 1: Hierarchical architecture (AAI/LLM ─ MAPPO ─ Environment)
- Figure 2: Time-slot operation flow (1 step의 7 단계)

학습/평가 결과와 무관 (architecture 그림이므로 코드만으로 그림).

Usage:
    python plot_system_diagram.py --out_dir /path/to/figs
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, ConnectionPatch
from matplotlib.lines import Line2D


plt.rcParams.update({
    'font.size': 10,
    'font.family': 'DejaVu Sans',
    'figure.dpi': 110,
    'savefig.dpi': 220,
    'savefig.bbox': 'tight',
})


# 색 팔레트 (논문/슬라이드 톤)
C_AAI    = '#9c4dcc'   # AAI / LLM
C_MAPPO  = '#1f5fa8'   # MAPPO
C_ENV    = '#2e7d32'   # Environment
C_PHYS   = '#5d6d7e'   # Physical layer (sensing/comm)
C_BS     = '#e07a3c'   # BS / fusion
C_REWARD = '#c62828'   # Reward / objective
C_BG     = '#fafafa'


def _box(ax, xy, w, h, text, fc, ec='black', tc='white',
         fontsize=10, fontweight='bold', boxstyle='round,pad=0.4'):
    cx, cy = xy[0] + w / 2, xy[1] + h / 2
    ax.add_patch(FancyBboxPatch(
        (xy[0], xy[1]), w, h,
        boxstyle=boxstyle,
        facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    ax.text(cx, cy, text, ha='center', va='center', color=tc,
            fontsize=fontsize, fontweight=fontweight, zorder=4)
    return (cx, cy)


def _arrow(ax, p_from, p_to, label=None, color='black', lw=1.5,
           label_offset=(0, 0), label_color=None, label_fontsize=9,
           rad=0.0, style='-|>', mutation_scale=14):
    arr = FancyArrowPatch(p_from, p_to,
                          arrowstyle=style,
                          mutation_scale=mutation_scale,
                          color=color, linewidth=lw,
                          connectionstyle=f'arc3,rad={rad}',
                          zorder=2)
    ax.add_patch(arr)
    if label:
        mx = (p_from[0] + p_to[0]) / 2 + label_offset[0]
        my = (p_from[1] + p_to[1]) / 2 + label_offset[1]
        ax.text(mx, my, label, ha='center', va='center',
                color=label_color or color, fontsize=label_fontsize,
                fontweight='semibold',
                bbox=dict(boxstyle='round,pad=0.25',
                          facecolor='white', edgecolor='none', alpha=0.9))


def plot_architecture(out_path):
    """3-layer hierarchical architecture: AAI, MAPPO, Environment."""
    fig, ax = plt.subplots(figsize=(12.5, 7.5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 70)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(C_BG)

    # ---- Layer 1: AAI / LLM (top) ----
    ax.add_patch(plt.Rectangle((1, 53), 98, 15,
                               facecolor='#f3e5f5', edgecolor=C_AAI,
                               linewidth=1.5, linestyle='--', zorder=1))
    ax.text(2.5, 65.5, 'High-level reasoner  ·  AAI / LLM',
            color=C_AAI, fontsize=12.5, fontweight='bold')

    aai_box = _box(ax, (8, 56), 30, 8.5,
                   'AAI module\n$f_\\mathrm{AAI}(\\Phi_t) \\to (\\epsilon_{k,t},\\, W_{k,t},\\, p_{u,t})$',
                   fc=C_AAI, fontsize=9.5)

    perception_box = _box(ax, (43, 56), 22, 8.5,
                          'Perception\n$\\Phi_t = \\{\\tilde S_k, \\tilde P_k, d^Z_k, E_u\\}$',
                          fc='#7e57c2', fontsize=9)

    reasoning_box = _box(ax, (70, 56), 22, 8.5,
                         'Reasoning\nrisk-aware priorities\n+ power assignment',
                         fc='#7e57c2', fontsize=9)

    _arrow(ax, (perception_box[0] - 11, perception_box[1]),
           (aai_box[0] + 15, aai_box[1]), color=C_AAI, lw=1.4)
    _arrow(ax, (perception_box[0] + 11, perception_box[1]),
           (reasoning_box[0] - 11, reasoning_box[1]),
           color=C_AAI, lw=1.4)
    _arrow(ax, (reasoning_box[0] - 11, reasoning_box[1] - 1.5),
           (aai_box[0] + 15, aai_box[1] - 1.5),
           color=C_AAI, lw=1.4, rad=-0.3)

    # ---- Layer 2: MAPPO (middle) ----
    ax.add_patch(plt.Rectangle((1, 28), 98, 22,
                               facecolor='#e3f2fd', edgecolor=C_MAPPO,
                               linewidth=1.5, linestyle='--', zorder=1))
    ax.text(2.5, 47.5, 'Low-level control  ·  MAPPO (CTDE)',
            color=C_MAPPO, fontsize=12.5, fontweight='bold')

    actor_box = _box(ax, (8, 33), 30, 10,
                     'Actor  $\\pi_\\theta(a_u | o_{u,t})$\n(decentralized, per UAV)',
                     fc=C_MAPPO, fontsize=10)
    critic_box = _box(ax, (45, 33), 30, 10,
                      'Centralized critic  $V_\\phi(s_t)$\n(uses global state)',
                      fc='#3b8edb', fontsize=10)
    buffer_box = _box(ax, (80, 33), 17, 10,
                      'PPO\nbuffer\n(GAE λ)',
                      fc='#5dade2', fontsize=9.5)

    # Actor ↔ Critic
    _arrow(ax, (actor_box[0] + 15, actor_box[1] + 3),
           (critic_box[0] - 15, critic_box[1] + 3),
           color=C_MAPPO, lw=1.2,
           label='advantage', label_offset=(0, 1.6))
    _arrow(ax, (critic_box[0] - 15, critic_box[1] - 3),
           (actor_box[0] + 15, actor_box[1] - 3),
           color=C_MAPPO, lw=1.2, rad=0.0)

    # Critic ↔ Buffer
    _arrow(ax, (critic_box[0] + 15, critic_box[1]),
           (buffer_box[0] - 8.5, buffer_box[1]),
           color=C_MAPPO, lw=1.2)
    _arrow(ax, (buffer_box[0] - 8.5, buffer_box[1] - 2),
           (critic_box[0] + 15, critic_box[1] - 2),
           color=C_MAPPO, lw=1.2, rad=0.3)

    # ---- Layer 3: Environment (bottom) ----
    ax.add_patch(plt.Rectangle((1, 2), 98, 22,
                               facecolor='#e8f5e9', edgecolor=C_ENV,
                               linewidth=1.5, linestyle='--', zorder=1))
    ax.text(2.5, 21.5, 'Environment  ·  Dec-POMDP (multi-target tracking)',
            color=C_ENV, fontsize=12.5, fontweight='bold')

    uav_box = _box(ax, (5, 7), 18, 11,
                   'UAV swarm\n$\\{q_u, v_u, E_u, p_u\\}$\nphysical motion',
                   fc=C_ENV, fontsize=9)
    target_box = _box(ax, (27, 7), 18, 11,
                      'Targets\n$\\{S_k\\}$ + risk zones\nground truth (hidden)',
                      fc='#43a047', fontsize=9)
    sense_box = _box(ax, (49, 7), 18, 11,
                     'Sensing model\n$\\mathrm{SNR}_\\mathrm{sen}, \\alpha_{u,k,t}$\nlocal EKF',
                     fc=C_PHYS, fontsize=9)
    bs_box = _box(ax, (71, 7), 26, 11,
                  'Base Station fusion\n$\\tilde S_k, \\tilde P_k, J_k \\to F_{k,t}$\nteam reward $r_t$',
                  fc=C_BS, fontsize=9)

    # Env internal arrows
    _arrow(ax, (uav_box[0] + 9, uav_box[1]),
           (target_box[0] - 9, target_box[1]),
           color=C_ENV, lw=1.0, mutation_scale=10)
    _arrow(ax, (target_box[0] + 9, target_box[1]),
           (sense_box[0] - 9, sense_box[1]),
           color=C_ENV, lw=1.0, mutation_scale=10)
    _arrow(ax, (sense_box[0] + 9, sense_box[1]),
           (bs_box[0] - 13, bs_box[1]),
           color=C_ENV, lw=1.0, mutation_scale=10)

    # ---- Inter-layer arrows ----
    # MAPPO → Env (action) - left side
    _arrow(ax, (actor_box[0] - 5, actor_box[1] - 5),
           (uav_box[0] + 4, uav_box[1] + 5.5),
           color=C_MAPPO, lw=2.0,
           label='action  $a_{u,t}$',
           label_offset=(-4, 0),
           label_color='black', label_fontsize=9)

    # Env → MAPPO (global state + reward) - right side, into critic
    _arrow(ax, (bs_box[0] - 4, bs_box[1] + 5.5),
           (critic_box[0] + 8, critic_box[1] - 5),
           color=C_ENV, lw=2.0,
           label='global state $s_t$\n+ team reward $r_t$',
           label_offset=(7, 0),
           label_color='black', label_fontsize=8.5)

    # Sense → Actor (local obs) - middle, slightly offset to avoid label overlap
    _arrow(ax, (sense_box[0], sense_box[1] + 5.5),
           (actor_box[0] + 14, actor_box[1] - 5),
           color=C_PHYS, lw=1.5,
           label='local obs $o_{u,t}$',
           label_offset=(7, 1),
           label_color='black', label_fontsize=8.5)

    # AAI → Env (parameters injected at every step) - bypasses MAPPO box
    # W_kt to targets — 왼쪽 외곽으로 큰 호
    _arrow(ax, (aai_box[0] - 14, aai_box[1] - 4),
           (target_box[0] + 4, target_box[1] + 5.5),
           color=C_AAI, lw=1.6, rad=-0.35,
           label=None)
    # p_ut to UAV — 더 바깥쪽
    _arrow(ax, (aai_box[0] - 14, aai_box[1] - 4.5),
           (uav_box[0] + 12, uav_box[1] + 5.5),
           color=C_AAI, lw=1.6, rad=-0.55,
           label=None)
    # AAI 라벨은 별도 텍스트로 명확히
    ax.text(2.5, 26, '$W_{k,t}, \\epsilon_{k,t}, p_{u,t}$',
            color=C_AAI, fontsize=10.5, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                      edgecolor=C_AAI, linewidth=1.0))

    # AAI ← Env (telemetry Φ_t) - right side, big curve from BS to perception
    _arrow(ax, (bs_box[0] + 8, bs_box[1] + 5.5),
           (perception_box[0], perception_box[1] - 4.5),
           color=C_BS, lw=1.5, rad=0.25,
           label='$\\Phi_t$  (telemetry)',
           label_offset=(8, 5),
           label_color='black', label_fontsize=9)

    # Title
    fig.suptitle('Hierarchical AAI + MAPPO architecture for energy-efficient UAV swarm tracking',
                 fontsize=13, fontweight='bold', y=0.97)

    # Legend (bottom right)
    legend_handles = [
        mpatches.Patch(color=C_AAI, label='AAI / LLM (high-level)'),
        mpatches.Patch(color=C_MAPPO, label='MAPPO (low-level RL)'),
        mpatches.Patch(color=C_ENV, label='Environment dynamics'),
        mpatches.Patch(color=C_PHYS, label='Sensing / EKF (physical)'),
        mpatches.Patch(color=C_BS, label='BS fusion / reward'),
    ]
    ax.legend(handles=legend_handles, loc='lower center',
              bbox_to_anchor=(0.5, -0.05), ncol=5, fontsize=9,
              frameon=False)

    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def plot_timestep_flow(out_path):
    """1 time slot의 7-step 운영 절차를 시각화"""
    fig, ax = plt.subplots(figsize=(13.5, 4.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(2, 26)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_facecolor(C_BG)

    steps = [
        ('1. Target\nstate update', '$S_{k,t+1} = F\\, S_{k,t} + w_{k,t}$',  C_ENV),
        ('2. UAV action\n(MAPPO)',  '$a_{u,t} \\sim \\pi_\\theta(\\cdot | o_{u,t})$\n→ velocity',    C_MAPPO),
        ('3. Sensing\n+ local EKF', 'SNR check $\\alpha_{u,k,t}$\nlocal $\\hat S_k, \\hat P_k$', C_PHYS),
        ('4. Comm. to BS',          'rate $R^c_{u,t}$\n(absence signal if $\\alpha=0$)',   C_PHYS),
        ('5. BS fusion',            'global $\\tilde S_k, \\tilde P_k$\nBFIM $J_k$, $F_{k,t}$',         C_BS),
        ('6. AAI re-call',          '$\\Phi_{t+1} \\to (\\epsilon, W, p)_{t+1}$',           C_AAI),
        ('7. Reward',               '$r_t$ (team)',                                                       C_REWARD),
    ]

    n = len(steps)
    box_w = 12
    gap = 1.4
    total_w = n * box_w + (n - 1) * gap
    start_x = (100 - total_w) / 2
    box_h = 12
    cy = 14

    centers = []
    for i, (title, formula, color) in enumerate(steps):
        x = start_x + i * (box_w + gap)
        cx = x + box_w / 2
        ax.add_patch(FancyBboxPatch(
            (x, cy - box_h / 2), box_w, box_h,
            boxstyle='round,pad=0.4',
            facecolor=color, edgecolor='black', linewidth=1.3, zorder=3))
        ax.text(cx, cy + 1.8, title, ha='center', va='center',
                color='white', fontsize=10, fontweight='bold', zorder=4)
        ax.text(cx, cy - 2.4, formula, ha='center', va='center',
                color='white', fontsize=8.5, zorder=4)
        centers.append((cx, cy))

    # Arrows between
    for (cx1, cy1), (cx2, cy2) in zip(centers[:-1], centers[1:]):
        _arrow(ax, (cx1 + box_w / 2 - 0.3, cy1),
               (cx2 - box_w / 2 + 0.3, cy2),
               color='#444', lw=1.4, mutation_scale=12)

    # Cycle back arrow (7 → 1) above the row
    _arrow(ax, (centers[-1][0], centers[-1][1] + box_h / 2 + 0.5),
           (centers[0][0], centers[0][1] + box_h / 2 + 0.5),
           color='#444', lw=1.5, rad=0.35,
           style='-|>', mutation_scale=14)
    ax.text(50, 24, 'next time slot $t \\to t+1$',
            ha='center', va='center', fontsize=10, color='#444',
            fontweight='semibold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                      edgecolor='#888', linewidth=1.0))

    fig.suptitle('Time-slot $\\delta$ operation: 7-step procedure',
                 fontsize=12.5, fontweight='bold', y=1.05)
    fig.savefig(out_path)
    plt.close(fig)
    print(f'[saved] {out_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', type=str, default='/tmp/diagrams')
    args = parser.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plot_architecture(os.path.join(args.out_dir, 'system_architecture.png'))
    plot_timestep_flow(os.path.join(args.out_dir, 'timestep_flow.png'))


if __name__ == '__main__':
    main()
