import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

# ============================================================================
# (12) 버전 - Dec-POMDP, AAI 출력: epsilon_kt, W_kt, p_ut
# (12) → (현재) 재점검 후 수정:
#   - SNR 단위 명시 (dB ↔ linear)
#   - 측정 잡음 분산에 floor (수치 안정성)
#   - 보상 스케일 재조정 (충돌이 학습 신호 지배 방지)
#   - 타겟/UAV 위치 클리핑 (식 28b/28c 강제)
#   - alpha=0 시 e_comm 절감 (absence signal 모델)
#   - 식 (28f) 미탐지 페널티 추가
#   - 학습 시 AAI 파라미터 randomization (domain randomization)
#   - 평가 시 외부 AAI(LLM) 주입 hook
#   - 시각화/로깅용 _episode_log 누적
# ============================================================================


class Target:
    """타겟 상태, EKF용 fused estimate, BFIM 관리"""
    def __init__(self, target_id, initial_pos, initial_velocity):
        self.id = target_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array(initial_velocity, dtype=np.float32)
        self.S_global = np.concatenate([self.pos, self.velocity])
        self.P_global = np.eye(4, dtype=np.float32) * 1.0
        self.J_matrix = np.eye(4, dtype=np.float32) * 0.1
        self.F_kt = 0.0
        self.measurement_info = np.zeros((4, 4), dtype=np.float32)
        self.epsilon_kt = 5.0
        self.W_kt = 1.0
        self.d_Z_kt = float('inf')

    def step(self, dt, Q=None, map_min=-500.0, map_max=500.0):
        """식 (1) + 경계 반사"""
        self.pos += self.velocity * dt
        if Q is not None:
            w_kt = np.random.multivariate_normal(np.zeros(4), Q).astype(np.float32)
            self.pos += w_kt[:2]
            self.velocity += w_kt[2:]
        for d in range(2):
            if self.pos[d] < map_min:
                self.pos[d] = map_min
                self.velocity[d] = abs(self.velocity[d])
            elif self.pos[d] > map_max:
                self.pos[d] = map_max
                self.velocity[d] = -abs(self.velocity[d])
        v_norm = np.linalg.norm(self.velocity)
        v_max_target = 6.0
        if v_norm > v_max_target:
            self.velocity = self.velocity / v_norm * v_max_target
        self.S_global = np.concatenate([self.pos, self.velocity])


class UAV:
    """UAV 행동/에너지. AAI가 p_ut 외부 주입"""
    def __init__(self, uav_id, initial_pos, altitude, max_speed=10.0, max_energy=100000.0):
        self.id = uav_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.H = altitude
        self.v_max = max_speed
        self.energy = max_energy
        self.max_energy = max_energy
        self.p_ut = 1.0
        self.tau_s_ratio = 0.5
        self.is_detected_per_target = {}
        self.last_e_tot = 0.0
        self.last_e_sen = 0.0
        self.last_e_comm = 0.0
        self.last_e_move = 0.0
        self.last_R_c = 0.0
        self.last_snr_sen = 0.0
        self.local_estimates = {}

        self.f_c = 2.4e9
        self.lam = 3e8 / self.f_c
        self.G_t = 1.0
        self.G_r = 30.0    # advisor 피드백 (p_ut=1, SNR=20dB, alt=15m)과 결합하여 detection 수평 ~103m
        self.sigma = 1.0
        self.N0 = 1e-14
        self.c1, self.c2 = 11.9, 0.13
        self.eta_LoS, self.eta_NLoS = 1.0, 100.0
        self.B_b = 1e6

        self.P0, self.P1 = 3.4, 20.0
        self.u_tip, self.v0 = 60.0, 5.4
        self.d0, self.rho, self.s0, self.A = 0.3, 1.225, 0.03, 0.28

    def apply_action(self, action, dt, map_min=-500.0, map_max=500.0):
        delta_q = np.array(action, dtype=np.float32) * self.v_max * dt
        delta_norm = np.linalg.norm(delta_q)
        if delta_norm > self.v_max * dt:
            delta_q = (delta_q / delta_norm) * self.v_max * dt
        new_pos = self.pos + delta_q
        boundary_hit = False
        for d in range(2):
            if new_pos[d] < map_min:
                new_pos[d] = map_min
                boundary_hit = True
            elif new_pos[d] > map_max:
                new_pos[d] = map_max
                boundary_hit = True
        actual_delta = new_pos - self.pos
        self.velocity = actual_delta / dt if dt > 0 else np.zeros(2)
        self.pos = new_pos
        return boundary_hit

    def get_sensing_model(self, target_pos, dt):
        d_uk = math.sqrt(np.sum((self.pos - target_pos)**2) + self.H**2)
        tau_s = self.tau_s_ratio * dt
        snr_sen = (self.p_ut * self.G_t * self.G_r * (self.lam**2) * self.sigma * tau_s) \
                  / (self.N0 * (4 * math.pi)**3 * (d_uk**4))
        e_sen = self.p_ut * tau_s
        return snr_sen, e_sen

    def get_comm_model(self, bs_pos, dt, alpha_ukt):
        d_ut_2d = np.linalg.norm(self.pos - bs_pos)
        d_ut_3d = math.sqrt(d_ut_2d**2 + self.H**2)
        theta_ut = math.atan(self.H / d_ut_2d) if d_ut_2d > 0 else math.pi/2
        p_los = 1 / (1 + self.c1 * math.exp(-self.c2 * (math.degrees(theta_ut) - self.c1)))
        p_nlos = 1 - p_los
        h_c = ((p_los * self.eta_LoS + p_nlos * self.eta_NLoS) * (self.lam**2)) \
              / ((4 * math.pi * d_ut_3d)**2 + 1e-12)
        snr_comm = (self.p_ut * h_c) / self.N0
        tau_c = (1 - self.tau_s_ratio) * dt
        R_c_ut = alpha_ukt * tau_c * self.B_b * math.log2(1 + snr_comm)
        # alpha=0이면 absence signal: tau_c의 1%만 송신
        if alpha_ukt == 1:
            e_comm = self.p_ut * tau_c
        else:
            e_comm = self.p_ut * tau_c * 0.01
        return e_comm, R_c_ut

    def get_mobility_energy(self, dt):
        v_norm = np.linalg.norm(self.velocity)
        if v_norm == 0:
            return (self.P0 + self.P1) * dt
        term1 = self.P0 * (1 + (3 * v_norm**2) / (self.u_tip**2))
        inner_sqrt = math.sqrt(1 + (v_norm**4) / (4 * self.v0**4))
        term2 = self.P1 * math.sqrt(max(inner_sqrt - (v_norm**2) / (2 * self.v0**2), 0))
        term3 = 0.5 * self.d0 * self.rho * self.s0 * self.A * (v_norm**3)
        return (term1 + term2 + term3) * dt


class UAVTrackingEnv(gym.Env):
    """Dec-POMDP MTT 환경"""

    def __init__(self, num_uavs=5, num_targets=2, dt=1.0,
                 use_aai=True, aai_callback=None,
                 randomize_aai=False,
                 sigma_w_sq=0.1, log_episode=False,
                 num_critical_zones=2,
                 aai_ema_alpha=0.2):
        """
        aai_ema_alpha: AAI 출력의 EMA 계수.
            1.0 = EMA 없음 (원본 동작, 매 step 새 값 즉시 반영)
            0.2 = 새 값 20%만 반영, 80%는 이전 값 유지 (부드럽게 변화)
            0.0 = 절대 안 바뀜 (초기값 고정)
            학습 시 randomize_aai=True이면 자동으로 우회됨.
        """
        super(UAVTrackingEnv, self).__init__()
        self.aai_ema_alpha = aai_ema_alpha
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.dt = dt
        self.max_steps = 150

        self.use_aai = use_aai
        self.aai_callback = aai_callback
        self.randomize_aai = randomize_aai
        self.log_episode = log_episode

        # 위험구역 (맵 300x300 기준 재조정)
        all_zones = [
            np.array([100.0,  100.0], dtype=np.float32),
            np.array([-80.0,  120.0], dtype=np.float32),
            np.array([120.0,  -90.0], dtype=np.float32),
            np.array([-100.0, -80.0], dtype=np.float32),
        ]
        self.critical_zones = all_zones[:num_critical_zones]

        # 20 dB → 선형
        self.snr_threshold_dB = 20.0
        self.snr_threshold = 10 ** (self.snr_threshold_dB / 10.0)
        self.R_c_threshold = 1.218e6
        self.d_min = 5.0
        self.map_min, self.map_max = -150.0, 150.0

        # 보상 가중치
        self.lam1 = 2.0    # 추적 정확도 (F_kt)
        self.lam2 = 15.0   # swarm 전체 실패 페널티
        self.lam3 = 1.0    # 충돌/경계 페널티
        self.lam4 = 0.0    # 통신 페널티 (삭제됨)
        self.lam5 = 0.0    # untracked 페널티 (제거)
        self.lam6 = 0.0    # r_approach shaping (제거)
        self.lam7 = 0.0    # r_detect 양수 보상 (제거)

        self.sigma_w_sq = sigma_w_sq
        self.sigma_r0_sq = 10.0
        self.sigma_theta0_sq = 1e-4
        self.sigma_r_sq_floor = 1e-3
        self.sigma_theta_sq_floor = 1e-7
        self._precompute_ekf_matrices()

        self.uavs = [UAV(i, [0, 0], altitude=80.0) for i in range(self.num_uavs)]
        self.targets = [Target(i, [0, 0], [0, 0]) for i in range(self.num_targets)]
        self.assignment = {u: 0 for u in range(self.num_uavs)}

        dummy_local, dummy_global = self._get_obs()
        local_dim = len(dummy_local[0])
        global_dim = len(dummy_global)

        self.action_space = [
            spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
            for _ in range(self.num_uavs)
        ]
        self.observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(local_dim,), dtype=np.float32)
            for _ in range(self.num_uavs)
        ]
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32)
            for _ in range(self.num_uavs)
        ]
        self._episode_log = None
        self.reset()

    def _precompute_ekf_matrices(self):
        dt = self.dt
        self.F_mat = np.array([[1, 0, dt, 0], [0, 1, 0, dt],
                                [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.F_inv = np.linalg.inv(self.F_mat)
        q11, q13, q33 = dt**4/4, dt**3/2, dt**2
        self.Q = np.array([[q11, 0, q13, 0], [0, q11, 0, q13],
                           [q13, 0, q33, 0], [0, q13, 0, q33]], dtype=np.float32) * self.sigma_w_sq
        self.Q += np.eye(4) * 1e-6
        self.Q_inv = np.linalg.inv(self.Q)
        self.Lambda = np.diag([1.0, 1.0, 0.0, 0.0]).astype(np.float32)

    def seed(self, seed=None):
        np.random.seed(seed if seed is not None else 1)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.time_slot = 0
        self.bs_pos = np.array([0.0, 0.0], dtype=np.float32)

        # 1) 타겟 먼저 생성 — 맵 전체 랜덤 (±150m)
        self.targets = [
            Target(i,
                   [np.random.uniform(self.map_min, self.map_max),
                    np.random.uniform(self.map_min, self.map_max)],
                   [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
            for i in range(self.num_targets)
        ]

        # 2) UAV를 타겟 근처 센싱 가능 범위(10~50m) 내 랜덤 배치
        #    수평 50m + 고도 90m → 3D ~103m, SNR >> threshold (탐지 보장)
        uavs_per_target = {t: [] for t in range(self.num_targets)}
        for u in range(self.num_uavs):
            uavs_per_target[u % self.num_targets].append(u)

        uav_positions = [None] * self.num_uavs
        for t_idx, target in enumerate(self.targets):
            for u_idx in uavs_per_target[t_idx]:
                for _ in range(30):
                    angle = np.random.uniform(0, 2.0 * math.pi)
                    radius = np.random.uniform(10.0, 50.0)
                    pos = target.pos + np.array([radius * math.cos(angle),
                                                 radius * math.sin(angle)], dtype=np.float32)
                    pos = np.clip(pos, self.map_min, self.map_max)
                    if all(uav_positions[j] is None or
                           np.linalg.norm(pos - uav_positions[j]) >= self.d_min
                           for j in range(self.num_uavs)):
                        break
                uav_positions[u_idx] = pos

        self.uavs = [
            UAV(i, uav_positions[i].tolist(), altitude=80.0)
            for i in range(self.num_uavs)
        ]

        for target in self.targets:
            target.d_Z_kt = min(np.linalg.norm(target.pos - cz) for cz in self.critical_zones)

        # EMA 버퍼 초기화 (이전 에피소드의 _aai_prev_* 잔재 제거)
        for attr in list(vars(self).keys()):
            if attr.startswith('_aai_prev_'):
                delattr(self, attr)

        self._invoke_aai()

        if self.log_episode:
            self._episode_log = {
                'uav_pos': [], 'target_pos': [], 'target_pos_est': [],
                'energy_per_uav': [], 'energy_total': [],
                'F_kt': [], 'alpha': [], 'p_ut': [], 'W_kt': [],
                'eps_kt': [], 'd_Z_kt': [], 'team_reward': [],
                'collisions': [], 'assignment': [],
            }
            self._log_step(team_reward=0.0, n_collisions=0)

        local_obs, global_state = self._get_obs()
        obs = np.array(local_obs, dtype=np.float32)
        share_obs = np.array([global_state] * self.num_uavs, dtype=np.float32)
        obs = self._sanitize(obs)
        share_obs = self._sanitize(share_obs)
        return obs, share_obs

    def _sanitize(self, x):
        x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(x, -1e4, 1e4)

    def _invoke_aai(self):
        # ─────────────────────────────────────────────────────────────────
        # EMA 처리 (길 3): AAI가 매 step 호출되더라도, 출력값이 부드럽게 바뀌도록
        # 새 raw 값을 _aai_buf_*에 받고, 실제 적용은 이전 값과 EMA 가중평균.
        # alpha=1.0 이면 EMA 없음(원본 동작), 0.0이면 절대 안 바뀜.
        # 학습 시 randomize_aai=True인 경우엔 매 step 완전 새 값이 필요하므로
        # EMA 우회. 평가/실제 운용 시에만 적용.
        # ─────────────────────────────────────────────────────────────────
        ema_alpha = getattr(self, 'aai_ema_alpha', 0.2)  # 0.2 = "20%만 새 값 반영"

        # 새 값 산출 (기존 로직 그대로)
        if self.aai_callback is not None:
            self.aai_callback(self)
            self._build_assignment()
        elif self.use_aai:
            self._run_aai_heuristic()
        else:
            for t in self.targets:
                t.W_kt = 1.0
                t.epsilon_kt = 5.0
                t.d_Z_kt = min(np.linalg.norm(t.S_global[:2] - cz) for cz in self.critical_zones)
            for u in self.uavs:
                u.p_ut = 1.0
            self._build_assignment()

        if self.randomize_aai:
            # 학습 중 domain randomization: EMA 우회, 매 step 완전 새 값
            for t in self.targets:
                t.W_kt = float(np.random.uniform(0.5, 3.5))
                t.epsilon_kt = float(np.random.uniform(1.0, 10.0))
            for u in self.uavs:
                u.p_ut = float(np.random.uniform(0.3, 3.0))
            # randomize 중에는 _aai_prev_* 초기화 X (EMA 비활성)
            return

        # EMA 적용: 첫 step이면 그대로, 이후엔 이전 값과 섞기
        if ema_alpha < 1.0 and self.time_slot > 0:
            for t in self.targets:
                key_W = f'_aai_prev_W_{t.id}'
                key_eps = f'_aai_prev_eps_{t.id}'
                if hasattr(self, key_W):
                    t.W_kt = ema_alpha * t.W_kt + (1 - ema_alpha) * getattr(self, key_W)
                    t.epsilon_kt = ema_alpha * t.epsilon_kt + (1 - ema_alpha) * getattr(self, key_eps)
                setattr(self, key_W, float(t.W_kt))
                setattr(self, key_eps, float(t.epsilon_kt))
            for u in self.uavs:
                key_p = f'_aai_prev_p_{u.id}'
                if hasattr(self, key_p):
                    u.p_ut = ema_alpha * u.p_ut + (1 - ema_alpha) * getattr(self, key_p)
                setattr(self, key_p, float(u.p_ut))
        else:
            # 첫 step: 현재 값을 prev로 초기화만
            for t in self.targets:
                setattr(self, f'_aai_prev_W_{t.id}', float(t.W_kt))
                setattr(self, f'_aai_prev_eps_{t.id}', float(t.epsilon_kt))
            for u in self.uavs:
                setattr(self, f'_aai_prev_p_{u.id}', float(u.p_ut))

    def _build_assignment(self):
        unassigned = list(range(self.num_uavs))
        self.assignment = {}
        for t_idx, target in enumerate(self.targets):
            if not unassigned:
                break
            est_pos = target.S_global[:2]
            best_uav = min(unassigned,
                           key=lambda u: np.linalg.norm(self.uavs[u].pos - est_pos))
            self.assignment[best_uav] = t_idx
            unassigned.remove(best_uav)
        for u_idx in unassigned:
            best_t = min(range(self.num_targets),
                         key=lambda t: np.linalg.norm(
                             self.uavs[u_idx].pos - self.targets[t].S_global[:2]))
            self.assignment[u_idx] = best_t

    def _run_aai_heuristic(self):
        for target in self.targets:
            est_pos = target.S_global[:2]
            target.d_Z_kt = min(np.linalg.norm(est_pos - cz) for cz in self.critical_zones)
        for target in self.targets:
            if target.d_Z_kt < 100.0:
                w_base = 1.5
                target.epsilon_kt = 2.0
            elif target.d_Z_kt < 200.0:
                w_base = 1.0
                target.epsilon_kt = 5.0
            else:
                w_base = 0.5
                target.epsilon_kt = 5.0
            # 추적 품질 urgency: F_kt 상승 시 W_kt 자동 증가 → pile-on 방지
            w_track = 1.5 * min(1.0, target.F_kt / 500.0)
            target.W_kt = w_base + w_track
        self._build_assignment()
        # p_ut: H=90m에서 slant_min=90m, p_ut=0.5W → SNR=90<100 → 탐지 불가
        # 근거리도 1W 유지, 200m 초과 시만 2W로 부스트
        for u_idx, uav in enumerate(self.uavs):
            target = self.targets[self.assignment[u_idx]]
            est_pos = target.S_global[:2]
            dist = np.linalg.norm(uav.pos - est_pos)
            if dist > 200.0:
                uav.p_ut = 2.0
            else:
                uav.p_ut = 1.0

    def _get_obs(self):
        POS_SCALE = 150.0
        E_SCALE = 100000.0
        F_SCALE = 1000.0
        V_SCALE = 10.0

        global_parts = []
        for u in self.uavs:
            global_parts.extend(u.pos / POS_SCALE)
            global_parts.append(u.energy / E_SCALE)
        for t in self.targets:
            global_parts.extend(t.S_global[:2] / POS_SCALE)
            v_clipped = np.clip(t.S_global[2:], -V_SCALE, V_SCALE)
            global_parts.extend(v_clipped / V_SCALE)
            global_parts.append(min(t.F_kt, 1000.0) / F_SCALE)
            global_parts.append(t.d_Z_kt / POS_SCALE)
            global_parts.append(t.epsilon_kt / 10.0)
            global_parts.append(t.W_kt / 5.0)
        global_state = np.array(global_parts, dtype=np.float32)

        local_obs_list = []
        for u_idx, uav in enumerate(self.uavs):
            obs = list(uav.pos / POS_SCALE)
            obs.append(uav.energy / E_SCALE)
            for k_idx, target in enumerate(self.targets):
                delta_q = (target.S_global[:2] - uav.pos) / POS_SCALE
                obs.extend(delta_q)
                v_clipped = np.clip(target.S_global[2:], -V_SCALE, V_SCALE)
                obs.extend(v_clipped / V_SCALE)
                obs.append(target.W_kt / 5.0)
                obs.append(target.epsilon_kt / 10.0)
                # 직전 step 탐지 여부 (binary)
                obs.append(float(uav.is_detected_per_target.get(k_idx, 0)))
                # ★ 정규화된 SNR: 탐지 실패 시에도 "얼마나 가까이 가야 하는지" 연속 신호 제공
                #   SNR이 클수록 1에 가깝고, threshold 아래이면 0~1 사이 연속값
                #   tanh(snr / snr_th)로 정규화 → [0, 1] 범위
                snr = uav.last_snr_sen if self.assignment.get(u_idx, -1) == k_idx else 0.0
                snr_normalized = float(math.tanh(snr / max(self.snr_threshold, 1e-9)))
                obs.append(snr_normalized)
            t_idx = self.assignment.get(u_idx, 0)
            for k in range(self.num_targets):
                obs.append(1.0 if k == t_idx else 0.0)
            local_obs_list.append(np.array(obs, dtype=np.float32))
        return local_obs_list, global_state

    def step(self, actions):
        self.time_slot += 1
        for target in self.targets:
            target.step(self.dt, Q=self.Q, map_min=self.map_min, map_max=self.map_max)

        boundary_violations = 0
        for i, uav in enumerate(self.uavs):
            hit = uav.apply_action(actions[i], self.dt,
                                   map_min=self.map_min, map_max=self.map_max)
            if hit:
                boundary_violations += 1

        self._update_local_tracking_and_energy()
        self._bs_fusion()
        self._invoke_aai()
        n_collisions = self._count_collisions()
        team_reward = self._calculate_team_reward(n_collisions, boundary_violations)
        rewards = np.array([[team_reward]] * self.num_uavs, dtype=np.float32)

        local_obs, global_state = self._get_obs()
        obs = np.array(local_obs, dtype=np.float32)
        share_obs = np.array([global_state] * self.num_uavs, dtype=np.float32)
        obs = self._sanitize(obs)
        share_obs = self._sanitize(share_obs)
        rewards = np.clip(np.nan_to_num(rewards, nan=-100.0), -1e3, 1e3)

        dones = np.array([self.time_slot >= self.max_steps] * self.num_uavs, dtype=bool)
        infos = [{
            'uav_energy': u.energy,
            'team_reward': team_reward,
            'last_e_tot': u.last_e_tot,
            'n_collisions': n_collisions,
            'boundary_hits': boundary_violations,
        } for u in self.uavs]

        if self.log_episode:
            self._log_step(team_reward, n_collisions)
        return obs, share_obs, rewards, dones, infos

    def _count_collisions(self):
        c = 0
        for i in range(self.num_uavs):
            for j in range(i+1, self.num_uavs):
                if np.linalg.norm(self.uavs[i].pos - self.uavs[j].pos) < self.d_min:
                    c += 1
        return c

    def _update_local_tracking_and_energy(self):
        for t in self.targets:
            t.measurement_info = np.zeros((4, 4), dtype=np.float32)

        # BUG FIX: stale 키 누적 방지 — 매 step UAV별 detection 상태 새로 시작
        for uav in self.uavs:
            uav.is_detected_per_target = {}

        for u_idx, uav in enumerate(self.uavs):
            t_idx = self.assignment[u_idx]
            target = self.targets[t_idx]

            snr_sen, e_sen = uav.get_sensing_model(target.pos, self.dt)
            uav.last_snr_sen = snr_sen
            alpha_ukt = 1 if snr_sen >= self.snr_threshold else 0
            uav.is_detected_per_target[t_idx] = alpha_ukt

            e_comm, R_c_ut = uav.get_comm_model(self.bs_pos, self.dt, alpha_ukt)
            uav.last_R_c = R_c_ut
            e_move = uav.get_mobility_energy(self.dt)
            uav.last_e_sen = e_sen
            uav.last_e_comm = e_comm
            uav.last_e_move = e_move
            e_tot = e_sen + e_comm + e_move
            uav.last_e_tot = e_tot
            uav.energy = max(uav.energy - e_tot, 0)

            if alpha_ukt == 1:
                dx = target.pos[0] - uav.pos[0]
                dy = target.pos[1] - uav.pos[1]
                r_sq = dx**2 + dy**2
                r = math.sqrt(r_sq) if r_sq > 0 else 1e-6

                H = np.array([
                    [dx/r,    dy/r,    0, 0],
                    [-dy/r_sq, dx/r_sq, 0, 0]
                ], dtype=np.float32)

                sigma_r_sq = max(self.sigma_r0_sq / max(snr_sen, 1e-6),
                                 self.sigma_r_sq_floor)
                sigma_theta_sq = max(self.sigma_theta0_sq / max(snr_sen, 1e-6),
                                     self.sigma_theta_sq_floor)
                R_t = np.diag([sigma_r_sq, sigma_theta_sq]).astype(np.float32)
                R_inv = np.linalg.inv(R_t)
                target.measurement_info += H.T @ R_inv @ H

                true_meas = np.array([math.sqrt(r_sq), math.atan2(dy, dx)], dtype=np.float32)
                noise = np.random.multivariate_normal([0, 0], R_t).astype(np.float32)
                z_ukt = true_meas + noise

                S_pred = self.F_mat @ target.S_global
                P_pred = self.F_mat @ target.P_global @ self.F_mat.T + self.Q
                S_innov = H @ P_pred @ H.T + R_t
                K = P_pred @ H.T @ np.linalg.inv(S_innov + np.eye(2)*1e-6)

                dx_pred = float(S_pred[0]) - float(uav.pos[0])
                dy_pred = float(S_pred[1]) - float(uav.pos[1])
                dist_sq = dx_pred**2 + dy_pred**2
                r_pred = math.sqrt(dist_sq) if dist_sq < 1e16 else 1e8
                pred_meas = np.array([r_pred, math.atan2(dy_pred, dx_pred)], dtype=np.float32)
                S_local = S_pred + K @ (z_ukt - pred_meas)
                P_local = (np.eye(4) - K @ H) @ P_pred
                if np.isnan(S_local).any() or np.isnan(P_local).any():
                    uav.local_estimates[t_idx] = None
                else:
                    uav.local_estimates[t_idx] = (S_local, P_local)
            else:
                uav.local_estimates[t_idx] = None

        # PCRLB 재귀 (조교님 지정 방식: Tichavsky form)
        # J_pred = (F · J⁻¹ · Fᵀ + Q)⁻¹
        for t_idx, target in enumerate(self.targets):
            try:
                J_prev_inv = np.linalg.inv(target.J_matrix)
                P_pred_fim = self.F_mat @ J_prev_inv @ self.F_mat.T + self.Q
                J_pred = np.linalg.inv(P_pred_fim)
            except np.linalg.LinAlgError:
                J_pred = target.J_matrix  # fallback

            target.J_matrix = J_pred + target.measurement_info

            try:
                PCRLB = np.linalg.inv(target.J_matrix)
                raw_F_kt = float(np.trace(self.Lambda @ PCRLB @ self.Lambda.T))
            except np.linalg.LinAlgError:
                raw_F_kt = 1000.0

            # ─────────────────────────────────────────────────────────────
            # F_kt 거리 기반 soft cap:
            #   탐지 실패 시 F_kt가 폭주하는데, 이때도 "가까울수록 F_kt가 낮다"는
            #   gradient를 정책에 제공하기 위해 UAV-타겟 거리에 비례한 soft cap 적용.
            #   cap = base_cap * (1 + dist_factor) 형태로,
            #   가장 가까운 UAV가 탐지 범위 안이면 cap 낮음 → 보상 gradient 유지.
            #   탐지 중(raw_F_kt 작음)엔 cap이 의미 없어서 기존과 동일하게 작동.
            # ─────────────────────────────────────────────────────────────
            min_dist = min(
                np.linalg.norm(uav.pos - target.S_global[:2])
                for uav in self.uavs
                if self.assignment.get(uav.id, -1) == t_idx
            ) if any(self.assignment.get(u.id, -1) == t_idx for u in self.uavs) \
              else min(np.linalg.norm(uav.pos - target.S_global[:2]) for uav in self.uavs)

            # 탐지 범위 ~70m 기준: 0m=cap×1.0, 70m=cap×1.0, 150m=cap×3.0
            dist_factor = min(min_dist / 70.0, 4.0)
            dynamic_cap = 50.0 * (1.0 + dist_factor)  # 50~250 범위
            target.F_kt = float(min(raw_F_kt, dynamic_cap))

    def _bs_fusion(self):
        for t_idx, target in enumerate(self.targets):
            contributors = []
            for u_idx, uav in enumerate(self.uavs):
                est = uav.local_estimates.get(t_idx)
                if est is not None:
                    contributors.append(est)
            if len(contributors) == 0:
                target.S_global = self.F_mat @ target.S_global
                target.P_global = self.F_mat @ target.P_global @ self.F_mat.T + self.Q
                continue

            omega = 1.0 / len(contributors)
            P_inv_sum = np.zeros((4, 4), dtype=np.float64)
            S_weighted = np.zeros(4, dtype=np.float64)
            for S_ut, P_ut in contributors:
                try:
                    P_ut64 = np.clip(P_ut.astype(np.float64), -1e6, 1e6)
                    P_ut_inv = np.linalg.inv(P_ut64 + np.eye(4) * 1e-4)
                except np.linalg.LinAlgError:
                    continue
                P_inv_sum += omega * P_ut_inv
                S_weighted += omega * (P_ut_inv @ S_ut.astype(np.float64))

            try:
                P_tilde = np.linalg.inv(P_inv_sum + np.eye(4) * 1e-4)
                S_tilde = (P_tilde @ S_weighted).astype(np.float32)
                P_tilde = P_tilde.astype(np.float32)
            except np.linalg.LinAlgError:
                continue

            S_pred_global = self.F_mat @ target.S_global
            P_pred_global = self.F_mat @ target.P_global @ self.F_mat.T + self.Q
            try:
                P_pred64 = P_pred_global.astype(np.float64)
                P_tilde64 = P_tilde.astype(np.float64)
                K_global = P_pred64 @ np.linalg.inv(P_pred64 + P_tilde64 + np.eye(4) * 1e-4)
                S_new = S_pred_global + (K_global @ (S_tilde - S_pred_global).astype(np.float64)).astype(np.float32)
                P_new = ((np.eye(4) - K_global) @ P_pred64).astype(np.float32)
                if np.isnan(S_new).any() or np.isinf(S_new).any():
                    target.S_global = S_pred_global
                    target.P_global = P_pred_global
                else:
                    target.S_global = S_new
                    target.P_global = np.clip(P_new, -1e6, 1e6)
            except np.linalg.LinAlgError:
                target.S_global = S_pred_global
                target.P_global = P_pred_global

    def _calculate_team_reward(self, n_collisions, boundary_violations):
        total_energy = sum(u.last_e_tot for u in self.uavs)
        r_energy = -total_energy / 100.0

        r_tracking = 0.0
        r_untracked = 0.0
        for t_idx, target in enumerate(self.targets):
            prod_loss = 1
            n_detected = 0
            for uav in self.uavs:
                a = uav.is_detected_per_target.get(t_idx, 0)
                prod_loss *= (1 - a)
                n_detected += a
            # ─────────────────────────────────────────────────────────────
            # 식 (28i): F_kt ≤ epsilon_kt 제약을 lam1 항에 흡수 (길 1)
            # F_kt 스케일링: tanh(F_kt / F_scale) × F_scale 로 급증 완화
            #   - 탐지 성공(F_kt 작음): 거의 그대로 (tanh(x)≈x for small x)
            #   - 탐지 실패(F_kt 폭주): tanh로 saturate → 보상 폭주 방지
            #   - 방향 gradient는 유지됨 (tanh는 단조증가)
            # ─────────────────────────────────────────────────────────────
            F_scale = 50.0  # dynamic_cap의 base와 동일하게
            F_scaled = F_scale * math.tanh(target.F_kt / F_scale)
            violation = max(0.0, F_scaled - target.epsilon_kt)
            r_tracking -= target.W_kt * (
                self.lam1 * (F_scaled + violation) / 100.0
                + self.lam2 * prod_loss
            )
            if n_detected == 0:
                r_untracked -= self.lam5

        r_collision = -self.lam3 * (n_collisions + boundary_violations)

        r_comm = 0.0
        for u in self.uavs:
            for t_idx, alpha in u.is_detected_per_target.items():
                if alpha == 1 and u.last_R_c < self.R_c_threshold:
                    r_comm -= self.lam4 * (self.R_c_threshold - u.last_R_c) / self.R_c_threshold

        # 항법 shaping: 전 UAV가 가장 가까운 타겟 기준으로 각자 보상 받음
        # → 할당과 무관하게 모든 UAV에 타겟 접근 gradient 제공
        # → 타겟별 가장 가까운 UAV만 반영 → 한 타겟에 쏠림 방지
        r_approach = 0.0
        for t_idx, target in enumerate(self.targets):
            min_dist = min(np.linalg.norm(uav.pos - target.pos) for uav in self.uavs)
            r_approach += max(0.0, 1.0 - min_dist / 200.0) * self.lam6

        return float(r_energy + r_tracking + r_collision + r_comm + r_untracked + r_approach)

    def _log_step(self, team_reward, n_collisions):
        log = self._episode_log
        log['uav_pos'].append(np.array([u.pos.copy() for u in self.uavs]))
        log['target_pos'].append(np.array([t.pos.copy() for t in self.targets]))
        log['target_pos_est'].append(np.array([t.S_global[:2].copy() for t in self.targets]))
        log['energy_per_uav'].append(np.array([u.last_e_tot for u in self.uavs]))
        log['energy_total'].append(sum(u.last_e_tot for u in self.uavs))
        log['F_kt'].append(np.array([t.F_kt for t in self.targets]))
        alpha_mat = np.zeros((self.num_uavs, self.num_targets))
        for u_idx, u in enumerate(self.uavs):
            for k_idx in range(self.num_targets):
                alpha_mat[u_idx, k_idx] = u.is_detected_per_target.get(k_idx, 0)
        log['alpha'].append(alpha_mat)
        log['p_ut'].append(np.array([u.p_ut for u in self.uavs]))
        log['W_kt'].append(np.array([t.W_kt for t in self.targets]))
        log['eps_kt'].append(np.array([t.epsilon_kt for t in self.targets]))
        log['d_Z_kt'].append(np.array([t.d_Z_kt for t in self.targets]))
        log['team_reward'].append(team_reward)
        log['collisions'].append(n_collisions)
        log['assignment'].append(np.array([self.assignment[u] for u in range(self.num_uavs)]))

    def get_episode_log(self):
        if self._episode_log is None:
            return None
        return {k: np.asarray(v) for k, v in self._episode_log.items()}