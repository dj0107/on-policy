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
        v_max_target = 8.0
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
                 num_critical_zones=2):
        super(UAVTrackingEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.dt = dt
        self.max_steps = 300

        self.use_aai = use_aai
        self.aai_callback = aai_callback
        self.randomize_aai = randomize_aai
        self.log_episode = log_episode

        # 위험구역
        all_zones = [
            np.array([200.0, 200.0], dtype=np.float32),
            np.array([-150.0, 300.0], dtype=np.float32),
            np.array([300.0, -250.0], dtype=np.float32),
            np.array([-250.0, -200.0], dtype=np.float32),
        ]
        self.critical_zones = all_zones[:num_critical_zones]

        # 20 dB → 선형
        self.snr_threshold_dB = 20.0
        self.snr_threshold = 10 ** (self.snr_threshold_dB / 10.0)
        self.R_c_threshold = 1.218e6
        self.d_min = 5.0
        self.map_min, self.map_max = -500.0, 500.0

        # 보상 가중치 (논문 충실 — value tuning만)
        # lam2 5→10: 트래킹 신호 약간 강화
        # lam3 10→1: 충돌 페널티가 다른 신호를 압도하던 문제 완화
        # 나머지는 paper 기본값
        self.lam1 = 1.0
        self.lam2 = 10.0
        self.lam3 = 0.5    # 1.0→0.5: 충돌 패널티 완화 → 더 적극적 항법 허용
        self.lam4 = 1.0
        self.lam5 = 5.0    # 15→5: lam7 detection reward 도입으로 binary penalty 완화
        self.lam6 = 5.0    # 2.0→5.0: 접근 보상 강화 (전 UAV 대상)
        self.lam7 = 20.0   # 신규: 탐지 성공 시 직접 양수 보상

        self.sigma_w_sq = sigma_w_sq
        self.sigma_r0_sq = 10.0
        self.sigma_theta0_sq = 1e-4
        self.sigma_r_sq_floor = 1e-3
        self.sigma_theta_sq_floor = 1e-7
        self._precompute_ekf_matrices()

        self.uavs = [UAV(i, [0, 0], altitude=15.0) for i in range(self.num_uavs)]
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

        # 1) 타겟 먼저 생성 (랜덤 위치/속도)
        self.targets = [
            Target(i,
                   [np.random.uniform(self.map_min*0.6, self.map_max*0.6),
                    np.random.uniform(self.map_min*0.6, self.map_max*0.6)],
                   [np.random.uniform(-2, 2), np.random.uniform(-2, 2)])
            for i in range(self.num_targets)
        ]

        # 2) UAV를 타겟 근처에 배치 → 초기 탐지 보장 (한 번 놓치면 EKF 복원 불가)
        #    각 타겟에 round-robin 할당: 첫 UAV는 바로 위 (0.01m 오프셋, angle singularity 회피)
        #    나머지는 반경 8m 원형 분포 (d_min=5m 보다 큼)
        uavs_per_target = {t: [] for t in range(self.num_targets)}
        for u in range(self.num_uavs):
            uavs_per_target[u % self.num_targets].append(u)

        uav_positions = [None] * self.num_uavs
        for t_idx, target in enumerate(self.targets):
            group = uavs_per_target[t_idx]
            n_around = max(1, len(group) - 1)
            for idx_in_group, u_idx in enumerate(group):
                if idx_in_group == 0:
                    offset = np.array([0.01, 0.0], dtype=np.float32)
                else:
                    angle = 2.0 * math.pi * (idx_in_group - 1) / n_around
                    radius = 8.0
                    offset = np.array([radius * math.cos(angle),
                                       radius * math.sin(angle)], dtype=np.float32)
                uav_positions[u_idx] = (target.pos + offset).astype(np.float32)

        self.uavs = [
            UAV(i, uav_positions[i].tolist(), altitude=15.0)
            for i in range(self.num_uavs)
        ]

        for target in self.targets:
            target.d_Z_kt = min(np.linalg.norm(target.pos - cz) for cz in self.critical_zones)

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
            for t in self.targets:
                t.W_kt = float(np.random.uniform(0.5, 3.5))
                t.epsilon_kt = float(np.random.uniform(1.0, 10.0))
            for u in self.uavs:
                u.p_ut = float(np.random.uniform(0.3, 3.0))

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
                target.W_kt = 3.0
                target.epsilon_kt = 2.0
            elif target.d_Z_kt < 200.0:
                target.W_kt = 2.0
                target.epsilon_kt = 5.0
            else:
                target.W_kt = 1.0
                target.epsilon_kt = 5.0
        self._build_assignment()
        # p_ut: 1W 기준으로 비례 축소 (이전 10/15/25 → 0.5/1.0/2.0)
        for u_idx, uav in enumerate(self.uavs):
            target = self.targets[self.assignment[u_idx]]
            est_pos = target.S_global[:2]
            dist = np.linalg.norm(uav.pos - est_pos)
            if dist > 200.0:
                uav.p_ut = 2.0
            elif dist < 50.0:
                uav.p_ut = 0.5
            else:
                uav.p_ut = 1.0

    def _get_obs(self):
        POS_SCALE = 500.0
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
                # 직전 step 탐지 여부 — 에이전트가 현재 위치로 탐지 가능 여부 직접 피드백
                obs.append(float(uav.is_detected_per_target.get(k_idx, 0)))
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

        # 표준 PCRLB 재귀 (Tichavsky 1998 / Van Trees)
        # 논문 식 (26)은 typo로 추정 — F_kt가 무한정 0에 수렴하는 문제 해결
        # J_{k|k-1} = [F J_{k-1}⁻¹ Fᵀ + Q]⁻¹
        # J_k       = J_{k|k-1} + Σ α HᵀR⁻¹H
        for target in self.targets:
            try:
                P_post_prev = np.linalg.inv(target.J_matrix)
                P_prior = self.F_mat @ P_post_prev @ self.F_mat.T + self.Q
                prior_J = np.linalg.inv(P_prior + np.eye(4, dtype=np.float32) * 1e-6)
                target.J_matrix = (prior_J + target.measurement_info).astype(np.float32)
                PCRLB = np.linalg.inv(target.J_matrix + np.eye(4, dtype=np.float32) * 1e-6)
                target.F_kt = float(min(np.trace(self.Lambda @ PCRLB @ self.Lambda.T), 1000.0))
            except np.linalg.LinAlgError:
                # 수치적으로 불안정한 경우 reset
                target.J_matrix = np.eye(4, dtype=np.float32) * 0.1
                target.F_kt = 1000.0

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
        r_detect = 0.0
        for t_idx, target in enumerate(self.targets):
            prod_loss = 1
            n_detected = 0
            for uav in self.uavs:
                a = uav.is_detected_per_target.get(t_idx, 0)
                prod_loss *= (1 - a)
                n_detected += a
            r_tracking -= target.W_kt * (
                self.lam1 * (target.F_kt / 100.0) + self.lam2 * prod_loss
            )
            if n_detected == 0:
                r_untracked -= self.lam5
            else:
                # 탐지 성공 시 직접 양수 보상 → 탐지 행동 명시적 강화
                r_detect += self.lam7

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

        return float(r_energy + r_tracking + r_collision + r_comm + r_untracked + r_detect + r_approach)

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
