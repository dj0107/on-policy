import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

# ============================================================================
# (12) 버전 기준 - Dec-POMDP, AAI 출력: epsilon_kt, W_kt, p_ut
# (8) → (12) 주요 변경:
#   - MDP → Dec-POMDP (global state s_t + local obs o_u,t 분리)
#   - AAI 출력: M_t (할당) → eps_kt(threshold), W_kt(weight), p_ut(power)
#   - 보상: UAV별 개별 r_u(t) → global team reward r_t (식 32)
#   - 측정모델: 절대좌표 → 상대좌표 (x_k - x_u, y_k - y_u)
#   - 운영절차 8단계 → 7단계 (AAI가 직접 파라미터 결정)
# ============================================================================


class Target:
    """타겟 상태, EKF용 fused estimate, BFIM 관리"""
    def __init__(self, target_id, initial_pos, initial_velocity):
        self.id = target_id
        self.pos = np.array(initial_pos, dtype=np.float32)        # 실제 위치 (시뮬용 ground truth)
        self.velocity = np.array(initial_velocity, dtype=np.float32)
        
        # BS에서 융합된 글로벌 추정 상태 (식 23, 24)
        self.S_global = np.concatenate([self.pos, self.velocity])  # \tilde{S}_k,t (4,)
        self.P_global = np.eye(4, dtype=np.float32) * 1.0          # \tilde{P}_k,t (4,4)
        
        # BFIM (식 26) 및 PCRLB 추적 정확도 (식 27)
        self.J_matrix = np.eye(4, dtype=np.float32) * 0.1
        self.F_kt = 0.0  # tr(Lambda * J^-1 * Lambda^T)
        self.measurement_info = np.zeros((4, 4), dtype=np.float32)
        
        # AAI가 결정 (식 23)
        self.epsilon_kt = 5.0   # tracking accuracy threshold
        self.W_kt = 1.0         # priority weight
        
        # 위험구역까지 최소 거리
        self.d_Z_kt = float('inf')

    def step(self, dt, Q=None):
        """식 (1): S_{k(t+1)} = F * S_kt + w_kt, w_kt ~ N(0, Q)"""
        # 결정론적 등속 이동
        self.pos += self.velocity * dt
        # 식 (1)의 process noise w_kt 적용
        if Q is not None:
            w_kt = np.random.multivariate_normal(np.zeros(4), Q).astype(np.float32)
            self.pos += w_kt[:2]
            self.velocity += w_kt[2:]


class UAV:
    """UAV 행동 제어, 에너지 관리. AAI가 송신전력 p_ut를 외부에서 주입"""
    def __init__(self, uav_id, initial_pos, altitude, max_speed=10.0, max_energy=100000.0):
        self.id = uav_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.H = altitude
        self.v_max = max_speed   # 논문 Table: v_max = 10 m/s
        self.energy = max_energy
        
        # AAI 동적 할당 변수 (식 23)
        self.p_ut = 15.0        # 송신 전력 (W) - AAI가 결정
        
        # 시간 슬롯 분할: tau_s + tau_c = delta
        # (12)에서는 mu가 AAI 출력에서 빠짐 → 일단 0.5로 고정
        self.tau_s_ratio = 0.5
        
        self.is_detected_per_target = {}  # {target_id: alpha_ukt}
        self.last_e_tot = 0.0
        self.last_R_c = 0.0   # 식 (28g) 통신률 제약 검사용
        
        # 로컬 EKF용 사후 추정 (BS로 전송)
        self.local_estimates = {}  # {target_id: (S_t|t, P_t|t)}
        
        # --- 물리 상수 (논문 Table) ---
        self.f_c = 2.4e9
        self.lam = 3e8 / self.f_c
        self.G_t = 100.0       # 20 dBi
        self.G_r = 1000.0      # 30 dBi
        self.sigma = 1.0       # RCS
        self.N0 = 1e-14        # -110 dBmW
        self.c1, self.c2 = 11.9, 0.13          # urban
        self.eta_LoS, self.eta_NLoS = 1.0, 100.0
        self.B_b = 1e6
        
        # 추진 에너지 상수 (논문 Table)
        self.P0, self.P1 = 3.4, 20.0
        self.u_tip, self.v0 = 60.0, 5.4
        self.d0, self.rho, self.s0, self.A = 0.3, 1.225, 0.03, 0.28

    def apply_action(self, action, dt):
        """식 (28d): ||q_t - q_{t-1}|| <= v_max * delta"""
        delta_q = np.array(action, dtype=np.float32) * self.v_max * dt
        delta_norm = np.linalg.norm(delta_q)
        if delta_norm > self.v_max * dt:
            delta_q = (delta_q / delta_norm) * self.v_max * dt
        self.velocity = delta_q / dt if dt > 0 else np.zeros(2)
        self.pos += delta_q

    def get_sensing_model(self, target_pos, dt):
        """식 (6): 레이더 SNR 및 (7): 센싱 에너지"""
        d_uk = math.sqrt(np.sum((self.pos - target_pos)**2) + self.H**2)
        tau_s = self.tau_s_ratio * dt
        snr_sen = (self.p_ut * self.G_t * self.G_r * (self.lam**2) * self.sigma * tau_s) \
                  / (self.N0 * (4 * math.pi)**3 * (d_uk**4))
        e_sen = self.p_ut * tau_s
        return snr_sen, e_sen

    def get_comm_model(self, bs_pos, dt, alpha_ukt):
        """식 (15)~(18): 통신 모델 및 에너지
        주: 식 (16) 분모는 d_u(t)로 표기되나, free-space path loss는 송수신단
            간 실제 거리(3D)를 쓰는 것이 물리적으로 정확. d_2D=0일 때
            (UAV가 BS 바로 위) 채널이득이 발산하는 것을 방지하기 위해 3D 사용.
        """
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
        
        e_comm = self.p_ut * tau_c
        return e_comm, R_c_ut

    def get_mobility_energy(self, dt):
        """식 (19): 이동 에너지"""
        v_norm = np.linalg.norm(self.velocity)
        if v_norm == 0:
            return (self.P0 + self.P1) * dt
        term1 = self.P0 * (1 + (3 * v_norm**2) / (self.u_tip**2))
        inner_sqrt = math.sqrt(1 + (v_norm**4) / (4 * self.v0**4))
        term2 = self.P1 * math.sqrt(max(inner_sqrt - (v_norm**2) / (2 * self.v0**2), 0))
        term3 = 0.5 * self.d0 * self.rho * self.s0 * self.A * (v_norm**3)
        return (term1 + term2 + term3) * dt


class UAVTrackingEnv(gym.Env):
    """
    Dec-POMDP 기반 UAV swarm MTT 환경 (논문 (12) 기준)
    - global state s_t (식 29): centralized critic 입력
    - local obs o_u,t (식 30): decentralized actor 입력
    - global team reward r_t (식 32)
    """
    
    def __init__(self, num_uavs=5, num_targets=2, dt=1.0):
        super(UAVTrackingEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.dt = dt          # 논문 Table: delta = 1s
        self.max_steps = 100
        
        # 위험구역 (Z)
        self.critical_zones = [np.array([200.0, 200.0]), np.array([-150.0, 300.0])]
        
        # 임계값 (논문 Table)
        self.snr_threshold = 20.0          # gamma_sen_th = 13 dB
        self.R_c_threshold = 1.218e6       # R_c_th
        self.d_min = 5.0                   # 식 (28e)
        self.map_min, self.map_max = -500.0, 500.0  # 식 (28b)(28c)
        
        # 보상 가중치 (식 32)
        self.lam1 = 1.0
        self.lam2 = 50.0
        self.lam3 = 100.0
        self.lam4 = 10.0      # 식 (28g) 통신률 제약 위반 페널티
        
        # EKF 사전계산
        self.sigma_w_sq = 5.0
        self.sigma_r0_sq = 10.0
        self.sigma_theta0_sq = 1e-4
        self._precompute_ekf_matrices()
        
        # 더미 초기화 (obs_dim 계산용)
        self.uavs = [UAV(i, [0, 0], altitude=100.0) for i in range(self.num_uavs)]
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
        # share_obs: 모든 에이전트가 동일 global state 공유 (CTDE)
        self.share_observation_space = [
            spaces.Box(low=-np.inf, high=np.inf, shape=(global_dim,), dtype=np.float32)
            for _ in range(self.num_uavs)
        ]
        
        self.reset()

    def _precompute_ekf_matrices(self):
        dt = self.dt
        # 식 (2)
        self.F_mat = np.array([
            [1, 0, dt, 0],
            [0, 1, 0,  dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1]
        ], dtype=np.float32)
        self.F_inv = np.linalg.inv(self.F_mat)
        
        # 식 (3)
        q11, q13, q33 = dt**4/4, dt**3/2, dt**2
        self.Q = np.array([
            [q11, 0,   q13, 0],
            [0,   q11, 0,   q13],
            [q13, 0,   q33, 0],
            [0,   q13, 0,   q33]
        ], dtype=np.float32) * self.sigma_w_sq
        self.Q += np.eye(4) * 1e-6
        self.Q_inv = np.linalg.inv(self.Q)
        
        # 식 (27)
        self.Lambda = np.diag([1.0, 1.0, dt, dt]).astype(np.float32)

    def seed(self, seed=None):
        np.random.seed(seed if seed is not None else 1)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        self.time_slot = 0
        self.bs_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        self.uavs = [
            UAV(i, [np.random.uniform(-50, 50), np.random.uniform(-50, 50)], altitude=100.0)
            for i in range(self.num_uavs)
        ]
        # 명시적 초기화 (이전 에피소드 잔재 방지)
        for uav in self.uavs:
            uav.is_detected_per_target = {}
            uav.local_estimates = {}
        self.targets = [
            Target(i,
                   [np.random.uniform(self.map_min, self.map_max),
                    np.random.uniform(self.map_min, self.map_max)],
                   [np.random.uniform(1, 3), np.random.uniform(1, 3)])
            for i in range(self.num_targets)
        ]
        
        for target in self.targets:
            target.d_Z_kt = min(np.linalg.norm(target.pos - cz) for cz in self.critical_zones)
        
        self._run_aai_heuristic()
        
        local_obs, global_state = self._get_obs()
        share_obs = np.array([global_state] * self.num_uavs, dtype=np.float32)
        return np.array(local_obs, dtype=np.float32), share_obs

    # ========================================================================
    # AAI Framework (논문 IV-A)
    # 출력: epsilon_kt, W_kt, p_ut
    # ========================================================================
    def _run_aai_heuristic(self):
        """
        AAI Heuristic (LLM 도입 전 임시 구현)
        - 식 (23): [{eps_kt}, {W_kt}, {p_ut}] = f_AAI(Phi_t)
        - Phi_t (식 24): {tilde_S, tilde_P, d_Z_kt}_k, {E_ut}_u
        """
        # ----- Perception: 글로벌 텔레메트리 -----
        for target in self.targets:
            target.d_Z_kt = min(
                np.linalg.norm(target.pos - cz) for cz in self.critical_zones
            )
        
        # ----- Reasoning: 파라미터 결정 -----
        for target in self.targets:
            # W_kt: 위험구역 접근 시 가중치 ↑
            if target.d_Z_kt < 100.0:
                target.W_kt = 3.0
            elif target.d_Z_kt < 200.0:
                target.W_kt = 2.0
            else:
                target.W_kt = 1.0
            
            # eps_kt: 위험구역 접근 시 임계값 ↓ (정확도 강화)
            if target.d_Z_kt < 100.0:
                target.epsilon_kt = 2.0
            else:
                target.epsilon_kt = 5.0
        
        # 보조 변수: 1:1 mapping (운영상 필요)
        unassigned = list(range(self.num_uavs))
        self.assignment = {}
        for t_idx, target in enumerate(self.targets):
            if not unassigned:
                break
            best_uav = min(unassigned,
                           key=lambda u: np.linalg.norm(self.uavs[u].pos - target.pos))
            self.assignment[best_uav] = t_idx
            unassigned.remove(best_uav)
        for u_idx in unassigned:
            best_t = min(range(self.num_targets),
                         key=lambda t: np.linalg.norm(self.uavs[u_idx].pos - self.targets[t].pos))
            self.assignment[u_idx] = best_t
        
        # p_ut: 거리/잔여에너지 기반 송신전력 조절
        for u_idx, uav in enumerate(self.uavs):
            target = self.targets[self.assignment[u_idx]]
            dist = np.linalg.norm(uav.pos - target.pos)
            if dist > 200.0:
                uav.p_ut = 25.0
            elif dist < 50.0:
                uav.p_ut = 10.0
            else:
                uav.p_ut = 15.0

    # ========================================================================
    # Observation: Dec-POMDP (식 29 / 식 30)
    # ========================================================================
    def _get_obs(self):
        """
        식 (29) global state: {q_u, E_u}_u ∪ {tilde_S_k, F_kt, d_Z_kt, eps_kt, W_kt}_k
        식 (30) local obs: (q_u, E_u, delta_q_uk, tilde_S_k, eps_kt, W_kt)
        """
        # ---- Global state ----
        global_parts = []
        for u in self.uavs:
            global_parts.extend(u.pos)
            global_parts.append(u.energy)
        for t in self.targets:
            global_parts.extend(t.S_global)
            global_parts.append(t.F_kt)
            global_parts.append(t.d_Z_kt)
            global_parts.append(t.epsilon_kt)
            global_parts.append(t.W_kt)
        global_state = np.array(global_parts, dtype=np.float32)
        
        # ---- Local obs ----
        local_obs_list = []
        for u_idx, uav in enumerate(self.uavs):
            t_idx = self.assignment.get(u_idx, 0)
            target = self.targets[t_idx]
            
            delta_q = target.S_global[:2] - uav.pos
            
            obs = (
                list(uav.pos)                # q_ut (2)
                + [uav.energy]               # E_ut (1)
                + list(delta_q)              # delta_q_uk (2)
                + list(target.S_global)      # tilde_S_kt (4)
                + [target.epsilon_kt]        # eps_kt (1)
                + [target.W_kt]              # W_kt (1)
            )
            local_obs_list.append(np.array(obs, dtype=np.float32))
        
        return local_obs_list, global_state

    # ========================================================================
    # Step (논문 II-A 7단계 운영 절차)
    # ========================================================================
    def step(self, actions):
        self.time_slot += 1
        
        # 1) 타겟 이동 (식 1) - process noise 포함
        for target in self.targets:
            target.step(self.dt, Q=self.Q)
        
        # 2) UAV 이동 (식 28d)
        for i, uav in enumerate(self.uavs):
            uav.apply_action(actions[i], self.dt)
        
        # 3) 로컬 EKF + 에너지
        self._update_local_tracking_and_energy()
        
        # 4) BS 융합 (식 21~24)
        self._bs_fusion()
        
        # 5) AAI 재호출
        self._run_aai_heuristic()
        
        # 6) Global team reward (식 32)
        team_reward = self._calculate_team_reward()
        rewards = np.array([[team_reward]] * self.num_uavs, dtype=np.float32)
        
        local_obs, global_state = self._get_obs()
        obs = np.array(local_obs, dtype=np.float32)
        share_obs = np.array([global_state] * self.num_uavs, dtype=np.float32)
        dones = np.array([self.time_slot >= self.max_steps] * self.num_uavs, dtype=bool)
        infos = [{'uav_energy': u.energy, 'team_reward': team_reward} for u in self.uavs]
        
        return obs, share_obs, rewards, dones, infos

    # ========================================================================
    # Local EKF + Energy
    # ========================================================================
    def _update_local_tracking_and_energy(self):
        for t in self.targets:
            t.measurement_info = np.zeros((4, 4), dtype=np.float32)
        
        for u_idx, uav in enumerate(self.uavs):
            t_idx = self.assignment[u_idx]
            target = self.targets[t_idx]
            
            snr_sen, e_sen = uav.get_sensing_model(target.pos, self.dt)
            alpha_ukt = 1 if snr_sen >= self.snr_threshold else 0
            uav.is_detected_per_target[t_idx] = alpha_ukt
            
            e_comm, R_c_ut = uav.get_comm_model(self.bs_pos, self.dt, alpha_ukt)
            uav.last_R_c = R_c_ut    # 식 (28g) 통신률 제약 검사용
            e_move = uav.get_mobility_energy(self.dt)
            
            e_tot = e_sen + e_comm + e_move
            uav.last_e_tot = e_tot
            uav.energy = max(uav.energy - e_tot, 0)
            
            # 식 (11): Jacobian H_ukt - 상대좌표
            if alpha_ukt == 1:
                dx = target.pos[0] - uav.pos[0]
                dy = target.pos[1] - uav.pos[1]
                r_sq = dx**2 + dy**2
                r = math.sqrt(r_sq) if r_sq > 0 else 1e-6
                
                H = np.array([
                    [dx/r,    dy/r,    0, 0],
                    [-dy/r_sq, dx/r_sq, 0, 0]
                ], dtype=np.float32)
                
                # 식 (5): R_t
                sigma_r_sq = self.sigma_r0_sq / max(snr_sen, 1e-6)
                sigma_theta_sq = self.sigma_theta0_sq / max(snr_sen, 1e-6)
                R_t = np.diag([sigma_r_sq, sigma_theta_sq]).astype(np.float32)
                R_inv = np.linalg.inv(R_t)
                
                # 식 (26): alpha_ukt * H^T R^-1 H
                target.measurement_info += H.T @ R_inv @ H
                
                # 로컬 EKF 사후 추정 (식 8~14)
                true_meas = np.array([
                    math.sqrt(r_sq),
                    math.atan2(dy, dx)
                ], dtype=np.float32)
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
                # NaN 가드
                if np.isnan(S_local).any() or np.isnan(P_local).any():
                    uav.local_estimates[t_idx] = None
                else:
                    uav.local_estimates[t_idx] = (S_local, P_local)
            else:
                uav.local_estimates[t_idx] = None
        
        # 각 타겟의 BFIM 누적 (식 26)
        for target in self.targets:
            prior_J = self.F_inv.T @ target.J_matrix @ self.F_inv + self.Q_inv
            target.J_matrix = prior_J + target.measurement_info
            try:
                PCRLB = np.linalg.inv(target.J_matrix)
                target.F_kt = float(np.trace(self.Lambda @ PCRLB @ self.Lambda.T))
            except np.linalg.LinAlgError:
                target.F_kt = 1000.0

    # ========================================================================
    # BS Fusion (식 21~24)
    # ========================================================================
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
            
            # 식 (22): P_ut 크기 폭발 방지 (float64로 업캐스트)
            P_inv_sum = np.zeros((4, 4), dtype=np.float64)
            S_weighted = np.zeros(4, dtype=np.float64)
            for S_ut, P_ut in contributors:
                try:
                    P_ut64 = P_ut.astype(np.float64)
                    # 공분산이 너무 크면 클리핑 (수치 안정성)
                    P_ut64 = np.clip(P_ut64, -1e6, 1e6)
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
            
            # 식 (23)(24)
            S_pred_global = self.F_mat @ target.S_global
            P_pred_global = self.F_mat @ target.P_global @ self.F_mat.T + self.Q
            try:
                P_pred64 = P_pred_global.astype(np.float64)
                P_tilde64 = P_tilde.astype(np.float64)
                K_global = P_pred64 @ np.linalg.inv(P_pred64 + P_tilde64 + np.eye(4) * 1e-4)
                S_new = S_pred_global + (K_global @ (S_tilde - S_pred_global).astype(np.float64)).astype(np.float32)
                P_new = ((np.eye(4) - K_global) @ P_pred64).astype(np.float32)
                # NaN/Inf 최종 가드
                if np.isnan(S_new).any() or np.isinf(S_new).any():
                    target.S_global = S_pred_global
                    target.P_global = P_pred_global
                else:
                    target.S_global = S_new
                    target.P_global = np.clip(P_new, -1e6, 1e6)
            except np.linalg.LinAlgError:
                target.S_global = S_pred_global
                target.P_global = P_pred_global

    # ========================================================================
    # Global Team Reward (식 32)
    # r_t = -sum_u E_tot - sum_k W_kt(lam1*F_kt + lam2*prod(1-alpha)) - lam3*C_t
    # ========================================================================
    def _calculate_team_reward(self):
        total_energy = sum(u.last_e_tot for u in self.uavs)
        r_energy = -total_energy
        
        # 식 (32) tracking term: 모든 UAV (할당 무관)에 대한 곱
        # 논문의 prod_{u in U} (1 - alpha_ukt)는 "어떤 UAV도 감지 못함"을 표현
        r_tracking = 0.0
        for t_idx, target in enumerate(self.targets):
            prod_loss = 1
            for uav in self.uavs:
                prod_loss *= (1 - uav.is_detected_per_target.get(t_idx, 0))
            r_tracking -= target.W_kt * (
                self.lam1 * target.F_kt + self.lam2 * prod_loss
            )
        
        # 식 (28e) UAV 간 충돌
        C_t = 0
        for i in range(self.num_uavs):
            for j in range(i+1, self.num_uavs):
                if np.linalg.norm(self.uavs[i].pos - self.uavs[j].pos) < self.d_min:
                    C_t += 1
        # 맵 경계 위반
        for u in self.uavs:
            if not (self.map_min <= u.pos[0] <= self.map_max and
                    self.map_min <= u.pos[1] <= self.map_max):
                C_t += 1
        r_collision = -self.lam3 * C_t
        
        # 식 (28g) 통신률 제약 위반 페널티
        r_comm = 0.0
        for u in self.uavs:
            # alpha=1인 UAV에 한해 R_c >= R_c_th 요구
            for t_idx, alpha in u.is_detected_per_target.items():
                if alpha == 1 and u.last_R_c < self.R_c_threshold:
                    r_comm -= self.lam4 * (self.R_c_threshold - u.last_R_c) / self.R_c_threshold
        
        return float(r_energy + r_tracking + r_collision + r_comm)
