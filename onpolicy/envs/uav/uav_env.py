import numpy as np
import math
import gymnasium as gym
from gymnasium import spaces

class Target:
    """이동하는 타겟의 상태와 EKF 추적을 위한 정보 행렬(J) 관리 (EKF는 아직 간소화함)"""
    def __init__(self, target_id, initial_pos, initial_velocity):
        self.id = target_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array(initial_velocity, dtype=np.float32)
        self.est_state = np.concatenate([self.pos, self.velocity])
        self.J_matrix = np.eye(4, dtype=np.float32) * 0.1 # BFIM 계산용
        self.F_ukt = 0.0  
        self.measurement_info = np.zeros((4, 4), dtype=np.float32)
    
    def step(self, dt):
        self.pos += self.velocity * dt

class UAV:
    """행동 제어 및 에너지 소비, AAI 변수 관리"""
    def __init__(self, uav_id, initial_pos, altitude, max_speed=20.0, max_energy=100000.0):
        self.id = uav_id
        self.pos = np.array(initial_pos, dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32) 
        self.H = altitude 
        self.v_max = max_speed 
        self.energy = max_energy 
        self.is_detected = False
        
        # AAI 할당 변수들
        self.assigned_target_id = None 
        self.W_t = 1.0  # 추적 우선순위 가중치
        self.mu = 0.5   # ISAC 센싱/통신 시간 비율
        self.last_e_tot = 0.0
        
        # --- ISAC 및 물리 모델 상수 ---
        self.p = 0.1 
        self.f_c = 2.4e9
        self.lam = 3e8 / self.f_c 
        self.G_t, self.G_r, self.sigma, self.N0 = 1, 1, 1.0, 1e-10 
        self.c1, self.c2, self.eta_LoS, self.eta_NLoS = 9.6, 0.28, 1.0, 20.0 
        self.B_b = 1e6 
        self.P0, self.P1 = 79.856, 88.628 
        self.u_up, self.v0 = 120.0, 4.03
        self.d0, self.rho, self.s0, self.A = 0.6, 1.225, 0.05, 0.5

    def apply_action(self, action, dt):
        desired_velocity = np.array(action, dtype=np.float32) * self.v_max
        v_norm = np.linalg.norm(desired_velocity)
        if v_norm > self.v_max:
            desired_velocity = (desired_velocity / v_norm) * self.v_max
        self.velocity = desired_velocity
        self.pos += self.velocity * dt

    def get_sensing_model(self, target_pos, dt):
        d_uk = math.sqrt(np.sum((self.pos - target_pos)**2) + self.H**2)
        h_sen = (self.G_t * self.G_r * (self.lam**2) * self.sigma) / ((4 * math.pi)**3 * (d_uk**4)) # 각종 값들 하나로 압축
        
        # 센싱 시간(mu * dt)을 분자에 곱하여 SNR 계산
        tau_s = self.mu * dt
        snr_sen = (self.p * h_sen * tau_s) / self.N0 # G, 람다, d등은 h_sen에 포함 (6)
        
        e_sen = self.p * self.mu * dt # (7)
        return snr_sen, e_sen

    def get_comm_model(self, bs_pos, dt):
        # 1. 수평 거리(2D) 계산
        d_ut_2d = np.linalg.norm(self.pos - bs_pos) 
        
        # 2. 채널 이득 계산
        theta_ut = math.atan(self.H / d_ut_2d) if d_ut_2d > 0 else math.pi/2
        p_los = 1 / (1 + self.c1 * math.exp(-self.c2 * (math.degrees(theta_ut) - self.c1)))
        
        # eq (16)~(17). self.eta_(N)LoS 는 상수, 아직은 안쓰임
        # h_c = ((p_los * self.eta_LoS + (1-p_los) * self.eta_NLoS) * (self.lam**2)) / ((4 * math.pi * d_ut_2d)**2 + 1e-9) 
        # Rcut 아직 계산 x

        # 3. 통신 에너지 계산 (18) 
        # p는 송신 전력, (1-mu)*dt는 통신 시간(tau_c)
        # 아직은 이 위로 안쓰임
        e_comm = self.p * (1 - self.mu) * dt
        
        return e_comm

    def get_mobility_energy(self, dt):
        # F의 (19) 윗 부분들 복잡해서 3항으로 나눠서 합침 
        v_norm = np.linalg.norm(self.velocity)
        if v_norm == 0: return (self.P0 + self.P1) * dt 
        term1 = self.P0 * (1 + (3 * v_norm**2) / (self.u_up**2))
        inner_sqrt = math.sqrt(1 + (v_norm**4) / (4 * self.v0**4))
        term2 = self.P1 * math.sqrt(inner_sqrt - (v_norm**2) / (2 * self.v0**2))
        term3 = 0.5 * self.d0 * self.rho * self.s0 * self.A * (v_norm**3) 
        return (term1 + term2 + term3) * dt


class UAVTrackingEnv(gym.Env):
    def __init__(self, num_uavs=5, num_targets=2, dt=0.5):
        super(UAVTrackingEnv, self).__init__()
        self.num_uavs = num_uavs
        self.num_targets = num_targets
        self.dt = dt
        self.max_steps = 100
        
        # 위험 구역(Critical Zones) 임시로 하드코딩
        self.critical_zones = [np.array([200.0, 200.0]), np.array([-150.0, 300.0])] 
        
        self.uavs = [UAV(i, [0, 0], altitude=100.0) for i in range(self.num_uavs)]
        self.targets = [Target(i, [0, 0], [0, 0]) for i in range(self.num_targets)]
        
        # Observation 차원 계산 
        dummy_local_obs, _ = self._get_obs() # 최초에 1회 관측해보기
        obs_dim = len(dummy_local_obs[0]) # 9차원
        share_obs_dim = obs_dim * self.num_uavs # 9 x uav수 차원
        
        # 각 uav의 행동
        self.action_space = [spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32) for _ in range(self.num_uavs)]
        # 각 uav의 관측
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32) for _ in range(self.num_uavs)] # 
        # 중앙의 관측(gymnasium 특성상 에이전트 수만큼의 리스트로 줘야 해서 for문 여기도 붙음)
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_uavs)] # uav들 관측 공간들 다 붙인거

        self.reset() # 초기화

    def seed(self, seed=None):
        # 시드 생성
        np.random.seed(seed if seed is not None else 1)

    def reset(self, seed=None, options=None):
        # 환경 초기화
        if seed is not None: self.seed(seed)
        self.time_slot = 0 # 처음이라서 0
        self.bs_pos = np.array([0.0, 0.0], dtype=np.float32) #중앙 좌표로
        self.snr_threshold = 10.0 # 탐지 성공 기준
        
        map_boundary = 500.0 # -X~X라서 맵 크기는 1000x1000 
        # uav랑 타겟 무작위로 배치 (추후 조정)
        self.uavs = [UAV(i, [np.random.uniform(-50, 50), np.random.uniform(-50, 50)], altitude=100.0) for i in range(self.num_uavs)]
        self.targets = [Target(i, 
                               [np.random.uniform(-map_boundary, map_boundary), np.random.uniform(-map_boundary, map_boundary)], 
                               [np.random.uniform(1, 3), np.random.uniform(1, 3)]) 
                        for i in range(self.num_targets)]
        
        self._run_aai_heuristic() # 초기에 AAI로 타겟 할당
        local_obs, _ = self._get_obs() # 최초 관측
        return np.array(local_obs, dtype=np.float32)

    def _run_aai_heuristic(self):
        """
        AAI Framework: 타겟 할당(A_t), ISAC 비율(mu_t), 우선순위(W_t) 동적 결정
        LLM 도입을 아직 안하고 임시로 넣은 naive에 가까운 형태라 제거 후 교체 예정
        """
        unassigned_uavs = list(range(self.num_uavs))
        
        # 1. Coverage 보장을 위한 기본 1:1 매핑
        for t_idx, target in enumerate(self.targets):
            if not unassigned_uavs: break
            best_uav = min(unassigned_uavs, key=lambda u_idx: np.linalg.norm(self.uavs[u_idx].pos - target.pos))
            self.uavs[best_uav].assigned_target_id = t_idx
            unassigned_uavs.remove(best_uav)
            
        # 2. 남은 UAV 중복 할당 (협력 추적)
        for u_idx in unassigned_uavs:
            best_target = min(range(self.num_targets), key=lambda t_idx: np.linalg.norm(self.uavs[u_idx].pos - self.targets[t_idx].pos))
            self.uavs[u_idx].assigned_target_id = best_target

        # 3. W_t 및 mu_t 조절
        for uav in self.uavs:
            target = self.targets[uav.assigned_target_id]
            d_kt_Z = min([np.linalg.norm(target.pos - cz) for cz in self.critical_zones])
            
            # W_t: Critical Zone 접근 시 가중치 3배
            uav.W_t = 3.0 if d_kt_Z < 100.0 else 1.0
                
            # mu_t: 거리에 따른 ISAC 센싱/통신 비율 조절
            dist_to_target = np.linalg.norm(uav.pos - target.pos)
            if dist_to_target > 150.0:
                uav.mu = 0.7  # 멀면 센싱 집중
            elif dist_to_target < 50.0:
                uav.mu = 0.3  # 가까우면 통신 집중
            else:
                uav.mu = 0.5

    def _get_obs(self):
        local_obs_list = []
        for uav in self.uavs:
            t_idx = uav.assigned_target_id if uav.assigned_target_id is not None else 0
            target = self.targets[t_idx] # 지금 추적중인 타겟
            
            # eq (29)의 S_t|t^k: 
            # 실제 target.pos가 아니라(이건 불가능한 이상적 상태), 시스템 내부에서 관리되는 추정 상태를 반환
            # (이 추정 상태는 _update_tracking_and_energy에서 EKF로 갱신됨)
            S_t_k = target.est_state # [x_est, y_est, vx_est, vy_est]
             
            d_kt_Z = min([np.linalg.norm(target.pos - cz) for cz in self.critical_zones])
            
            # q_ut(2), E_ut(1), F_ukt(1), S_t_k(4), d_kt_Z(1) -> 9차원
            obs_i = list(uav.pos) + [uav.energy, target.F_ukt] + list(S_t_k) + [d_kt_Z]
            local_obs_list.append(np.array(obs_i, dtype=np.float32))
            
        global_state = np.array(local_obs_list).flatten()
        return local_obs_list, global_state

    def step(self, actions):
        self.time_slot += 1
        
        # 1. 물리 모델 업데이트 (타겟, 드론 이동)
        for target in self.targets: target.step(self.dt) # 1스텝 진행
        for i, uav in enumerate(self.uavs): uav.apply_action(actions[i], self.dt)
            
        # 2. AAI 재할당 및 파라미터 업데이트
        self._run_aai_heuristic()
        
        # 3. EKF 추적 업데이트 및 에너지 소모
        self._update_tracking_and_energy()

        # 4. Eq (31) 기반 보상
        rewards = self._calculate_rewards()
        
        local_obs, _ = self._get_obs()
        obs = np.array(local_obs, dtype=np.float32)
        dones = np.array([self.time_slot >= self.max_steps] * self.num_uavs, dtype=bool)
        infos = [{'uav_energy': u.energy} for u in self.uavs]
        
        return obs, rewards, dones, infos # MAPPO 형식

    def _update_tracking_and_energy(self):
        dt = self.dt
        sigma_w_sq = 0.1 # 논문 기준 noise variance (sigma_w^2)

        # 1. 전이 행렬 F 정의 (2)
        F_mat = np.array([[1, 0, dt, 0], # (델타 = dt)
                          [0, 1, 0, dt], 
                          [0, 0, 1, 0], 
                          [0, 0, 0, 1]], dtype=np.float32)
        F_inv = np.linalg.inv(F_mat)

        # 2. 공정 잡음 행렬 Q (3)
        # q11~q33 4개는 Q에 넣기 위한 변수들
        q11 = dt**4 / 4; q13 = dt**3 / 2
        q31 = dt**3 / 2; q33 = dt**2
        
        Q = np.array([
            [q11, 0,   q13, 0],
            [0,   q11, 0,   q13],
            [q31, 0,   q33, 0],
            [0,   q31, 0,   q33]
        ], dtype=np.float32) * sigma_w_sq
        
        # Singular Matrix 방지를 위해 아주 작은 값(epsilon)을 대각선에 더해줌 
        Q += np.eye(4) * 1e-6 # 밑줄에서 종종 역행렬 없는 오류 있어서 작은 값 더해줌 
        Q_inv = np.linalg.inv(Q) # determinant 구해서 0 나오면 그 때만 노이즈 추가하는 식으로 변경할 수도 있음

        # 3. 정규화 행렬 Lambda (27)
        Lambda = np.diag([1.0, 1.0, dt, dt]).astype(np.float32)

        for t in self.targets:
            # t.is_detected = False
            t.measurement_info = np.zeros((4, 4), dtype=np.float32) # 각 타임 슬롯마다 초기화

        for uav in self.uavs:
            uav.is_detected = False
            target = self.targets[uav.assigned_target_id]
            snr_sen, e_sen = uav.get_sensing_model(target.pos, self.dt)
            e_comm = uav.get_comm_model(self.bs_pos, self.dt)
            e_move = uav.get_mobility_energy(self.dt)

            e_tot = e_sen + e_comm + e_move # 이번 슬롯에서 총 에너지 소모량
            uav.energy -= e_tot # 에너지 차감
            uav.last_e_tot = e_tot # 마지막 에너지 소모량 저장 (보상 계산할때 쓰임)

            if snr_sen >= self.snr_threshold: # 임계값 넘기면 성공
                uav.is_detected = True
                ## (11), 식은 uav 중심 위치여서 대신 dx dy로 xkt, ykt 표현
                dx, dy = target.pos[0] - uav.pos[0], target.pos[1] - uav.pos[1]
                dist_2d_sq = dx**2 + dy**2
                if dist_2d_sq > 0:
                    H = np.array([[dx/np.sqrt(dist_2d_sq), dy/np.sqrt(dist_2d_sq), 0, 0],
                                [-dy/dist_2d_sq, dx/dist_2d_sq, 0, 0]], dtype=np.float32)
                else:
                    H = np.zeros((2, 4), dtype=np.float32) # 위치 정확히 고정한게 아니라면 이럴 가능성은 없음
                # (5) 잘 이해 안가서 일단 snr 비례하게 근사
                R_inv = np.array([[snr_sen, 0], [0, snr_sen/0.1]], dtype=np.float32)
                target.measurement_info += H.T @ R_inv @ H # 알파ukt 곱 안하는 이유: 0이면 어차피 if문 안들어옴

        # PCRLB 및 est_state 갱신
        for target in self.targets:
            prior_J = F_inv.T @ target.J_matrix @ F_inv + Q_inv # (26)의 Q부분까지
            target.J_matrix = prior_J + target.measurement_info
            try:
                PCRLB = np.linalg.inv(target.J_matrix) # J(Skt)의 역행렬
                target.F_ukt = np.trace(Lambda @ PCRLB @ Lambda.T)
            except np.linalg.LinAlgError:
                target.F_ukt = 1000.0 # 임의값

            # 임시로 EKF 직접 안하고 무작위 노이즈로 (추후 수정)
            noise_std = np.sqrt(max(target.F_ukt, 1e-6)) * 0.1
            noise = np.random.normal(0, noise_std, size=4)
            target.est_state = np.concatenate([target.pos, target.velocity]) + noise 

    def _calculate_rewards(self):
        """Eq (31) 기반 다중 목적 보상 계산"""
        rewards = []
        lam1, lam2, lam3 = 1.0, 50.0, 100.0 
        epsilon = 5.0 
        d_th = 100.0  
        epsilon_0 = 0.01 

        for uav in self.uavs:
            target = self.targets[uav.assigned_target_id]
            d_kt_Z = min([np.linalg.norm(target.pos - cz) for cz in self.critical_zones])
            
            r1 = -uav.last_e_tot
            
            # W_t를 곱하여 위험 구역에서 추적 오차에 더 큰 페널티를 부여
            r2 = -lam1 * max(target.F_ukt - epsilon, 0) * uav.W_t
            
            alpha_ukt = 1 if uav.is_detected else 0
            r3 = -lam2 * (1 - alpha_ukt)
            
            r4 = 0
            if d_kt_Z <= d_th:
                r4 = -(lam3 / (d_kt_Z**2 + epsilon_0))
            
            total_reward = r1 + r2 + r3 + r4
            rewards.append([total_reward])
            
        return np.array(rewards, dtype=np.float32)
    
# TO-DO
# BS 퓨전 도입하기
# EKF 정식 도입하기
# 식 (5) 도입하기
# AAI에 LLM 도입하기
# 논문 수정되면 반영하기