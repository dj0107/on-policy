"""
UAV 환경 전용 VecEnv wrapper와 Runner.
환경의 reset/step이 (obs, share_obs)와 (obs, share_obs, rewards, dones, infos)를
반환하는 형식 전용. ShareDummyVecEnv와 달리 available_actions이 없음.
"""
import numpy as np
from onpolicy.envs.env_wrappers import ShareVecEnv


class UAVDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        ShareVecEnv.__init__(
            self, len(env_fns),
            env.observation_space,
            env.share_observation_space,
            env.action_space
        )
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, dones, infos = map(np.array, zip(*results))
        
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    o, s = self.envs[i].reset()
                    obs[i] = o
                    share_obs[i] = s
            else:
                if np.all(done):
                    o, s = self.envs[i].reset()
                    obs[i] = o
                    share_obs[i] = s
        self.actions = None
        return obs, share_obs, rews, dones, infos

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs = map(np.array, zip(*results))
        return obs, share_obs

    def close(self):
        for env in self.envs:
            env.close()
