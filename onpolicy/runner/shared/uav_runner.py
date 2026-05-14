import time
import os
import json
import pickle
import numpy as np
import torch
from onpolicy.runner.shared.base_runner import Runner
import wandb


def _t2n(x):
    return x.detach().cpu().numpy()


class UAVRunner(Runner):
    """
    UAV Tracking 환경 전용 Runner.
    """
    def __init__(self, config):
        super(UAVRunner, self).__init__(config)
        # 학습 시작 시 환경 메타데이터를 한 번만 저장 (평가 시 정확한 dim 복원용)
        self._save_run_meta()

    def _save_run_meta(self):
        """평가 시 actor를 정확히 재구성하는 데 필요한 정보를 저장."""
        meta = {
            'algorithm_name': self.algorithm_name,
            'num_agents': self.num_agents,
            'episode_length': self.episode_length,
            'use_centralized_V': self.use_centralized_V,
            'use_recurrent_policy': self.all_args.use_recurrent_policy,
            'use_naive_recurrent_policy': self.all_args.use_naive_recurrent_policy,
            'hidden_size': self.all_args.hidden_size,
            'recurrent_N': self.all_args.recurrent_N,
            'obs_shape': list(self.envs.observation_space[0].shape),
            'share_obs_shape': list(self.envs.share_observation_space[0].shape),
            'action_shape': list(self.envs.action_space[0].shape),
            'action_low': float(self.envs.action_space[0].low.min()),
            'action_high': float(self.envs.action_space[0].high.max()),
        }
        save_dir = str(self.save_dir)
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, 'run_meta.json'), 'w') as f:
            json.dump(meta, f, indent=2)
        # all_args 전체도 pickle로 저장 (정확한 복원 가능)
        try:
            with open(os.path.join(save_dir, 'all_args.pkl'), 'wb') as f:
                pickle.dump(vars(self.all_args), f)
        except Exception as e:
            print(f'[warn] all_args.pkl save failed: {e}')

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)

                # 환경에서 진짜 global state를 share_obs로 받음
                obs, share_obs, rewards, dones, infos = self.envs.step(actions_env)

                data = (obs, share_obs, rewards, dones, infos,
                        values, actions, action_log_probs, rnn_states, rnn_states_critic)

                self.insert(data)

            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.all_args.scenario_name,
                              self.algorithm_name,
                              self.experiment_name,
                              episode, episodes,
                              total_num_steps, self.num_env_steps,
                              int(total_num_steps / (end - start))))

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                self.log_train(train_infos, total_num_steps)
                env_infos = {}
                self.log_env(env_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # 환경에서 (obs, share_obs) 반환
        obs, share_obs = self.envs.reset()

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        value, action, action_log_prob, rnn_states, rnn_states_critic \
            = self.trainer.policy.get_actions(np.concatenate(self.buffer.share_obs[step]),
                                              np.concatenate(self.buffer.obs[step]),
                                              np.concatenate(self.buffer.rnn_states[step]),
                                              np.concatenate(self.buffer.rnn_states_critic[step]),
                                              np.concatenate(self.buffer.masks[step]))
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))
        rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))
        rnn_states_critic = np.array(np.split(_t2n(rnn_states_critic), self.n_rollout_threads))

        if self.envs.action_space[0].__class__.__name__ == 'Box':
            actions_env = actions
        else:
            raise NotImplementedError("UAV env는 연속 행동(Box)만 지원")

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(share_obs, obs, rnn_states, rnn_states_critic,
                           actions, action_log_probs, values, rewards, masks)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs, eval_share_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, *self.buffer.rnn_states.shape[2:]), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            self.trainer.prep_rollout()
            eval_action, eval_rnn_states = self.trainer.policy.act(np.concatenate(eval_obs),
                                                                   np.concatenate(eval_rnn_states),
                                                                   np.concatenate(eval_masks),
                                                                   deterministic=True)
            eval_actions = np.array(np.split(_t2n(eval_action), self.n_eval_rollout_threads))
            eval_rnn_states = np.array(np.split(_t2n(eval_rnn_states), self.n_eval_rollout_threads))

            eval_actions_env = eval_actions

            eval_obs, eval_share_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        eval_env_infos = {'eval_average_episode_rewards': np.sum(eval_episode_rewards, axis=0)}
        eval_average_episode_rewards = np.mean(eval_env_infos['eval_average_episode_rewards'])
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)
