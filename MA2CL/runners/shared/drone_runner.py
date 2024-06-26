import time

import numpy as np
import torch
import wandb

from MA2CL.runners.shared.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class DroneRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(DroneRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                ) = self.collect(step)
                # actions = faulty_action(actions, self.all_args.faulty_node)
                #
                # # Obser reward and next obs
                # obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                # Obser reward and next obs
                (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                ) = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)
                reward_env = np.mean(rewards, axis=1).flatten()
                train_episode_rewards += reward_env
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0

                data = (
                    obs,
                    share_obs,
                    rewards,
                    dones,
                    infos,
                    available_actions,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (
                (episode + 1) * self.episode_length * self.n_rollout_threads
            )
            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(episode)

            # log information
            if episode % self.log_interval == 0:
                end_new = time.time()
                interval = end_new - start if episode == 0 else end_new - end
                tmp_FPS = int(self.log_interval*(self.episode_length * self.n_rollout_threads) / interval)
                end = end_new
                remaining_time = (end - start) / (episode + 1) * (episodes - episode - 1) / 3600
                total_time = remaining_time + (end - start) / 3600
                print(
                    "\n Scenario {} Algo {} Exp {} Seed {} updates {}/{} episodes, total num timesteps {}/{}, FPS_tmp {}/ FPS {}, remaining_time {:.1f}h/{:.1f}h.\n".format(
                        self.all_args.scenario,
                        self.algorithm_name,
                        self.experiment_name,
                        self.all_args.seed,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        tmp_FPS,
                        int(total_num_steps / (end - start)),
                        remaining_time,
                        total_time,
                    )
                )

                self.log_train(train_infos, total_num_steps)

                if len(done_episodes_rewards) > 0:
                    aver_episode_rewards = np.mean(done_episodes_rewards)
                    print("some episodes done, average rewards: ", aver_episode_rewards)
                    if not self.use_wandb:
                        self.writter.add_scalars(
                            "train_episode_rewards",
                            {"aver_rewards": aver_episode_rewards},
                            total_num_steps,
                        )
                    done_episodes_rewards = []

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()

        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        if self.use_share_obs:
            self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        obs_batch = np.concatenate(self.buffer.obs[step])
        if self.use_share_obs == False and "ppo" in self.algorithm_name:
            BxN, C, H, W = obs_batch.shape
            N = self.num_agents
            B = BxN // N
            state_batch = obs_batch.reshape(B, N * C, H, W).repeat(N, 0)
        else:
            state_batch = np.concatenate(self.buffer.share_obs[step]) if self.use_share_obs else None
        self.trainer.prep_rollout()
        (
            value,
            action,
            action_log_prob,
            rnn_state,
            rnn_state_critic,
        ) = self.trainer.policy.get_actions(
            state_batch,
            obs_batch,
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(self.buffer.rnn_states_critic[step]),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )
        rnn_states = np.array(np.split(_t2n(rnn_state), self.n_rollout_threads))
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_state_critic), self.n_rollout_threads)
        )

        return values, actions, action_log_probs, rnn_states, rnn_states_critic

    def insert(self, data):
        (
            obs,
            share_obs,
            rewards,
            dones,
            infos,
            available_actions,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        rnn_states_critic[dones_env == True] = np.zeros(
            (
                (dones_env == True).sum(),
                self.num_agents,
                *self.buffer.rnn_states_critic.shape[3:],
            ),
            dtype=np.float32,
        )

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        active_masks = np.ones(
            (self.n_rollout_threads, self.num_agents, 1), dtype=np.float32
        )
        active_masks[dones == True] = np.zeros(
            ((dones == True).sum(), 1), dtype=np.float32
        )
        active_masks[dones_env == True] = np.ones(
            ((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
        )

        # bad_masks = np.array([[[0.0] if info[agent_id]['bad_transition'] else [1.0] for agent_id in range(self.num_agents)] for info in infos])

        if not self.use_centralized_V:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
            None,
            active_masks,
            None,
        )

    def log_train(self, train_infos, total_num_steps):
        train_infos["average_step_rewards"] = np.mean(self.buffer.rewards)
        print("average_step_rewards is {}.".format(train_infos["average_step_rewards"]))
        for k, v in train_infos.items():
            if self.use_wandb:
                wandb.log({k: v}, step=total_num_steps)
            else:
                self.writter.add_scalars(k, {k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, _ = self.eval_envs.reset()
        eval_rnn_states = np.zeros(
            (
                self.all_args.eval_episodes,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones(
            (self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32
        )

        while True:
            self.trainer.prep_rollout()
            eval_actions, eval_rnn_states = self.trainer.policy.act(
                np.concatenate(eval_share_obs),
                np.concatenate(eval_obs),
                np.concatenate(eval_rnn_states),
                np.concatenate(eval_masks),
                deterministic=True,
            )
            eval_actions = np.array(
                np.split(_t2n(eval_actions), self.all_args.eval_episodes)
            )
            eval_rnn_states = np.array(
                np.split(_t2n(eval_rnn_states), self.all_args.eval_episodes)
            )

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                _,
            ) = self.eval_envs.step(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_dones_env = np.all(eval_dones, axis=1)
            eval_rnn_states[eval_dones_env == True] = np.zeros(
                (
                    (eval_dones_env == True).sum(),
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            eval_masks = np.ones(
                (self.all_args.eval_episodes, self.num_agents, 1), dtype=np.float32
            )
            eval_masks[eval_dones_env == True] = np.zeros(
                ((eval_dones_env == True).sum(), self.num_agents, 1), dtype=np.float32
            )

            for eval_i in range(self.all_args.eval_episodes):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[eval_i])
                    one_episode_rewards[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = "eval_average_episode_rewards"
                key_max = "eval_max_episode_rewards"
                eval_env_infos = {
                    key_average: eval_episode_rewards,
                    key_max: [np.max(eval_episode_rewards)],
                }
                self.log_env(eval_env_infos, total_num_steps)
                print(
                    "eval_average_episode_rewards is {}.".format(
                        np.mean(eval_episode_rewards)
                    )
                )
                break
