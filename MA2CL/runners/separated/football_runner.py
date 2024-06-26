import time

import numpy as np
import torch
import wandb

from MA2CL.runners.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()


class FootballRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""

    def __init__(self, config):
        super(FootballRunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = (
            int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        )

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_rewards = []

        train_episode_scores = [0 for _ in range(self.n_rollout_threads)]
        done_episodes_scores = []

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

                score_env = [t_info[0]["score_reward"] for t_info in infos]
                train_episode_scores += np.array(score_env)
                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[t])
                        train_episode_rewards[t] = 0
                        done_episodes_scores.append(train_episode_scores[t])
                        train_episode_scores[t] = 0

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
                    if not self.use_wandb:
                        self.writter.add_scalars(
                            "train_episode_rewards",
                            {"aver_rewards": aver_episode_rewards},
                            total_num_steps,
                        )
                    done_episodes_rewards = []

                    aver_episode_scores = np.mean(done_episodes_scores)
                    if not self.use_wandb:
                        self.writter.add_scalars(
                            "train_episode_scores",
                            {"aver_scores": aver_episode_scores},
                            total_num_steps,
                        )
                    done_episodes_scores = []
                    print(
                        "some episodes done, average rewards: {}, scores: {}".format(
                            aver_episode_rewards, aver_episode_scores
                        )
                    )

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[
                :, agent_id
            ].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
                self.buffer[agent_id].available_actions[step],
            )
            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))
        # [self.envs, agents, dim]
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

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
                *self.buffer[0].rnn_states_critic.shape[2:],
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

        # bad_masks = np.array(
        #     [
        #         [
        #             [0.0] if info[agent_id]["bad_transition"] else [1.0]
        #             for agent_id in range(self.num_agents)
        #         ]
        #         for info in infos
        #     ]
        # )

        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(
                share_obs[:, agent_id],
                obs[:, agent_id],
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
                None,
                active_masks[:, agent_id],
                available_actions[:, agent_id],
            )

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(
                self.buffer[agent_id].rewards
            )
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                if self.use_wandb:
                    wandb.log({agent_k: v}, step=total_num_steps)
                else:
                    self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        one_episode_rewards = [0 for _ in range(self.all_args.eval_episodes)]
        eval_episode_scores = []
        one_episode_scores = [0 for _ in range(self.all_args.eval_episodes)]

        eval_obs, eval_share_obs, eval_available_actions = self.eval_envs.reset()
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
            eval_actions_collector = []
            eval_rnn_states_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = self.trainer[agent_id].policy.act(
                    eval_obs[:, agent_id],
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    eval_available_actions[:, agent_id],
                    deterministic=True,
                )
                eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                eval_actions_collector.append(_t2n(eval_actions))

            eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)

            # Obser reward and next obs
            (
                eval_obs,
                eval_share_obs,
                eval_rewards,
                eval_dones,
                eval_infos,
                eval_available_actions,
            ) = self.eval_envs.step(eval_actions)
            eval_rewards = np.mean(eval_rewards, axis=1).flatten()
            one_episode_rewards += eval_rewards

            eval_scores = [t_info[0]["score_reward"] for t_info in eval_infos]
            one_episode_scores += np.array(eval_scores)

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

                    eval_episode_scores.append(one_episode_scores[eval_i])
                    one_episode_scores[eval_i] = 0

            if eval_episode >= self.all_args.eval_episodes:
                key_average = "eval_average_episode_rewards"
                key_max = "eval_max_episode_rewards"
                key_scores = "eval_average_episode_scores"
                eval_env_infos = {
                    key_average: eval_episode_rewards,
                    key_max: [np.max(eval_episode_rewards)],
                    key_scores: eval_episode_scores,
                }
                self.log_env(eval_env_infos, total_num_steps)

                print(
                    "eval average episode rewards: {}, scores: {}.".format(
                        np.mean(eval_episode_rewards), np.mean(eval_episode_scores)
                    )
                )
                break
