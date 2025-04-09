import itertools
import os
import pickle
import random
import sys
from copy import deepcopy

import numpy as np
import torch
from stable_baselines3.common import vec_env
from torch.optim import Adam

import src.thrifty.algos.core as core
from src.thrifty.utils.logx import EpochLogger


class ReplayBuffer:
    """
    A simple FIFO experience replay buffer.
    """

    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act):
        self.obs_buf[self.ptr] = obs
        self.act_buf[self.ptr] = act
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     act=self.act_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}

    def fill_buffer(self, obs, act):
        # Replace loop with vectorized operation
        indices = np.arange(len(obs))
        self.obs_buf[indices % self.max_size] = obs
        self.act_buf[indices % self.max_size] = act
        self.ptr = (self.ptr + len(obs)) % self.max_size
        self.size = min(self.size + len(obs), self.max_size)

    def save_buffer(self, name='replay'):
        pickle.dump({'obs_buf': self.obs_buf, 'act_buf': self.act_buf,
                     'ptr': self.ptr, 'size': self.size}, open('{}_buffer.pkl'.format(name), 'wb'))
        print('buf size', self.size)

    def load_buffer(self, name='replay'):
        p = pickle.load(open('{}_buffer.pkl'.format(name), 'rb'))
        self.obs_buf = p['obs_buf']
        self.act_buf = p['act_buf']
        self.ptr = p['ptr']
        self.size = p['size']

    def clear(self):
        self.ptr, self.size = 0, 0


class QReplayBuffer:
    # Replay buffer for training Qrisk
    def __init__(self, obs_dim, act_dim, size, device):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size
        self.device = device

    def store(self, obs, act, next_obs, rew, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.done_buf[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample_batch(self, batch_size=32, pos_fraction=None):
        if pos_fraction is not None:
            pos_size = min(len(tuple(np.argwhere(self.rew_buf).ravel())), int(batch_size * pos_fraction))
            neg_size = batch_size - pos_size
            pos_idx = np.array(random.sample(tuple(np.argwhere(self.rew_buf).ravel()), pos_size))
            neg_idx = np.array(random.sample(tuple(np.argwhere((1 - self.rew_buf)[:self.size]).ravel()), neg_size))
            idxs = np.hstack((pos_idx, neg_idx))
            np.random.shuffle(idxs)
        else:
            idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32, device=self.device) for k, v in batch.items()}

    def fill_buffer(self, data):
        obs_dim = data['obs'].shape[1]
        for i in range(len(data['obs'])):
            # If time boundary without a positive reward, skip storing
            if data['done'][i] and not data['rew'][i]:
                continue
            elif data['done'][i] and data['rew'][i]:
                self.store(data['obs'][i], data['act'][i],
                           np.zeros(obs_dim, dtype=np.float32),
                           data['rew'][i], data['done'][i])
            else:
                self.store(data['obs'][i], data['act'][i],
                           data['obs'][i + 1],
                           data['rew'][i], data['done'][i])

    def fill_buffer_from_BC(self, data, goals_only=False):
        """
        Load buffer from offline demos (only obs/act).
        goals_only: if True, only store transitions with a positive reward.
        (Currently, the parameter is kept for API consistency; you can extend its behavior as needed.)
        """
        num_bc = len(data['obs'])
        obs_dim = data['obs'].shape[1]
        for i in range(num_bc - 1):
            self.store(data['obs'][i], data['act'][i], data['obs'][i + 1], 0, 0)
        self.store(data['obs'][num_bc - 1], data['act'][num_bc - 1],
                   np.zeros(obs_dim, dtype=np.float32), 1, 1)

    def clear(self):
        self.ptr, self.size = 0, 0


def generate_offline_data(venv, expert_policy, action_space, num_episodes=0, output_file='data.pkl', seed=0):
    # Determine observation and action shapes from the environment
    obs_shape = venv.observation_space.shape
    act_shape = action_space.shape
    num_envs = venv.num_envs
    act_limit = venv.action_space.high[0]

    np.random.seed(seed)

    # Start with an initial capacity (number of total samples) guess
    init_capacity = 600 * num_envs * num_episodes
    obs_data = np.empty((init_capacity, *obs_shape), dtype=np.float32)
    act_data = np.empty((init_capacity, *act_shape), dtype=np.float32)
    rew_data = np.empty(init_capacity, dtype=np.float32)

    sample_idx = 0  # Counter for total samples collected

    def resize_arrays(new_capacity):
        nonlocal obs_data, act_data, rew_data
        obs_data = np.resize(obs_data, (new_capacity, *obs_shape))
        act_data = np.resize(act_data, (new_capacity, *act_shape))
        rew_data = np.resize(rew_data, new_capacity)

    for episode in range(num_episodes):
        print(f'Episode #{episode}')
        total_ret, t = 0, 0
        venv_obs = venv.reset()
        active = np.ones(num_envs, dtype=bool)

        while np.any(active):
            # Predict actions for all environments
            venv_act = expert_policy._predict(venv_obs)
            venv_act = venv_act.cpu().numpy().reshape((-1, *act_shape))
            venv_act = np.clip(venv_act, -act_limit, act_limit)

            # Identify active environment indices
            active_indices = np.nonzero(active)[0]
            n_new = len(active_indices)

            # Ensure capacity in our preallocated arrays
            if sample_idx + n_new > obs_data.shape[0]:
                new_capacity = max(obs_data.shape[0] * 2, sample_idx + n_new)
                resize_arrays(new_capacity)

            # Store observations and actions for active environments
            obs_data[sample_idx:sample_idx + n_new] = venv_obs[active_indices]
            act_data[sample_idx:sample_idx + n_new] = venv_act[active_indices]

            # Step the environment
            venv_obs, rewards, dones, info = venv.step(venv_act)
            rew_data[sample_idx:sample_idx + n_new] = rewards[active_indices]

            sample_idx += n_new
            active &= ~dones
            total_ret += np.sum(rewards)
            t += 1
        print("Collected episode with return {} length {}".format(total_ret, sample_idx))
    # Trim arrays to the actual number of samples
    obs_data = obs_data[:sample_idx]
    act_data = act_data[:sample_idx]
    rew_data = rew_data[:sample_idx]


    print("Ep Mean, Std Dev:", rew_data.mean(), np.array(rew_data).std())
    pickle.dump({'obs': obs_data, 'act': act_data}, open(output_file, 'wb'))


def thrifty(venv: vec_env.VecEnv, iters=5, actor_critic=core.Ensemble, ac_kwargs=dict(),
            seed=0, grad_steps=500, obs_per_iter=2000, replay_size=int(3e4), pi_lr=1e-3,
            batch_size=64, logger_kwargs=dict(), num_test_episodes=10, bc_epochs=5,
            input_file='data.pkl', device_idx=0, expert_policy=None, num_nets=5,
            target_rate=0.1, hg_dagger=None,
            q_learning=False, gamma=0.9999, init_model=None):
    """
    obs_per_iter: environment steps per algorithm iteration
    num_nets: number of neural nets in the policy ensemble
    input_file: where initial BC data is stored (output of generate_offline_data())
    target_rate: desired rate of context switching
    robosuite: whether to enable robosuite specific code (and use robosuite_cfg)
    hg_dagger: if not None, use this function as the switching condition (i.e. run HG-DAgger)
    q_learning: if True, train Q_risk safety critic
    gamma: discount factor for Q-learning
    num_test_episodes: run this many episodes after each iter without interventions
    init_model: initial NN weights
    """
    print('start thrifty')
    logger = EpochLogger(**logger_kwargs)
    # _locals = locals()
    # del _locals['venv']
    # logger.save_config(_locals)
    if device_idx >= 0:
        device = torch.device("cuda", device_idx)
    else:
        device = torch.device("cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs_dim = venv.observation_space.shape
    act_dim = venv.action_space.shape[0]
    act_limit = venv.action_space.high[0]
    # assert act_limit == -1 * venv.action_space.low[0], "Action space should be symmetric"

    # venv specific
    num_envs = venv.num_envs

    held_out_data, qbuffer, replay_buffer = create_buffer(act_dim, device, input_file, obs_dim, replay_size)
    # initialize actor and classifier NN
    ac = actor_critic(venv.observation_space, venv.action_space, device, num_nets=num_nets, **ac_kwargs)
    if init_model:
        ac = torch.load(init_model, map_location=device).to(device)
        ac.device = device
    ac_targ = deepcopy(ac)
    for p in ac_targ.parameters():
        p.requires_grad = False

    # Set up optimizers
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)

    # Set up model saving
    # logger.setup_pytorch_saver(ac)


    # Set up function for computing actor loss
    def compute_loss_pi(data, i):
        o, a = data['obs'], data['act']
        a_pred = ac.pis[i](o)
        return torch.mean(torch.sum((a - a_pred) ** 2, dim=1))

    def compute_loss_q(data):
        o, a, o2, r, d = data['obs'], data['act'], data['obs2'], data['rew'], data['done']
        # Compute target action and Q-values in a single no_grad block.
        with torch.no_grad():
            # Compute the average action prediction over all ensemble policies.
            a2 = torch.mean(torch.stack([pi(o2) for pi in ac.pis], dim=0), dim=0)
            # Compute target Q-values.
            q1_t = ac_targ.q1(o2, a2)
            q2_t = ac_targ.q2(o2, a2)
            backup = r + gamma * (1 - d) * torch.min(q1_t, q2_t)
        # Compute current Q estimates.
        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)
        # Use the built-in MSE loss for clarity.
        loss_q1 = torch.nn.functional.mse_loss(q1, backup)
        loss_q2 = torch.nn.functional.mse_loss(q2, backup)
        return loss_q1 + loss_q2

    def update_pi(data, i):
        # run one gradient descent step for pi.
        pi_optimizers[i].zero_grad()
        loss_pi = compute_loss_pi(data, i)
        loss_pi.backward()
        pi_optimizers[i].step()
        return loss_pi.item()

    def update_q(data, timer):
        q_optimizer.zero_grad()
        loss_q = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()
        # Update target network using Polyak averaging every two updates.
        if timer % 2 == 0:
            tau = 0.995
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # torch.lerp performs a linear interpolation:
                    # output = p_targ + (1 - tau) * (p - p_targ)
                    p_targ.data.copy_(torch.lerp(p_targ.data, p.data, 1 - tau))
        return loss_q.item()

    # Prepare for interaction with environment
    online_burden = 0  # how many labels we get from supervisor
    num_switch_to_human = 0  # context switches (due to novelty)
    num_switch_to_human2 = 0  # context switches (due to risk)
    num_switch_to_robot = 0


    if iters == 0 and num_test_episodes > 0:  # only run evaluation.
        test_agent(venv, ac, act_dim, act_limit, num_test_episodes, logger_kwargs, 0)
        sys.exit(0)

    # train policy
    for net_idx in range(ac.num_nets):
        if ac.num_nets > 1:  # create new datasets via sampling with replacement
            print('Net #{}'.format(net_idx))
            tmp_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            tmp_buffer = replay_buffer
        for _ in range(bc_epochs):
            loss_pi = []
            for _ in range(grad_steps):
                batch = tmp_buffer.sample_batch(batch_size)
                loss_pi.append(update_pi(batch, net_idx))
            validation = []
            for j in range(len(held_out_data['obs'])):
                a_pred = ac.act(held_out_data['obs'][j], i=net_idx)
                a_sup = held_out_data['act'][j]
                validation.append(sum(a_pred - a_sup) ** 2)
            print('LossPi', sum(loss_pi) / len(loss_pi))
            print('LossValid', sum(validation) / len(validation))

    # estimate switch-back parameter and initial switch-to parameter from data
    switch2human_thresh, switch2human_thresh2, switch2robot_thresh, switch2robot_thresh2 = estimate_threshholds(ac,
                                                                                                                held_out_data,
                                                                                                                num_envs,
                                                                                                                replay_buffer,
                                                                                                                target_rate)

    torch.cuda.empty_cache()
    # we only needed the held out set to check valid loss and compute thresholds, so we can get rid of it.
    replay_buffer.fill_buffer(held_out_data['obs'], held_out_data['act'])

    total_env_interacts = 0
    ep_num = 0
    fail_ct = 0
    for t in range(iters + 1):
        print('start episode', t)
        logging_data = []  # for verbose logging
        estimates = [[] for _ in range(num_envs)]
        estimates2 = [[] for _ in range(num_envs)]  # refit every iter
        i = 0
        if t == 0:  # skip data collection on iter 0 to train Q
            i = obs_per_iter
        while i < obs_per_iter:

            venv_obs = venv.reset()
            dones = np.zeros(num_envs, dtype=bool)
            active = np.ones(num_envs, dtype=bool)
            expert_mode = np.zeros(num_envs, dtype=bool)
            ep_len = np.zeros(num_envs)

            obs_list = [[] for _ in range(num_envs)]
            act_list = [[] for _ in range(num_envs)]
            rew_list = [[] for _ in range(num_envs)]
            done_list = [[] for _ in range(num_envs)]
            sup_list = [[] for _ in range(num_envs)]
            var_list = [[] for _ in range(num_envs)]
            risk_list = [[] for _ in range(num_envs)]
            for env_idx in range(num_envs):
                obs_list[env_idx].append(venv_obs[env_idx])
                var_list[env_idx].append(ac.variance(venv_obs[env_idx]))

            while i < obs_per_iter:
                venv_act = expert_policy._predict(venv_obs)
                venv_act = venv_act.cpu().numpy().reshape((-1, act_dim))
                venv_act = np.clip(venv_act, -act_limit, act_limit)
                first_done = np.ones(num_envs, dtype=bool)
                for env_idx in range(num_envs):
                    if not active[env_idx]:
                        continue
                    obs = venv_obs[env_idx]
                    a = ac.act(obs)
                    a = np.clip(a, -act_limit, act_limit)
                    if not expert_mode[env_idx]:
                        estimates[env_idx].append(ac.variance(obs))
                        estimates2[env_idx].append(ac.safety(obs, a))

                        venv_act[env_idx] = a
                    if expert_mode[env_idx]:
                        a_expert = venv_act[env_idx]

                        replay_buffer.store(obs, a_expert)
                        online_burden += 1
                        risk_list[env_idx].append(ac.safety(obs, a_expert))
                        # print('switch2robot_thresh: ', switch2robot_thresh[env_idx])
                        # print('sum((a - a_expert) ** 2): ', sum((a - a_expert) ** 2))
                        # print('switch2robot_thresh2: ', switch2robot_thresh2[env_idx])
                        # print('ac.safety(obs, a): ', ac.safety(obs, a))
                        if (hg_dagger and a_expert[3] != 0) or (
                                not hg_dagger and sum((a - a_expert) ** 2) < switch2robot_thresh[env_idx]
                                and (not q_learning or ac.safety(obs, a) > switch2robot_thresh2[env_idx])):
                            # print("Switch to Robot")
                            expert_mode[env_idx] = False
                            num_switch_to_robot += 1

                        act_list[env_idx].append(a_expert)
                        sup_list[env_idx].append(1)
                    # hg-dagger switching for hg-dagger, or novelty switching for thriftydagger
                    elif (hg_dagger and hg_dagger()) or (
                            not hg_dagger and ac.variance(obs) > switch2human_thresh[env_idx]):
                        # print("Switch to Human (Novel)")
                        num_switch_to_human += 1
                        expert_mode[env_idx] = True
                        continue
                    # second switch condition: if not novel, but also not safe
                    elif not hg_dagger and q_learning and ac.safety(obs, a) < switch2human_thresh2[env_idx]:
                        # print("Switch to Human (Risk)")
                        num_switch_to_human2 += 1
                        expert_mode[env_idx] = True
                        continue
                    else:
                        risk_list[env_idx].append(ac.safety(obs, a))
                        act_list[env_idx].append(a)
                        sup_list[env_idx].append(0)
                    ep_len[env_idx] += 1
                    var_list[env_idx].append(ac.variance(obs))
                i += 1
                next_venv_obs, reward, dones, infos = venv.step(venv_act)
                active &= ~dones
                for idx, (is_active, is_first, d, r) in enumerate(zip(active, first_done, dones, reward)):
                    if is_active or is_first:
                        done_list[idx].append(d)
                        rew_list[idx].append(r)
                        qbuffer.store(venv_obs[idx], venv_act[idx], next_venv_obs[idx], r,
                                      d)
                        if not is_active:
                            first_done[idx] = False

                venv_obs = next_venv_obs
            ep_num += np.sum(dones)
            fail_ct += np.sum(["distance_truncated" in info  or "TimeLimit.truncated" in info for info in infos])
            total_env_interacts += np.sum(ep_len)
            # logging_data.append({'obs': np.array(obs_list), 'act': np.array(act_list), 'done': np.array(done_list),
            #                      'rew': np.array(rew_list), 'sup': np.array(sup_list), 'var': np.array(var_list),
            #                      'risk': np.array(risk_list), 'beta_H': np.array(switch2human_thresh),
            #                      'beta_R': np.array(switch2robot_thresh), 'eps_H': np.array(switch2human_thresh2),
            #                      'eps_R': np.array(switch2robot_thresh2),
            #                      })
            # pickle.dump(logging_data, open(logger_kwargs['output_dir'] + '/iter{}.pkl'.format(t), 'wb'))

            # recompute thresholds from data after every episode
            recompute_thresholds(estimates, estimates2, num_envs, switch2human_thresh, switch2human_thresh2,
                                 switch2robot_thresh2, target_rate)

        if t > 0:
            # retrain policy from scratch
            ac, loss_pi = train_policy(ac, ac_kwargs, act_dim, actor_critic, batch_size, bc_epochs, device, grad_steps,
                                       loss_pi, num_nets, obs_dim, pi_lr, replay_buffer, replay_size, t, update_pi,
                                       venv)
        # retrain Qrisk
        if q_learning:
            loss_q = train_q(ac, act_dim, act_limit, batch_size, bc_epochs, grad_steps, logger_kwargs,
                             num_test_episodes, pi_lr, qbuffer, t, update_q, venv)

        # end of epoch logging
        logger.save_state(dict())
        print('Epoch', t)
        if t > 0:
            print('LossPi', sum(loss_pi) / len(loss_pi))
        if q_learning:
            print('LossQ', sum(loss_q) / len(loss_q))
        print('TotalEpisodes', ep_num)
        print('TotalSuccesses', ep_num - fail_ct)
        print('TotalEnvInteracts', total_env_interacts)
        print('OnlineBurden', online_burden)
        print('NumSwitchToNov', num_switch_to_human)
        print('NumSwitchToRisk', num_switch_to_human2)
        print('NumSwitchBack', num_switch_to_robot)
    return ac


def create_buffer(act_dim, device, input_file, obs_dim, replay_size):
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    input_data = pickle.load(open(input_file, 'rb'))
    # shuffle and create small held out set to check valid loss
    num_bc = len(input_data['obs'])
    idxs = np.arange(num_bc)
    np.random.shuffle(idxs)
    replay_buffer.fill_buffer(input_data['obs'][idxs][:int(0.9 * num_bc)], input_data['act'][idxs][:int(0.9 * num_bc)])
    held_out_data = {'obs': input_data['obs'][idxs][int(0.9 * num_bc):],
                     'act': input_data['act'][idxs][int(0.9 * num_bc):]}
    qbuffer = QReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
    qbuffer.fill_buffer_from_BC(input_data)
    return held_out_data, qbuffer, replay_buffer


def estimate_threshholds(ac, held_out_data, num_envs, replay_buffer, target_rate):
    discrepancies, estimates = [], []
    for buffer_idx in range(replay_buffer.size):
        a_pred = ac.act(replay_buffer.obs_buf[buffer_idx])
        a_sup = replay_buffer.act_buf[buffer_idx]
        discrepancies.append(sum((a_pred - a_sup) ** 2))
        estimates.append(ac.variance(replay_buffer.obs_buf[buffer_idx]))
    heldout_discrepancies, heldout_estimates = [], []
    for data_idx in range(len(held_out_data['obs'])):
        a_pred = ac.act(held_out_data['obs'][data_idx])
        a_sup = held_out_data['act'][data_idx]
        heldout_discrepancies.append(sum((a_pred - a_sup) ** 2))
        heldout_estimates.append(ac.variance(held_out_data['obs'][data_idx]))
    s2rt = np.array(discrepancies).mean()
    switch2robot_thresh = np.full(num_envs, s2rt)
    target_idx = int((1 - target_rate) * len(heldout_estimates))
    s2ht = sorted(heldout_estimates)[target_idx]
    switch2human_thresh = np.full(num_envs, s2ht)
    print("Estimated switch-back threshold: {}".format(switch2robot_thresh))
    print("Estimated switch-to threshold: {}".format(switch2human_thresh))
    switch2human_thresh2 = np.full(num_envs,
                                   0.48)  # a priori guess: 48% discounted probability of success. Could also estimate from data
    switch2robot_thresh2 = np.full(num_envs, 0.3)
    return switch2human_thresh, switch2human_thresh2, switch2robot_thresh, switch2robot_thresh2


def train_policy(ac, ac_kwargs, act_dim, actor_critic, batch_size, bc_epochs, device, grad_steps, loss_pi, num_nets,
                 obs_dim, pi_lr, replay_buffer, replay_size, t, update_pi, venv):
    loss_pi = []
    ac = actor_critic(venv.observation_space, venv.action_space, device, num_nets=num_nets, **ac_kwargs)
    pi_optimizers = [Adam(ac.pis[i].parameters(), lr=pi_lr) for i in range(ac.num_nets)]
    for net_idx in range(ac.num_nets):
        if ac.num_nets > 1:  # create new datasets via sampling with replacement
            tmp_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size, device=device)
            for _ in range(replay_buffer.size):
                idx = np.random.randint(replay_buffer.size)
                tmp_buffer.store(replay_buffer.obs_buf[idx], replay_buffer.act_buf[idx])
        else:
            tmp_buffer = replay_buffer
        for _ in range(grad_steps * (bc_epochs + t)):
            batch = tmp_buffer.sample_batch(batch_size)
            loss_pi.append(update_pi(batch, net_idx))
    return ac, loss_pi


def train_q(ac, act_dim, act_limit, batch_size, bc_epochs, grad_steps, logger_kwargs, num_test_episodes, pi_lr, qbuffer,
            t, update_q, venv):
    if num_test_episodes > 0:
        rollout_data = test_agent(venv, ac, act_dim, act_limit, num_test_episodes, logger_kwargs,
                                  t)  # collect samples offline from pi_R
        qbuffer.fill_buffer(rollout_data)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())
    q_optimizer = Adam(q_params, lr=pi_lr)
    loss_q = []
    total_q_updates = bc_epochs * grad_steps * 5
    for update_count in range(total_q_updates):
        batch = qbuffer.sample_batch(batch_size // 2)
        loss_q.append(update_q(batch, timer=update_count))
    return loss_q


def recompute_thresholds(estimates, estimates2, num_envs, switch2human_thresh, switch2human_thresh2,
                         switch2robot_thresh2, target_rate):
    for env_idx in range(num_envs):
        print("len(estimates): {}".format(len(estimates[env_idx])))
        if len(estimates[env_idx]) > 25:
            target_idx = int((1 - target_rate) * len(estimates[env_idx]))
            switch2human_thresh[env_idx] = sorted(estimates[env_idx])[target_idx]
            switch2human_thresh2[env_idx] = sorted(estimates2[env_idx], reverse=True)[target_idx]
            switch2robot_thresh2[env_idx] = sorted(estimates2[env_idx])[int(0.5 * len(estimates[env_idx]))]
            print("len(estimates): {}, New switch thresholds: {} {} {}".format(len(estimates[env_idx]),
                                                                               switch2human_thresh[env_idx],
                                                                               switch2human_thresh2[env_idx],
                                                                               switch2robot_thresh2[env_idx]))


def test_agent(venv, ac, act_dim, act_limit, num_test_episodes, logger_kwargs=None, epoch=0):
    """Run test episodes"""
    num_envs = venv.num_envs
    obs, act, done, rew = [], [], [], []
    for j in range(num_test_episodes):
        venv_obs = venv.reset()
        dones = np.zeros(num_envs, dtype=bool)
        active = np.ones(num_envs, dtype=bool)
        success = np.ones(num_envs, dtype=bool)
        first_done = np.ones(num_envs, dtype=bool)
        while np.any(active):
            venv_act = np.zeros((num_envs, act_dim))
            for env_idx in range(num_envs):
                if not active[env_idx]:
                    continue
                obs.append(venv_obs[env_idx])
                a = ac.act(venv_obs[env_idx])
                a = np.clip(a, -act_limit, act_limit)
                venv_act[env_idx] = a
                act.append(a)
            venv_obs, rewards, dones, info = venv.step(venv_act)
            active &= ~dones
            for idx, (is_active, is_first, d, r, i) in enumerate(zip(active, first_done, dones, rewards, info)):
                if is_active or is_first:
                    done.append(d)
                    rew.append(r)
                    if not is_active:
                        if "distance_truncated" in i or "TimeLimit.truncated" in i:
                            success[idx] = False
                        first_done[idx] = False

        print('episode #{} success? {}'.format(j, success))
    print('Test Success Rate:', sum(rew) / num_test_episodes)
    # pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)},
    #             open('test-rollouts.pkl', 'wb'))
    # pickle.dump({'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)},
    #             open(logger_kwargs['output_dir'] + '/test{}.pkl'.format(epoch), 'wb'))

    return {'obs': np.stack(obs), 'act': np.stack(act), 'done': np.array(done), 'rew': np.array(rew)}


