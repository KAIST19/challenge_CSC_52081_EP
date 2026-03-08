import os
import sys
import numpy as np
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from student_client import create_student_gym_env

TOKEN = 'ecole_polytechnique_RL'

IDX_HPC_TOUT = 0
IDX_HP_NMECH = 1
IDX_HPC_POUT = 5

STATE_DIM = 18
BUFFER_FILE = 'experience_buffer.json'

def log(msg):
    print(msg, flush=True)

def fmt_eta(seconds):
    if seconds < 60: return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m{s:02d}s"

def get_means(obs):
    if obs.ndim == 1: return obs.copy()
    return obs.mean(axis=0)

def obs_to_state(obs, initial_sensors, step_count, last_repair_step, total_repairs):
    means = get_means(obs)
    if initial_sensors is None:
        ratios = np.ones(9)
    else:
        ratios = means / (initial_sensors + 1e-8)
    state = np.zeros(STATE_DIM, dtype=np.float32)
    state[0:9] = means
    state[9:18] = ratios
    return state, means

class QNetwork(nn.Module):
    def __init__(self, state_dim=STATE_DIM, action_dim=3, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.LayerNorm(hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, action_dim)
        )
    def forward(self, x): return self.net(x)

class ReplayBuffer:
    def __init__(self, capacity=200000): self.buffer = deque(maxlen=capacity)
    def add(self, s, a, r, ns, d): self.buffer.append((s, a, r, ns, d))
    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (torch.FloatTensor(np.array(states)), torch.LongTensor(actions),
                torch.FloatTensor(rewards), torch.FloatTensor(np.array(next_states)), torch.FloatTensor(dones))
    def save(self, path=BUFFER_FILE):
        data = [{'state': s.tolist(), 'action': int(a), 'reward': float(r), 'next_state': ns.tolist(), 'done': bool(d)} for s, a, r, ns, d in self.buffer]
        with open(path, 'w') as f: json.dump(data, f)
    def load(self, path=BUFFER_FILE):
        if not os.path.exists(path): return 0
        try:
            with open(path, 'r') as f: data = json.load(f)
            for item in data:
                if len(item['state']) == STATE_DIM: self.add(np.array(item['state'], dtype=np.float32), item['action'], item['reward'], np.array(item['next_state'], dtype=np.float32), item['done'])
            return len(self.buffer)
        except: return 0
    def __len__(self): return len(self.buffer)

class DQNAgent:
    def __init__(self, lr=1e-4, gamma=0.995):
        self.gamma, self.q_net, self.target_net = gamma, QNetwork(), QNetwork()
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        self.replay_buffer, self.train_steps, self.episodes_done = ReplayBuffer(), 0, 0
        self.reset()
    def reset(self): self.step_count, self.total_repairs, self.initial_sensors, self.last_repair_step = 0, 0, None, 0
    def get_state(self, obs): return obs_to_state(obs, self.initial_sensors, self.step_count, self.last_repair_step, self.total_repairs)
    
    def act_expert(self, state):
        self.step_count += 1
        r_hp, r_tout, r_pout = state[10], state[9], state[14]
        deg = (1 - r_hp) + (1 - r_tout) + (1 - r_pout)
        ssr = self.step_count - self.last_repair_step
        if self.step_count >= 32: return 2
        if deg > 0.048: return 1 if ssr >= 5 else 2
        if ssr >= 8: return 1
        if ssr >= 6 and deg > 0.03: return 1
        return 0

    def act(self, obs, explore=True):
        self.step_count += 1
        state, means = self.get_state(obs)
        if self.step_count <= 2:
            if self.step_count == 2: self.initial_sensors = means.copy()
            return 0, state
        
        # Safety for DQN
        r_hp, r_tout, r_pout = state[10], state[9], state[14]
        deg = (1 - r_hp) + (1 - r_tout) + (1 - r_pout)
        if deg > 0.055: return 2, state
        if deg > 0.048 and (self.step_count - self.last_repair_step) >= 5: return 1, state

        if explore and np.random.random() < 0.1: return np.random.randint(0, 3), state
        with torch.no_grad(): action = self.q_net(torch.FloatTensor(state).unsqueeze(0)).argmax(dim=1).item()
        return action, state

    def train_batch(self, batch_size=256):
        if len(self.replay_buffer) < batch_size: return None
        s, a, r, ns, d = self.replay_buffer.sample(batch_size)
        q = self.q_net(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            na = self.q_net(ns).argmax(dim=1)
            nq = self.target_net(ns).gather(1, na.unsqueeze(1)).squeeze(1)
            target = (r / 1000.0) + self.gamma * nq * (1 - d)
        loss = nn.SmoothL1Loss()(q, target)
        self.optimizer.zero_grad(); loss.backward(); self.optimizer.step()
        self.train_steps += 1
        if self.train_steps % 200 == 0: self.target_net.load_state_dict(self.q_net.state_dict())
        return loss.item()

def run_episode(agent, mode='expert'):
    env = create_student_gym_env(user_token=TOKEN)
    obs, info = env.reset()
    agent.reset()
    transitions, total_reward, prev_state, prev_action = [], 0, None, None
    
    for step in range(100):
        if mode == 'expert':
            # Initial sensors handling for expert
            means = get_means(obs)
            if agent.step_count == 1: agent.initial_sensors = means.copy()
            state, _ = agent.get_state(obs)
            action = agent.act_expert(state)
        else:
            action, state = agent.act(obs)
        
        next_obs, reward, term, trunc, info = env.step(action=action)
        done = term or trunc
        total_reward += reward
        
        if prev_state is not None:
            transitions.append((prev_state, prev_action, reward, state, False))
        
        prev_state, prev_action = state, action
        if action == 1: agent.last_repair_step = agent.step_count
        obs = next_obs
        if done:
            ns, _ = agent.get_state(next_obs)
            transitions.append((state, action, reward, ns, True))
            break
    env.close()
    return transitions, total_reward

if __name__ == '__main__':
    agent = DQNAgent()
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'collect'
    if cmd == 'collect':
        agent.replay_buffer.load(BUFFER_FILE)
        n = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        for ep in range(n):
            try:
                t, r = run_episode(agent, mode='expert')
                for trans in t: agent.replay_buffer.add(*trans)
                agent.replay_buffer.save()
                log(f"  [{ep+1}/{n}] expert_reward={r:.0f} | buf={len(agent.replay_buffer)}")
                for _ in range(100): agent.train_batch()
                torch.save(agent.q_net.state_dict(), 'dqn_model_best.pt')
            except Exception as e: log(f"Error: {e}"); time.sleep(2)
