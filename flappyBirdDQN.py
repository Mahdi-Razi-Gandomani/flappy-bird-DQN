from collections import deque, namedtuple
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random


MEMORY_SIZE = 100000 
GAMMA = 0.99
ALPHA = 1e-3
BATCH_SIZE = 128
UPDATE_TARGET_EVERY = 1000
NUM_STEPS_FOR_UPDATE = 4
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
MAX_EPISODES = 3000    
MAX_TIMESTEPS = 10000
NUM_P_AV = 500

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = gym.make('FlappyBird-v0', use_lidar=False)
state, _ = env.reset(seed=SEED)
state_size = env.observation_space.shape[0]
num_actions = env.action_space.n



class QNetwork(nn.Module):
    def __init__(self, state_size, num_actions):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, num_actions)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

q_network = QNetwork(state_size, num_actions).to(device)
target_q_network = QNetwork(state_size, num_actions).to(device)
target_q_network.load_state_dict(q_network.state_dict())
target_q_network.eval()

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
optimizer = optim.Adam(q_network.parameters(), lr=ALPHA)

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences
    
    states = torch.FloatTensor(np.vstack(states)).to(device)
    next_states = torch.FloatTensor(np.vstack(next_states)).to(device)
    actions = torch.LongTensor(actions).to(device)
    rewards = torch.FloatTensor(rewards).to(device)
    done_vals = torch.FloatTensor(done_vals).to(device)

    with torch.no_grad():
        next_q_values = q_network(next_states)
        next_actions = next_q_values.argmax(dim=1)
        target_q_values = target_q_network(next_states)
        max_qsa = target_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
        y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    
    q_values = q_network(states)
    q_values = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
    
    loss = F.mse_loss(q_values, y_targets)
    return loss



def agent_learn(experiences, gamma):
    loss = compute_loss(experiences, gamma, q_network, target_q_network)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()


epsilon = EPSILON_START
total_point_history = []
memory_buffer = deque(maxlen=MEMORY_SIZE)


global_counter = 0
for i in range(MAX_EPISODES):
    state, _ = env.reset(seed=SEED + i)
    total_points = 0

    for t in range(MAX_TIMESTEPS):
        global_counter += 1
        
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = q_network(state_tensor)
                action = q_values.argmax().item()


        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # Pass a pipe
        if reward > 0.5:
            reward = 10.0
        elif done:  # Died
            reward = -10.0
        else:  # Staying alive
            reward = 0.1

        memory_buffer.append(experience(state, action, reward, next_state, done))
        state = next_state.copy()
        total_points += reward
        
        if len(memory_buffer) >= 5000:
            if t % NUM_STEPS_FOR_UPDATE == 0:
                mini_batch = np.random.choice(len(memory_buffer), BATCH_SIZE, replace=False)
                experiences = [memory_buffer[idx] for idx in mini_batch]
                experiences = experience(*zip(*experiences))
                agent_learn(experiences, GAMMA)
        
        if global_counter % UPDATE_TARGET_EVERY == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-NUM_P_AV:])
    print(f"\rEpisode {i+1}.             Average of the last {NUM_P_AV} episodes: {av_latest_points:.1f}", end="")

    if (i+1) % NUM_P_AV == 0:
        print(f"\rEpisode {i+1}.             Average of the last {NUM_P_AV} episodes: {av_latest_points:.1f}")




# Test Runs
env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
q_network.eval()
test_scores = []
for ep in range(5):
    observation, info = env.reset(seed=SEED + MAX_EPISODES + ep)
    episode_reward = 0

    for step in range(MAX_TIMESTEPS):
        with torch.no_grad():
            state_tensor = torch.FloatTensor(observation).unsqueeze(0).to(device)
            q_values = q_network(state_tensor)
            action = q_values.argmax().item()
        
        observation, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        
        if terminated or truncated:
            test_scores.append(episode_reward)
            print(episode_reward)
            break

env.close()
