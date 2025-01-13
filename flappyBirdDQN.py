import time
from collections import deque, namedtuple, defaultdict
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.losses import MSE
from tensorflow.keras.optimizers.legacy import Adam

MEMORY_SIZE = 100000 
GAMMA = 0.95
ALPHA = 1e-3 
NUM_STEPS_FOR_UPDATE = 2  # perform a learning update every C time steps

env = gym.make('FlappyBird-v0', use_lidar=False)
state, _ = env.reset()

state_size = env.observation_space.shape
num_actions = env.action_space.n

# DQN
q_network = Sequential([
    Input(shape=state_size),
    Dense(units=128, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=64, activation='relu'),
    Dense(units=num_actions, activation='linear'),
])

optimizer = Adam(learning_rate=ALPHA)

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

def compute_loss(experiences, gamma, q_network):
    states, actions, rewards, next_states, done_vals = experiences
    states = tf.convert_to_tensor(np.vstack(states))
    next_states = tf.convert_to_tensor(np.vstack(next_states))
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    done_vals = tf.convert_to_tensor(done_vals, dtype=tf.float32)

    max_qsa = tf.reduce_max(q_network(next_states), axis=-1)
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)
    return loss

def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

start = time.time()

num_episodes = 20000
max_num_timesteps = 1000

total_point_history = []

num_p_av = 100 
c = 2.0   

# Initialize action counts
action_counts = np.zeros(num_actions)
total_action_count = 0

# Create a memory buffer D with capacity N
memory_buffer = deque(maxlen=MEMORY_SIZE)

for i in range(num_episodes):
    state, _ = env.reset()
    total_points = 0

    for t in range(max_num_timesteps):
        state_qn = np.expand_dims(state, axis=0)
        q_values = q_network(state_qn).numpy()[0]

        # Calculate UCB scores
        ucb_scores = q_values + c * np.sqrt(np.log(total_action_count + 1) / (action_counts + 1e-5))
        action = np.argmax(ucb_scores)

        # Update action counts
        action_counts[action] += 1
        total_action_count += 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory_buffer.append(experience(state, action, reward, next_state, done))
        if t % NUM_STEPS_FOR_UPDATE == 0 and len(memory_buffer) >= 128:
            mini_batch = np.random.choice(len(memory_buffer), 128, replace=False)
            experiences = [memory_buffer[idx] for idx in mini_batch]
            experiences = experience(*zip(*experiences))
            agent_learn(experiences, GAMMA)

        state = next_state.copy()
        total_points += reward

        if done:
            break

    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-num_p_av:])
    print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % num_p_av == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {num_p_av} episodes: {av_latest_points:.2f}")

q_network.save('flappy_bird_model_ucb.h5')

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Run the environment with UCB policy
env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
observation, info = env.reset()
for _ in range(1000):
    state_input = np.expand_dims(observation, axis=0)
    q_values = q_network.predict(state_input)
    action = np.argmax(q_values[0])
    observation, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        observation, info = env.reset()
env.close()
