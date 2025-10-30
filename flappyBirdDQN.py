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
GAMMA = 0.99
ALPHA = 1e-4
BATCH_SIZE = 128
UPDATE_TARGET_EVERY = 1000
NUM_STEPS_FOR_UPDATE = 4 # perform a learning update every C time steps

EPSILON_START = 1.0
EPSILON_END = 0.02
EPSILON_DECAY = 0.995

MAX_EPISODES = 10000    
MAX_TIMESTEPS = 10000
NUM_P_AV = 100
NUM_TEST_EPISODES = 10

env = gym.make('FlappyBird-v0', use_lidar=False)
state, _ = env.reset()
state_size = env.observation_space.shape
num_actions = env.action_space.n

def build_q_network():
    model = Sequential([
        Input(shape=state_size),
        Dense(units=128, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=64, activation='relu'),
        Dense(units=num_actions, activation='linear'),
    ])
    return model

q_network = build_q_network()
target_q_network = build_q_network()
target_q_network.set_weights(q_network.get_weights())

# Store experiences as named tuples
experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
optimizer = Adam(learning_rate=ALPHA)

def compute_loss(experiences, gamma, q_network, target_q_network):
    states, actions, rewards, next_states, done_vals = experiences
    states = tf.convert_to_tensor(np.vstack(states))
    next_states = tf.convert_to_tensor(np.vstack(next_states))
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    done_vals = tf.convert_to_tensor(done_vals, dtype=tf.float32)

    next_q_values = q_network(next_states)
    next_actions = tf.argmax(next_q_values, axis=1)
    target_q_values = target_q_network(next_states)
    
    max_qsa = tf.gather_nd(target_q_values, tf.stack([tf.range(target_q_values.shape[0]),
                                                      tf.cast(next_actions, tf.int32)], axis=1))
    y_targets = rewards + (gamma * max_qsa * (1 - done_vals))
    q_values = q_network(states)
    q_values = tf.gather_nd(q_values, tf.stack([tf.range(q_values.shape[0]),
                                                tf.cast(actions, tf.int32)], axis=1))
    loss = MSE(y_targets, q_values)
    return loss

def agent_learn(experiences, gamma):
    with tf.GradientTape() as tape:
        loss = compute_loss(experiences, gamma, q_network, target_q_network)
    gradients = tape.gradient(loss, q_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, q_network.trainable_variables))

epsilon = EPSILON_START
total_point_history = []
memory_buffer = deque(maxlen=MEMORY_SIZE)
start = time.time()

for i in range(MAX_EPISODES):
    state, _ = env.reset()
    total_points = 0

    for t in range(MAX_TIMESTEPS):
        if np.random.rand() < epsilon:
            action = np.random.randint(num_actions)
        else:
            state_qn = np.expand_dims(state, axis=0)
            q_values = q_network(state_qn).numpy()[0]
            action = np.argmax(q_values)

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        memory_buffer.append(experience(state, action, reward, next_state, done))
        state = next_state.copy()
        total_points += reward
        if t % NUM_STEPS_FOR_UPDATE == 0 and len(memory_buffer) >= BATCH_SIZE:
            mini_batch = np.random.choice(len(memory_buffer), BATCH_SIZE, replace=False)
            experiences = [memory_buffer[idx] for idx in mini_batch]
            experiences = experience(*zip(*experiences))
            agent_learn(experiences, GAMMA)
        
        # Update target
        if t % UPDATE_TARGET_EVERY == 0:
            target_q_network.set_weights(q_network.get_weights())

        if done:
            break

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    
    total_point_history.append(total_points)
    av_latest_points = np.mean(total_point_history[-NUM_P_AV:])
    print(f"\rEpisode {i+1} | Total point average of the last {NUM_P_AV} episodes: {av_latest_points:.2f}", end="")

    if (i+1) % NUM_P_AV == 0:
        print(f"\rEpisode {i+1} | Total point average of the last {NUM_P_AV} episodes: {av_latest_points:.2f}")

tot_time = time.time() - start
print(f"\nTotal Runtime: {tot_time:.2f} s ({(tot_time/60):.2f} min)")

# Run the environment after training
env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
for ep in range(NUM_TEST_EPISODES):
    observation, info = env.reset()

    for step in range(MAX_TIMESTEPS):
        state_input = np.expand_dims(observation, axis=0)
        q_values = q_network.predict(state_input)
        action = np.argmax(q_values[0])
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            break
env.close()
