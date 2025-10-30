# Flappy Bird Reinforcement Learning

This project implements a **Deep Q-Network (DQN)** agent to play the **Flappy Bird** game.  
It leverages **experience replay**, **target networks**, and **epsilon-greedy exploration** to train a neural network that learns to play Flappy Bird autonomously.

---

## Trained Agent Demo

Below is a GIF showing the DQN agent successfully playing Flappy Bird after training.

<p align="center">
  <img src="results/demo.gif" alt="Flappy Bird agent playing autonomously">
</p>

---

## Features

- **Deep Q-Learning (DQN)** with experience replay
- Target network updates for stable training
- Epsilon-greedy action selection
---

## Code Structure

### 1. Environment Setup
- The Flappy Bird environment is created using `gymnasium` and `flappy_bird_gymnasium`.


### 2. Initialization
   - Two neural networks: `q_network` and `target_q_network`
   - Replay buffer to store experience tuples

### 3. Training Loop
   - Epsilon-greedy policy for exploration vs. exploitation
   - Stores `(state, action, reward, next_state, done)` in memory
   - Periodically samples mini-batches to update Q-values
   - Synchronizes target network weights every fixed number of steps

### 4. Testing
   - After training, the trained model plays the game visually

---

## Usage

### 1. Clone or download this repository
```bash
git clone https://github.com/Mahdi-Razi-Gandomani/flappy-bird-DQN.git
cd flappy-bird-DQN
```

### 2. Run training
```bash
python3 flappyBirdDQN.py
```

During training, you'll see logs like:
```
Episode 100 | Total point average of the last 100 episodes: -6.28
Episode 200 | Total point average of the last 100 episodes: -2.11
```

### 3. Watch the trained agent play
Once training is complete, the script automatically runs a few test episodes:
```
env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
```

You can also load and run the saved model manually:
```python
from tensorflow.keras.models import load_model
import flappy_bird_gymnasium
import gymnasium as gym
import numpy as np

env = gym.make('FlappyBird-v0', render_mode='human', use_lidar=False)
model = load_model('flappy_bird_model.h5')

observation, info = env.reset()
done = False
while not done:
    state_input = np.expand_dims(observation, axis=0)
    q_values = model.predict(state_input)
    action = np.argmax(q_values[0])
    state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
```
---
## Learning Curve

The following plot shows the average reward progression over 5000 episodes of training:

![Learning Curve](results/learning_curve.png) 

---

## Hyperparameters

| Parameter | Description | Default |
|------------|-------------|----------|
| `MEMORY_SIZE` | Size of replay memory | 100000 |
| `GAMMA` | Discount factor | 0.99 |
| `ALPHA` | Learning rate | 1e-4 |
| `BATCH_SIZE` | Mini-batch size | 128 |
| `EPSILON_START` | Initial exploration rate | 1.0 |
| `EPSILON_END` | Minimum exploration rate | 0.02 |
| `EPSILON_DECAY` | Epsilon decay factor | 0.995 |
| `UPDATE_TARGET_EVERY` | Target network update frequency | 1000 steps |
| `NUM_P_AV` | Average score window size | 100 |

---
