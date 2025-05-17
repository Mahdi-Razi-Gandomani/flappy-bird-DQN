# Flappy Bird Reinforcement Learning

This project implements a Deep Q-Network (DQN) with Upper Confidence Bound (UCB) exploration to train a reinforcement learning model to play the Flappy Bird game. The model learns to maximize its score by interacting with the environment and using a combination of Q-learning and UCB for exploration.

---

## Features

- **Deep Q-Network (DQN)**:
  - Uses a neural network to approximate the Q-value function.
  - Trains the model to maximize cumulative rewards.

- **Upper Confidence Bound (UCB) Exploration**:
  - Balances exploration and exploitation by selecting actions based on their estimated Q-values and uncertainty.

- **Experience Replay and Training**:
  - Stores past experiences in a replay buffer to train the model on a diverse set of experiences.
  - Trains the model for a specified number of episodes.
---

## Code Structure

### 1. Environment Setup
- The Flappy Bird environment is created using `gymnasium` and `flappy_bird_gymnasium`.

### 2. Deep Q-Network (DQN)
- The network consists of:
  - Input layer with the size of the state space.
  - Two hidden layers with ReLU activation.
  - Output layer with linear activation for Q-values.

### 3. Experience Replay
- Experiences (state, action, reward, next state, done) are stored in a replay buffer.
- Mini-batches of experiences are sampled for training.

### 4. UCB Exploration
- UCB scores are calculated for each action to balance exploration and exploitation.
- Actions are selected based on the highest UCB score.

### 5. Training Loop
- The model interacts with the environment, collects experiences, and updates the Q-network.
- Training is performed for a specified number of episodes.

---

## Usage

1. Train the agent by running the script:

   ```bash
   python flappyBirdDQN.py
2. The trained model will be saved as `flappy_bird_model_ucb.h5`
