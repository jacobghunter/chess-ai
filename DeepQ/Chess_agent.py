import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
from Chess_nn import ChessNN
import numpy as np


if torch.backends.mps.is_available():
  device = torch.device("mps")
elif torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'next_state', 'done'])
class DeepQLearningAgent:
  def __init__(self, state_size, action_size):
      self.state_size = state_size
      self.action_size = action_size
      self.memory = deque(maxlen=100000)  # Increased memory size

      # Hyperparameters
      self.gamma = 0.95        # Discount factor
      self.epsilon = 1.0       # Exploration rate
      self.epsilon_min = 0.01
      self.epsilon_decay = 0.995
      self.learning_rate = 0.00001
      self.target_update_frequency = 10
      self.batch_size = 64

      # Neural Networks
      self.model = ChessNN()
      self.target_model = ChessNN()
      self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

      # Initialize target network
      self.update_target_model()

  def update_target_model(self):
      """Update target network weights"""
      self.target_model.load_state_dict(self.model.state_dict())

  def remember(self, state, action, reward, next_state, done):
      """Store experience in replay memory"""
      self.memory.append(Experience(state, action, reward, next_state, done))

  def act(self, state, legal_moves_mask):
      """Choose action using epsilon-greedy policy"""
      if random.random() < self.epsilon:
          legal_moves = np.where(legal_moves_mask == 1)[0]
          return np.random.choice(legal_moves) if len(legal_moves) > 0 else 0

      state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
      with torch.no_grad():
          q_values = self.model(state_tensor).cpu().numpy().squeeze()
          # Mask illegal moves
          q_values[legal_moves_mask == 0] = -np.inf
          return np.argmax(q_values)

  def replay(self, batch_size):
      """Train on batch of experiences"""
      if len(self.memory) < batch_size:
          return

      batch = random.sample(self.memory, batch_size)

      # Prepare batch tensors
      states = torch.FloatTensor([e.state for e in batch]).to(device)
      actions = torch.LongTensor([e.action for e in batch]).to(device)
      rewards = torch.FloatTensor([e.reward for e in batch]).to(device)
      next_states = torch.FloatTensor([e.next_state for e in batch]).to(device)
      dones = torch.FloatTensor([e.done for e in batch]).to(device)

      # Current Q-values
      current_q_values = self.model(states).gather(1, actions.unsqueeze(1))

      # Next Q-values from target model
      with torch.no_grad():
          next_q_values = self.target_model(next_states).max(1)[0]
          target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

      # Compute loss and optimize
      loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)

      # Gradient descent
      self.optimizer.zero_grad()
      loss.backward()
      # Clip gradients to prevent exploding gradients
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
      self.optimizer.step()

      # Update epsilon
      self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

      return loss.item()
