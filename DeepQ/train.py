import torch
from Chess_agent import DeepQLearningAgent
from Chess_env import ChessEnvironment



# Define experience tuple structure

def train_agent(episodes=1000, render_every=50, save_every=100):
  """Main training loop"""
  env = ChessEnvironment()
  state_size = (12, 8, 8)
  action_size = 4672
  agent = DeepQLearningAgent(state_size, action_size)

  # Training statistics
  best_reward = float('-inf')
  episode_rewards = []
  episode_lengths = []
  losses = []

  try:
      for episode in range(episodes):
          state = env.reset()
          total_reward = 0
          done = False
          moves = 0
          episode_loss = 0

          while not done and moves < 100:  # Limit moves to prevent infinite games
              if episode % render_every == 0:
                  env.render()

              # Get action
              legal_moves_mask = env.get_legal_moves_mask()
              action = agent.act(state, legal_moves_mask)

              # Take action
              next_state, reward, done = env.step(action)

              # Store experience
              agent.remember(state, action, reward, next_state, done)

              # Train
              if len(agent.memory) >= agent.batch_size:
                  loss = agent.replay(agent.batch_size)
                  episode_loss += loss if loss is not None else 0

              state = next_state
              total_reward += reward
              moves += 1

          # Update target network periodically
          if episode % agent.target_update_frequency == 0:
              agent.update_target_model()

          # Update statistics
          episode_rewards.append(total_reward)
          episode_lengths.append(moves)
          losses.append(episode_loss / moves if moves > 0 else 0)

          # Update best reward
          if total_reward > best_reward:
              best_reward = total_reward
              # Save best model
              torch.save({
                  'episode': episode,
                  'model_state_dict': agent.model.state_dict(),
                  'optimizer_state_dict': agent.optimizer.state_dict(),
                  'reward': best_reward,
              }, 'best_model.pth')

          # Save checkpoint
          if episode % save_every == 0:
              torch.save({
                  'episode': episode,
                  'model_state_dict': agent.model.state_dict(),
                  'optimizer_state_dict': agent.optimizer.state_dict(),
                  'reward': total_reward,
              }, f'checkpoint_episode_{episode}.pth')

          # Print progress
          print(f"Episode: {episode}")
          print(f"Total Reward: {total_reward:.2f}")
          print(f"Best Reward: {best_reward:.2f}")
          print(f"Epsilon: {agent.epsilon:.2f}")
          print(f"Moves: {moves}")
          print(f"Average Loss: {losses[-1]:.4f}")
          print("-" * 50)

  except KeyboardInterrupt:
      print("\nTraining interrupted by user")
  finally:
      # Save final model
      torch.save({
          'episode': episode,
          'model_state_dict': agent.model.state_dict(),
          'optimizer_state_dict': agent.optimizer.state_dict(),
          'reward': total_reward,
      }, 'final_model.pth')

    
