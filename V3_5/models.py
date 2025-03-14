import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import os
import time
from collections import deque

from config import (device, MEMORY_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE,
                    PPO_EPOCHS, PPO_EPSILON, data_lock, metrics_data)


# Enhanced Neural Network for the PPO Agent with Layer Normalization
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()

        # Improved feature extractor with layer normalization
        self.feature_layer1 = nn.Linear(state_dim, 256)
        self.feature_ln1 = nn.LayerNorm(256)
        self.feature_layer2 = nn.Linear(256, 256)
        self.feature_ln2 = nn.LayerNorm(256)

        # Actor (policy) layers
        self.actor_layer = nn.Linear(256, 128)
        self.actor_ln = nn.LayerNorm(128)
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std_layer = nn.Linear(128, action_dim)

        # Critic (value) layers - separate branch
        self.critic_layer = nn.Linear(256, 128)
        self.critic_ln = nn.LayerNorm(128)
        self.value_layer = nn.Linear(128, 1)

        self.max_action = max_action

    def forward(self, state):
        # Handle single sample case (during action selection)
        if state.dim() == 1:
            state = state.unsqueeze(0)
            single_sample = True
        else:
            single_sample = False

        # Shared feature extraction
        x = F.relu(self.feature_ln1(self.feature_layer1(state)))
        x = F.relu(self.feature_ln2(self.feature_layer2(x)))

        # Actor branch
        ax = F.relu(self.actor_ln(self.actor_layer(x)))
        action_mean = torch.tanh(self.mean_layer(ax)) * self.max_action

        # Log standard deviation with clamping for stability
        action_log_std = self.log_std_layer(ax)
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        # Critic branch
        cx = F.relu(self.critic_ln(self.critic_layer(x)))
        value = self.value_layer(cx)

        # Handle single sample case
        if single_sample and value.dim() > 1:
            value = value.squeeze(0)
            action_mean = action_mean.squeeze(0)
            action_std = action_std.squeeze(0)

        return action_mean, action_std, value

    def get_action(self, state, evaluation=False):
        action_mean, action_std, _ = self.forward(state)

        if evaluation:
            return action_mean

        dist = Normal(action_mean, action_std)
        action = dist.sample()
        action = torch.clamp(action, -self.max_action, self.max_action)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)

        return action, log_prob

    def evaluate(self, state, action):
        action_mean, action_std, value = self.forward(state)

        dist = Normal(action_mean, action_std)
        log_prob = dist.log_prob(action).sum(dim=-1, keepdim=True)
        entropy = dist.entropy().mean()

        return log_prob, value, entropy


# PPO Agent with enhanced features
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)

        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.trajectory = []

        # Track statistics for adaptive parameters
        self.episode_count = 0
        self.cumulative_rewards = []

    def select_action(self, state, evaluation=False):
        self.actor_critic.eval()  # Set to eval mode for inference

        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        with torch.no_grad():
            if evaluation:
                action = self.actor_critic.get_action(state, evaluation=True)
                result = action.cpu().data.numpy().flatten()
            else:
                action, log_prob = self.actor_critic.get_action(state)
                # Ensure consistent shape for log_prob - convert to scalar value
                log_prob_numpy = log_prob.cpu().data.numpy()
                # Reshape to always be scalar
                if log_prob_numpy.shape != (1, 1):
                    log_prob_numpy = log_prob_numpy.reshape(1, 1)
                log_prob_scalar = float(log_prob_numpy[0, 0])  # Convert to scalar
                result = action.cpu().data.numpy().flatten(), log_prob_scalar

        self.actor_critic.train()  # Back to train mode
        return result

    def add_to_trajectory(self, state, action, log_prob, reward, next_state, done):
        self.trajectory.append((state, action, log_prob, reward, next_state, done))

    def end_trajectory(self, last_value=0):
        # Process the trajectory and add it to memory
        states, actions, log_probs, rewards, next_states, dones = zip(*self.trajectory)

        # Calculate returns and advantages
        returns = []
        advantages = []
        value = last_value

        for t in reversed(range(len(rewards))):
            next_value = value if t == len(rewards) - 1 else returns[0]
            return_t = rewards[t] + GAMMA * next_value * (1 - dones[t])
            returns.insert(0, return_t)

            with torch.no_grad():
                state_t = torch.FloatTensor(states[t]).to(device).unsqueeze(0)
                # Set to eval mode for inference
                self.actor_critic.eval()
                _, _, value_t = self.actor_critic(state_t)
                self.actor_critic.train()  # Back to train mode

                value_t = value_t.cpu().data.numpy()
                if isinstance(value_t, np.ndarray):
                    value_t = float(value_t.item())

            advantage_t = return_t - value_t
            advantages.insert(0, advantage_t)

        # Normalize advantages
        advantages = np.array(advantages)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Add to memory
        for t in range(len(self.trajectory)):
            self.memory.append((
                states[t],
                actions[t],
                log_probs[t],
                returns[t],
                advantages[t]
            ))

        # Clear trajectory
        self.trajectory = []

    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return 0, 0, 0, 0  # Return losses as zeros when no update happens

        # Sample from memory
        minibatch = random.sample(self.memory, BATCH_SIZE)
        states, actions, old_log_probs, returns, advantages = zip(*minibatch)

        # Convert lists to numpy arrays first, then to tensors (more efficient)
        states = torch.FloatTensor(np.array(states)).to(device)
        actions = torch.FloatTensor(np.array(actions)).to(device)

        # Process old_log_probs - ensure they're all scalar floats
        old_log_probs = torch.FloatTensor(np.array(old_log_probs, dtype=np.float32)).to(device).unsqueeze(1)
        returns = torch.FloatTensor(np.array(returns, dtype=np.float32)).to(device).unsqueeze(1)
        advantages = torch.FloatTensor(np.array(advantages, dtype=np.float32)).to(device).unsqueeze(1)

        # Normalize advantages (improves training stability)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Track losses
        total_loss = 0
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_loss_total = 0

        # Calculate adaptive entropy coefficient
        # Start with higher entropy for exploration, gradually reduce
        entropy_coef = max(0.01, 0.1 * (0.995 ** self.episode_count))

        # Multiple epochs of PPO update
        for _ in range(PPO_EPOCHS):
            # Evaluate actions and calculate ratio
            log_probs, values, entropy = self.actor_critic.evaluate(states, actions)

            # Calculate ratio (importance sampling)
            ratios = torch.exp(log_probs - old_log_probs)

            # Calculate surrogate losses
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages

            # Calculate actor and critic losses
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = -entropy_coef * entropy  # Adaptive entropy coefficient

            # Total loss
            loss = actor_loss + 0.5 * critic_loss + entropy_loss

            # Update network
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()

            # Track losses
            total_loss += loss.item()
            actor_loss_total += actor_loss.item()
            critic_loss_total += critic_loss.item()
            entropy_loss_total += entropy_loss.item()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Return average losses
        return (
            total_loss / PPO_EPOCHS,
            actor_loss_total / PPO_EPOCHS,
            critic_loss_total / PPO_EPOCHS,
            entropy_loss_total / PPO_EPOCHS
        )

    def end_episode(self, episode_reward):
        self.episode_count += 1
        self.cumulative_rewards.append(episode_reward)

        # Store current learning rate in metrics
        current_lr = self.scheduler.get_last_lr()[0]
        entropy_coef = max(0.01, 0.1 * (0.995 ** self.episode_count))
        with data_lock:
            metrics_data['learning_rate'] = current_lr
            metrics_data['entropy_coef'] = entropy_coef

    def save(self, filename="ppo_car_model_V2.2.pth"):
        # Calculate current session training time
        current_session_time = time.time() - metrics_data['start_time']

        # Update total training time
        with data_lock:
            metrics_data['total_training_time'] += current_session_time
            total_training_time = metrics_data['total_training_time']

            # Reset start time for next session
            metrics_data['start_time'] = time.time()

            # Get current training metrics to save
            current_metrics = {
                'episode_rewards': metrics_data['episode_rewards'].copy() if metrics_data['episode_rewards'] else [],
                'episode_lengths': metrics_data['episode_lengths'].copy() if metrics_data['episode_lengths'] else [],
                'episode_laps': metrics_data['episode_laps'].copy() if metrics_data['episode_laps'] else [],
                'avg_rewards': metrics_data['avg_rewards'].copy() if metrics_data['avg_rewards'] else [],
                'total_training_time': total_training_time
            }

        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'episode_count': self.episode_count,
            'cumulative_rewards': self.cumulative_rewards,
            'training_metrics': current_metrics,
            'total_episodes_trained': len(metrics_data['episode_rewards']) if metrics_data['episode_rewards'] else 0,
            'save_time': time.time()
        }, filename)

        # Format the time for display (days, hours, minutes, seconds)
        total_time_str = self.format_time(total_training_time)
        print(
            f"Model saved to {filename} (Episodes trained: {self.episode_count}, Total training time: {total_time_str})")

    def load(self, filename="ppo_car_model_V2.2.pth"):
        if os.path.isfile(filename):
            try:
                # Try with default settings first
                checkpoint = torch.load(filename, map_location=device)
            except Exception as e:
                print(f"Error with default load settings: {e}")
                print("Attempting to load with weights_only=False for backwards compatibility...")
                # Use weights_only=False for backward compatibility with models saved in older PyTorch versions
                checkpoint = torch.load(filename, map_location=device, weights_only=False)

            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            if 'scheduler_state_dict' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if 'episode_count' in checkpoint:
                self.episode_count = checkpoint['episode_count']

            if 'cumulative_rewards' in checkpoint:
                self.cumulative_rewards = checkpoint['cumulative_rewards']

            # Load saved metrics if available
            if 'training_metrics' in checkpoint:
                with data_lock:
                    if 'episode_rewards' in checkpoint['training_metrics']:
                        metrics_data['episode_rewards'] = checkpoint['training_metrics']['episode_rewards']
                    if 'episode_lengths' in checkpoint['training_metrics']:
                        metrics_data['episode_lengths'] = checkpoint['training_metrics']['episode_lengths']
                    if 'episode_laps' in checkpoint['training_metrics']:
                        metrics_data['episode_laps'] = checkpoint['training_metrics']['episode_laps']
                    if 'avg_rewards' in checkpoint['training_metrics']:
                        metrics_data['avg_rewards'] = checkpoint['training_metrics']['avg_rewards']
                    if 'total_training_time' in checkpoint['training_metrics']:
                        metrics_data['total_training_time'] = checkpoint['training_metrics']['total_training_time']

            # Display info about the loaded model
            total_episodes = checkpoint.get('total_episodes_trained', self.episode_count)
            save_time = checkpoint.get('save_time', None)
            save_time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(save_time)) if save_time else "unknown"

            # Get total training time
            total_training_time = metrics_data['total_training_time']
            total_time_str = self.format_time(total_training_time)

            print(f"Model loaded from {filename}")
            print(f"Episodes trained: {total_episodes}")
            print(f"Total training time: {total_time_str}")
            print(f"Last saved: {save_time_str}")
            return True
        return False

    @staticmethod
    def format_time(seconds):
        """Format seconds into days, hours, minutes, seconds"""
        days, remainder = divmod(int(seconds), 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)

        time_parts = []
        if days > 0:
            time_parts.append(f"{days}d")
        if hours > 0 or days > 0:
            time_parts.append(f"{hours}h")
        if minutes > 0 or hours > 0 or days > 0:
            time_parts.append(f"{minutes}m")
        time_parts.append(f"{seconds}s")

        return " ".join(time_parts)