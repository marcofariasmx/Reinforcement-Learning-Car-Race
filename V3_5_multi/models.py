import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np
import random
import os
import time
import copy
import threading
from collections import deque

from config import (device, MEMORY_SIZE, BATCH_SIZE, GAMMA, LEARNING_RATE,
                    PPO_EPOCHS, PPO_EPSILON, data_lock, metrics_lock, metrics_data,
                    USE_MIXED_PRECISION, USE_PIN_MEMORY, USE_GPU_FOR_INFERENCE,
                    USE_ASYNC_SAVE, save_queue)


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


# Asynchronous model saving function to prevent training freezes
def async_save_model(save_dict, filename):
    try:
        torch.save(save_dict, filename)
        print(f"Model saved to {filename}")
    except Exception as e:
        print(f"Error saving model: {e}")


# Background thread to handle model saving
def save_model_worker():
    while True:
        try:
            save_info = save_queue.get()
            if save_info is None:  # Shutdown signal
                break

            save_dict, filename = save_info
            async_save_model(save_dict, filename)
            save_queue.task_done()
        except Exception as e:
            print(f"Error in save model worker: {e}")


# Start the save model worker thread
save_thread = threading.Thread(target=save_model_worker, daemon=True)
save_thread.start()


# PPO Agent with enhanced features
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)

        # Add learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=100, gamma=0.9)

        # Add CPU version of network for faster inference if needed
        self.use_cpu_for_inference = device.type == 'cuda' and not USE_GPU_FOR_INFERENCE
        if self.use_cpu_for_inference:
            self.cpu_actor_critic = ActorCritic(state_dim, action_dim, max_action).cpu()
            # Initialize with same weights
            self.cpu_actor_critic.load_state_dict(self.actor_critic.state_dict())
            print("Using CPU model for inference to reduce transfer overhead")

        # Setup for mixed precision training
        self.use_mixed_precision = device.type == 'cuda' and USE_MIXED_PRECISION
        if self.use_mixed_precision:
            # FIX: Use the updated PyTorch syntax for GradScaler
            self.scaler = torch.amp.GradScaler('cuda')
            print("Using mixed precision training for better GPU performance")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        # Use more efficient data storage
        self.memory = []
        self.memory_ptr = 0
        self.memory_size = MEMORY_SIZE
        self.memory_full = False
        self.trajectory = []

        # Track statistics for adaptive parameters
        self.episode_count = 0
        self.cumulative_rewards = []

        # Prepare tensors for batched operations
        self.use_pin_memory = USE_PIN_MEMORY and device.type == 'cuda'
        print(f"Using pin_memory: {self.use_pin_memory}")

        # Performance tracking
        self.update_times = deque(maxlen=100)
        self.transfer_times = deque(maxlen=100)

    def select_action(self, state, evaluation=False):
        """Optimized action selection to reduce CPU-GPU transfers"""
        start_time = time.time()

        # For evaluation or when using CPU for inference, avoid transfers
        if evaluation or self.use_cpu_for_inference:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)

                if self.use_cpu_for_inference:
                    # If CPU inference is enabled, use CPU model
                    self.cpu_actor_critic.eval()
                    if evaluation:
                        action = self.cpu_actor_critic.get_action(state_tensor, evaluation=True)
                        result = action.numpy().flatten()
                    else:
                        action, log_prob = self.cpu_actor_critic.get_action(state_tensor)
                        log_prob_scalar = float(log_prob.item())
                        result = action.numpy().flatten(), log_prob_scalar
                    self.cpu_actor_critic.train()
                else:
                    # Use GPU but optimize for evaluation
                    state_tensor = state_tensor.to(device)
                    self.actor_critic.eval()
                    action = self.actor_critic.get_action(state_tensor, evaluation=True)
                    result = action.cpu().numpy().flatten()
                    self.actor_critic.train()
        else:
            # Standard GPU path
            with torch.no_grad():
                self.actor_critic.eval()
                state_tensor = torch.FloatTensor(state).to(device).unsqueeze(0)
                action, log_prob = self.actor_critic.get_action(state_tensor)
                log_prob_scalar = float(log_prob.cpu().item())
                result = action.cpu().numpy().flatten(), log_prob_scalar
                self.actor_critic.train()

        # Track transfer time
        transfer_time = time.time() - start_time
        self.transfer_times.append(transfer_time)

        # Update metrics occasionally (not every call to reduce overhead)
        if random.random() < 0.01:  # Update metrics ~1% of the time
            with metrics_lock:
                metrics_data['transfer_time'] = np.mean(self.transfer_times)

        return result

    def add_to_trajectory(self, state, action, log_prob, reward, next_state, done):
        """Add transition to current trajectory buffer"""
        self.trajectory.append((state, action, log_prob, reward, next_state, done))

    def add_to_memory(self, item):
        """More efficient memory management using circular buffer without deque"""
        if len(self.memory) < self.memory_size:
            self.memory.append(item)
        else:
            self.memory[self.memory_ptr] = item
            self.memory_ptr = (self.memory_ptr + 1) % self.memory_size
            self.memory_full = True

    def end_trajectory(self, last_value=0):
        """Process the trajectory and add it to memory with optimized GPU usage"""
        if not self.trajectory:
            return  # Nothing to process

        # Process the trajectory and add it to memory
        states, actions, log_probs, rewards, next_states, dones = zip(*self.trajectory)

        # Calculate returns and advantages
        returns = []
        advantages = []
        value = last_value

        # Process trajectory in reverse for bootstrapping
        for t in reversed(range(len(rewards))):
            next_value = value if t == len(rewards) - 1 else returns[0]
            return_t = rewards[t] + GAMMA * next_value * (1 - dones[t])
            returns.insert(0, return_t)

            # Get value estimate for current state
            with torch.no_grad():
                if self.use_cpu_for_inference:
                    # Use CPU model for inference
                    state_t = torch.FloatTensor(states[t]).unsqueeze(0)
                    self.cpu_actor_critic.eval()
                    _, _, value_t = self.cpu_actor_critic(state_t)
                    self.cpu_actor_critic.train()
                    value_t = value_t.numpy()
                else:
                    # Use GPU model
                    state_t = torch.FloatTensor(states[t]).to(device).unsqueeze(0)
                    self.actor_critic.eval()
                    _, _, value_t = self.actor_critic(state_t)
                    self.actor_critic.train()
                    value_t = value_t.cpu().numpy()

                if isinstance(value_t, np.ndarray):
                    value_t = float(value_t.item())

            advantage_t = return_t - value_t
            advantages.insert(0, advantage_t)

        # Normalize advantages
        advantages = np.array(advantages)
        if len(advantages) > 1:  # Only normalize if we have more than one element
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Add to memory
        for t in range(len(self.trajectory)):
            self.add_to_memory((
                states[t],
                actions[t],
                log_probs[t],
                returns[t],
                advantages[t]
            ))

        # Clear trajectory
        self.trajectory = []

    def update(self):
        """Optimized PPO update using mixed precision and reduced CPU-GPU transfers"""
        # Skip update if not enough data
        if len(self.memory) < BATCH_SIZE:
            return 0, 0, 0, 0  # Return zeros when no update happens

        update_start = time.time()

        # Sample from memory
        indices = random.sample(range(len(self.memory)), BATCH_SIZE)
        batch = [self.memory[idx] for idx in indices]

        # Extract components efficiently
        states = np.array([item[0] for item in batch], dtype=np.float32)
        actions = np.array([item[1] for item in batch], dtype=np.float32)
        old_log_probs = np.array([item[2] for item in batch], dtype=np.float32).reshape(-1, 1)
        returns = np.array([item[3] for item in batch], dtype=np.float32).reshape(-1, 1)
        advantages = np.array([item[4] for item in batch], dtype=np.float32).reshape(-1, 1)

        # Convert to tensors with optimized transfer
        if self.use_pin_memory:
            # Pin memory for faster transfers
            states_tensor = torch.from_numpy(states).pin_memory().to(device, non_blocking=True)
            actions_tensor = torch.from_numpy(actions).pin_memory().to(device, non_blocking=True)
            old_log_probs_tensor = torch.from_numpy(old_log_probs).pin_memory().to(device, non_blocking=True)
            returns_tensor = torch.from_numpy(returns).pin_memory().to(device, non_blocking=True)
            advantages_tensor = torch.from_numpy(advantages).pin_memory().to(device, non_blocking=True)
        else:
            # Standard transfer
            states_tensor = torch.FloatTensor(states).to(device)
            actions_tensor = torch.FloatTensor(actions).to(device)
            old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(device)
            returns_tensor = torch.FloatTensor(returns).to(device)
            advantages_tensor = torch.FloatTensor(advantages).to(device)

        # Normalize advantages (improves training stability)
        advantages_tensor = (advantages_tensor - advantages_tensor.mean()) / (advantages_tensor.std() + 1e-8)

        # Track losses
        total_loss = 0
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_loss_total = 0

        # Calculate adaptive entropy coefficient
        # Start with higher entropy for exploration, gradually reduce
        entropy_coef = max(0.01, 0.1 * (0.995 ** self.episode_count))

        # Multiple epochs of PPO update - use mixed precision if available
        for _ in range(PPO_EPOCHS):
            if self.use_mixed_precision:
                # Mixed precision training path - with fixed PyTorch syntax
                with torch.amp.autocast('cuda'):
                    # Evaluate actions and calculate ratio
                    log_probs, values, entropy = self.actor_critic.evaluate(states_tensor, actions_tensor)

                    # Calculate ratio (importance sampling)
                    ratios = torch.exp(log_probs - old_log_probs_tensor)

                    # Calculate surrogate losses
                    surr1 = ratios * advantages_tensor
                    surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages_tensor

                    # Calculate actor and critic losses
                    actor_loss = -torch.min(surr1, surr2).mean()
                    critic_loss = F.mse_loss(values, returns_tensor)
                    entropy_loss = -entropy_coef * entropy  # Adaptive entropy coefficient

                    # Total loss
                    loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Update with mixed precision
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training path
                # Evaluate actions and calculate ratio
                log_probs, values, entropy = self.actor_critic.evaluate(states_tensor, actions_tensor)

                # Calculate ratio (importance sampling)
                ratios = torch.exp(log_probs - old_log_probs_tensor)

                # Calculate surrogate losses
                surr1 = ratios * advantages_tensor
                surr2 = torch.clamp(ratios, 1 - PPO_EPSILON, 1 + PPO_EPSILON) * advantages_tensor

                # Calculate actor and critic losses
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(values, returns_tensor)
                entropy_loss = -entropy_coef * entropy  # Adaptive entropy coefficient

                # Total loss
                loss = actor_loss + 0.5 * critic_loss + entropy_loss

                # Update network
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
                self.optimizer.step()

            # Track losses (detached from computation graph)
            total_loss += loss.item()
            actor_loss_total += actor_loss.item()
            critic_loss_total += critic_loss.item()
            entropy_loss_total += entropy_loss.item()

        # Step the learning rate scheduler
        self.scheduler.step()

        # Update CPU model if used for inference
        if self.use_cpu_for_inference and random.random() < 0.1:  # Only sync occasionally
            self.cpu_actor_critic.load_state_dict(self.actor_critic.state_dict())

        # Track update time
        update_time = time.time() - update_start
        self.update_times.append(update_time)

        # Update metrics
        with metrics_lock:
            metrics_data['nn_update_time'] = np.mean(self.update_times)

        # Return average losses
        return (
            total_loss / PPO_EPOCHS,
            actor_loss_total / PPO_EPOCHS,
            critic_loss_total / PPO_EPOCHS,
            entropy_loss_total / PPO_EPOCHS
        )

    def end_episode(self, episode_reward):
        """Update agent state at the end of an episode"""
        self.episode_count += 1
        self.cumulative_rewards.append(episode_reward)

        # Store current learning rate and entropy coefficient in metrics
        current_lr = self.scheduler.get_last_lr()[0]
        entropy_coef = max(0.01, 0.1 * (0.995 ** self.episode_count))

        # Update metrics with minimal lock time
        with metrics_lock:
            metrics_data['learning_rate'] = current_lr
            metrics_data['entropy_coef'] = entropy_coef

            # Update GPU metrics if available
            if torch.cuda.is_available():
                try:
                    # Some NVIDIA GPUs support temperature querying
                    metrics_data['gpu_util'] = torch.cuda.utilization()
                except:
                    pass

    def save(self, filename="ppo_car_model.pth"):
        """Save model with optimized async approach to prevent freezing and ensure training time is saved"""
        # Calculate current session training time
        current_session_time = time.time() - metrics_data['start_time']

        # Get current total training time first
        with metrics_lock:
            total_training_time = metrics_data['total_training_time'] + current_session_time

            # Important: Update the metrics_data with the new total
            metrics_data['total_training_time'] = total_training_time

            # Reset start time for next session
            metrics_data['start_time'] = time.time()

            # Get current training metrics to save
            current_metrics = {
                'episode_rewards': metrics_data['episode_rewards'].copy() if metrics_data['episode_rewards'] else [],
                'episode_lengths': metrics_data['episode_lengths'].copy() if metrics_data['episode_lengths'] else [],
                'episode_laps': metrics_data['episode_laps'].copy() if metrics_data['episode_laps'] else [],
                'avg_rewards': metrics_data['avg_rewards'].copy() if metrics_data['avg_rewards'] else [],
                'total_training_time': total_training_time,  # Include this in both places for backwards compatibility
                'loss_history': metrics_data.get('loss_history', [])[-1000:] if metrics_data.get('loss_history',
                                                                                                 []) else [],
                'actor_loss_history': metrics_data.get('actor_loss_history', [])[-1000:] if metrics_data.get(
                    'actor_loss_history', []) else [],
                'critic_loss_history': metrics_data.get('critic_loss_history', [])[-1000:] if metrics_data.get(
                    'critic_loss_history', []) else []
            }

        # Create a save dictionary with deep copies to avoid reference issues
        save_dict = {
            'actor_critic_state_dict': copy.deepcopy(self.actor_critic.state_dict()),
            'optimizer_state_dict': copy.deepcopy(self.optimizer.state_dict()),
            'scheduler_state_dict': copy.deepcopy(self.scheduler.state_dict()),
            'episode_count': self.episode_count,
            'cumulative_rewards': self.cumulative_rewards.copy(),
            'training_metrics': current_metrics,
            'total_training_time': total_training_time,  # Save it at the top level too for robustness
            'total_episodes_trained': len(metrics_data['episode_rewards']) if metrics_data['episode_rewards'] else 0,
            'save_time': time.time()
        }

        print(f"Saving model with total training time: {self.format_time(total_training_time)}")

        if USE_ASYNC_SAVE:
            try:
                # Queue the save operation to be handled by the background thread
                save_queue.put((save_dict, filename))
                print(f"Model queued for saving to {filename}")
            except Exception as e:
                print(f"Error queueing model save: {e}")
                # Fall back to synchronous save if queueing fails
                torch.save(save_dict, filename, weights_only=False)
                print(f"Model saved synchronously to {filename}")
        else:
            # Synchronous save
            torch.save(save_dict, filename, weights_only=False)
            print(f"Model saved synchronously to {filename}")

        # Format the time for display
        total_time_str = self.format_time(total_training_time)
        print(
            f"Model save requested for {filename} (Episodes trained: {self.episode_count}, Total training time: {total_time_str})")

    def load(self, filename="ppo_car_model.pth"):
        """Load model with improved error handling and training time preservation"""
        if os.path.isfile(filename):
            try:
                # Try with weights_only=False first since we know this is likely to work with older models
                print(f"Loading model from {filename} with compatibility mode...")
                checkpoint = torch.load(filename, map_location=device, weights_only=False)

                # Debug: print all keys in the checkpoint
                print(f"Checkpoint contains keys: {list(checkpoint.keys())}")
                if 'training_metrics' in checkpoint:
                    print(f"Training metrics contains keys: {list(checkpoint['training_metrics'].keys())}")

                # Load model state dict
                self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

                # Update CPU model if used
                if self.use_cpu_for_inference:
                    self.cpu_actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])

                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load scheduler if available
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

                # Load episode count
                if 'episode_count' in checkpoint:
                    self.episode_count = checkpoint['episode_count']

                # Load rewards history
                if 'cumulative_rewards' in checkpoint:
                    self.cumulative_rewards = checkpoint['cumulative_rewards']

                # IMPORTANT: Load total training time and ensure it's preserved
                loaded_total_time = 0

                # Load from different possible locations in the checkpoint
                if 'training_metrics' in checkpoint and 'total_training_time' in checkpoint['training_metrics']:
                    loaded_total_time = checkpoint['training_metrics']['total_training_time']
                    print(f"Loaded total training time from metrics: {self.format_time(loaded_total_time)}")
                elif 'total_training_time' in checkpoint:
                    loaded_total_time = checkpoint['total_training_time']
                    print(f"Loaded total training time directly: {self.format_time(loaded_total_time)}")

                # Load saved metrics if available
                with metrics_lock:
                    if 'training_metrics' in checkpoint:
                        if 'episode_rewards' in checkpoint['training_metrics']:
                            metrics_data['episode_rewards'] = checkpoint['training_metrics']['episode_rewards']
                        if 'episode_lengths' in checkpoint['training_metrics']:
                            metrics_data['episode_lengths'] = checkpoint['training_metrics']['episode_lengths']
                        if 'episode_laps' in checkpoint['training_metrics']:
                            metrics_data['episode_laps'] = checkpoint['training_metrics']['episode_laps']
                        if 'avg_rewards' in checkpoint['training_metrics']:
                            metrics_data['avg_rewards'] = checkpoint['training_metrics']['avg_rewards']

                        # Load loss history if available
                        if 'loss_history' in checkpoint['training_metrics']:
                            metrics_data['loss_history'] = checkpoint['training_metrics']['loss_history']
                        if 'actor_loss_history' in checkpoint['training_metrics']:
                            metrics_data['actor_loss_history'] = checkpoint['training_metrics']['actor_loss_history']
                        if 'critic_loss_history' in checkpoint['training_metrics']:
                            metrics_data['critic_loss_history'] = checkpoint['training_metrics']['critic_loss_history']

                    # VERY IMPORTANT: Explicitly set the total training time from what we loaded
                    if loaded_total_time > 0:
                        metrics_data['total_training_time'] = loaded_total_time
                        print(f"Successfully restored total training time: {self.format_time(loaded_total_time)}")
                    else:
                        # If no training time found, estimate it based on episode count
                        estimated_time = self.episode_count * 10  # Assume 10 seconds per episode as a rough estimate
                        metrics_data['total_training_time'] = estimated_time
                        print(f"No training time found, using estimate: {self.format_time(estimated_time)}")

                # Display info about the loaded model
                total_episodes = checkpoint.get('total_episodes_trained', self.episode_count)
                save_time = checkpoint.get('save_time', None)
                save_time_str = time.strftime("%Y-%m-%d %H:%M:%S",
                                              time.localtime(save_time)) if save_time else "unknown"

                # Get total training time
                total_training_time = metrics_data['total_training_time']
                total_time_str = self.format_time(total_training_time)

                print(f"Model loaded from {filename}")
                print(f"Episodes trained: {total_episodes}")
                print(f"Total training time: {total_time_str}")
                print(f"Last saved: {save_time_str}")

                # Reset start time to avoid double-counting
                metrics_data['start_time'] = time.time()

                return True

            except Exception as e:
                print(f"Error loading model: {e}")
                print("Loading failed, starting with a new model")
                return False
        else:
            print(f"No model file found at {filename}, starting with a new model")
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