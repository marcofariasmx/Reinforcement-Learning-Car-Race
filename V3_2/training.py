import torch
import numpy as np
import time
import random
from collections import deque

from config import data_lock, frame_data, metrics_data, running, MAX_EPISODES, BATCH_SIZE, device
from models import PPOAgent
from environment import CarEnv


# Training thread with enhanced exploration and learning
def training_thread():
    global frame_data, running, metrics_data

    # Create environment and agent
    env = CarEnv()
    agent = PPOAgent(env.state_dim, env.action_dim, 1.0)  # Actions range from -1 to 1

    # Load model if exists
    model_loaded = agent.load()

    # Experience replay warmup - generate some initial experiences
    if not model_loaded and len(agent.memory) < BATCH_SIZE:
        print("Performing experience replay warmup...")
        warmup_env = CarEnv()
        warmup_state = warmup_env.reset()

        for _ in range(BATCH_SIZE):
            # Take random actions during warmup
            action = np.random.uniform(-1, 1, size=env.action_dim)
            # For PPO, we need a fake log_prob with consistent type and shape
            log_prob = 0.0  # Simple scalar float

            next_state, reward, done, _ = warmup_env.step(action)

            # Store in trajectory
            agent.add_to_trajectory(warmup_state, action, log_prob, reward, next_state, done)

            warmup_state = next_state
            if done:
                warmup_state = warmup_env.reset()

        # End the warmup trajectory
        agent.end_trajectory()
        print(f"Warmup complete. Memory size: {len(agent.memory)}")

    # Training parameters
    episode_rewards = []
    episode_lengths = []
    episode_laps = []
    recent_rewards = deque(maxlen=100)  # Track recent rewards for plotting

    for episode in range(MAX_EPISODES):
        if not running:
            break

        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        # Record start time for this episode
        episode_start_time = time.time()

        # Tracking variables for this episode
        episode_actions = []
        episode_step_rewards = []

        # Add episode noise for exploration - annealing over time
        exploration_noise = max(0.1, 1.0 * (0.98 ** episode))  # Decay from 1.0 to 0.1

        while not done:
            # Select action with exploration noise that decreases over time
            if np.random.random() < exploration_noise and episode < 100:
                # Sometimes take random actions early in training
                action = np.random.uniform(-1, 1, size=env.action_dim)
                log_prob = 0.0  # Simple scalar for random actions
            else:
                # Regular action selection
                action, log_prob = agent.select_action(state)

            # Execute action
            next_state, reward, done, info = env.step(action)

            # Track actions and rewards for this step
            episode_actions.append(action.tolist())
            episode_step_rewards.append(reward)

            # Store in agent's trajectory
            agent.add_to_trajectory(state, action, log_prob, reward, next_state, done)

            # Update state and episode reward
            state = next_state
            episode_reward += reward
            steps += 1

            # Update visualization data
            with data_lock:
                frame_data['car'] = env.car
                frame_data['walls'] = env.walls
                frame_data['checkpoints'] = env.checkpoints
                frame_data['reward'] = reward
                frame_data['total_reward'] = episode_reward
                frame_data['laps'] = env.car.laps_completed
                frame_data['episode'] = episode
                frame_data['info'] = info

            # Update agent
            if len(agent.trajectory) >= BATCH_SIZE or done:
                # If done, use 0 as the last value, otherwise use the value of the next state
                if done:
                    last_value = 0
                else:
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(next_state).to(device).unsqueeze(0)
                        agent.actor_critic.eval()  # Set to eval mode for inference
                        _, _, last_value = agent.actor_critic(next_state_tensor)
                        agent.actor_critic.train()  # Back to train mode
                        last_value = last_value.cpu().data.numpy()
                        if isinstance(last_value, np.ndarray):
                            last_value = float(last_value.item())

                agent.end_trajectory(last_value)
                total_loss, actor_loss, critic_loss, entropy_loss = agent.update()

                # Update metrics
                with data_lock:
                    metrics_data['last_loss'] = total_loss
                    metrics_data['actor_loss'] = actor_loss
                    metrics_data['critic_loss'] = critic_loss
                    metrics_data['entropy_loss'] = entropy_loss
                    metrics_data['updates_performed'] += 1
                    metrics_data['memory_usage'] = len(agent.memory)

                    # Track recent actions and rewards
                    metrics_data['recent_actions'] = episode_actions[-20:] if episode_actions else []
                    metrics_data['recent_rewards'] = episode_step_rewards[-100:] if episode_step_rewards else []
                    metrics_data['training_step'] += 1

        # End of episode - update agent's episode tracking
        agent.end_episode(episode_reward)

        # End of episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_laps.append(env.car.laps_completed)
        recent_rewards.append(episode_reward)

        # Calculate episode time and update time metrics
        episode_time = time.time() - episode_start_time
        metrics_data['episode_times'].append(episode_time)

        # Calculate average episode time and estimate completion time
        avg_episode_time = np.mean(metrics_data['episode_times'][-100:]) if metrics_data['episode_times'] else 0
        remaining_episodes = MAX_EPISODES - episode - 1
        estimated_completion_time = avg_episode_time * remaining_episodes

        # Check if car crashed
        if env.car.crashed:
            with data_lock:
                metrics_data['crash_locations'].append((env.car.x, env.car.y, episode))
                # Keep only most recent 50 crash locations
                if len(metrics_data['crash_locations']) > 50:
                    metrics_data['crash_locations'].pop(0)

        # Update metrics data
        with data_lock:
            metrics_data['episode_rewards'] = episode_rewards
            metrics_data['episode_lengths'] = episode_lengths
            metrics_data['episode_laps'] = episode_laps
            metrics_data['avg_rewards'].append(np.mean(recent_rewards))
            metrics_data['avg_episode_time'] = avg_episode_time
            metrics_data['estimated_completion_time'] = estimated_completion_time
            metrics_data['estimated_runtime'] = time.time() - metrics_data['start_time']

        # Adaptive reporting - less frequent as training progresses
        if episode < 100 or episode % 10 == 0:
            current_lr = agent.scheduler.get_last_lr()[0]
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {np.mean(recent_rewards):.2f}, "
                  f"Laps = {env.car.laps_completed}, Steps = {steps}, LR = {current_lr:.6f}")

        # Save model periodically
        if episode > 0 and episode % 10 == 0:
            agent.save()

    # Save final model
    agent.save()