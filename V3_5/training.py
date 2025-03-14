import torch
import numpy as np
import time
import random
from collections import deque
import copy

from config import data_lock, frame_data, metrics_data, running, MAX_EPISODES, BATCH_SIZE, device, frame_buffer
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

    # Use the episode count from the loaded model
    start_episode = agent.episode_count if model_loaded else 0

    # Update metrics_data to store the starting episode number
    with data_lock:
        metrics_data['start_episode'] = start_episode

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

    # Timing variables for neural network updates
    nn_update_count = 0
    last_nn_update_time = time.time()
    avg_nn_update_time = 0.0

    # Start from the correct episode number
    for episode in range(start_episode, MAX_EPISODES):
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

            # Prepare visualization data - create a frame snapshot
            frame_snapshot = {
                'car': copy.copy(env.car),  # Make copies to avoid reference issues
                'walls': env.walls,  # Walls and checkpoints don't change often
                'checkpoints': env.checkpoints,
                'reward': reward,
                'total_reward': episode_reward,
                'laps': env.car.laps_completed,
                'episode': episode,
                'info': info.copy() if info else {}
            }

            # Try to update frame buffer without blocking - priority is to keep training fast
            try:
                # Put new frame in buffer, discarding old frames if full
                if frame_buffer.full():
                    try:
                        frame_buffer.get_nowait()  # Remove oldest frame
                    except:
                        pass
                frame_buffer.put_nowait(frame_snapshot)
            except:
                pass  # Never block training for visualization

            # Also update the traditional frame_data for backward compatibility
            with data_lock:
                frame_data.update(frame_snapshot)

            # Update agent - this is potentially expensive and may block for a while
            if len(agent.trajectory) >= BATCH_SIZE or done:
                # Start timing for neural network update
                nn_update_start = time.time()

                # Calculate last value before updating
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

                # End trajectory and perform update (potentially expensive operation)
                agent.end_trajectory(last_value)
                total_loss, actor_loss, critic_loss, entropy_loss = agent.update()

                # Record neural network update time
                nn_update_time = time.time() - nn_update_start
                nn_update_count += 1
                avg_nn_update_time = avg_nn_update_time * 0.9 + nn_update_time * 0.1  # Exponential moving avg

                # Save update time info for debugging
                time_since_last_update = time.time() - last_nn_update_time
                last_nn_update_time = time.time()

                # Prepare metrics data outside the lock
                local_metrics = {
                    'last_loss': total_loss,
                    'actor_loss': actor_loss,
                    'critic_loss': critic_loss,
                    'entropy_loss': entropy_loss,
                    'recent_actions': episode_actions[-20:] if episode_actions else [],
                    'recent_rewards': episode_step_rewards[-100:] if episode_step_rewards else [],
                    'memory_usage': len(agent.memory),
                    'updates_performed': metrics_data['updates_performed'] + 1,
                    'training_step': metrics_data['training_step'] + 1,
                    'last_nn_update_time': nn_update_time,
                    'avg_nn_update_time': avg_nn_update_time,
                    'time_between_updates': time_since_last_update
                }

                # Update metrics data with minimal lock time
                with data_lock:
                    metrics_data.update(local_metrics)

        # End of episode - update agent's episode tracking
        agent.end_episode(episode_reward)

        # End of episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_laps.append(env.car.laps_completed)
        recent_rewards.append(episode_reward)

        # Calculate episode time and update time metrics
        episode_time = time.time() - episode_start_time

        # Update episode metrics in one atomic operation to minimize lock time
        with data_lock:
            metrics_data['episode_times'].append(episode_time)
            metrics_data['episode_rewards'] = episode_rewards.copy()
            metrics_data['episode_lengths'] = episode_lengths.copy()
            metrics_data['episode_laps'] = episode_laps.copy()
            metrics_data['avg_rewards'].append(np.mean(recent_rewards) if recent_rewards else 0)

            # Calculate average episode time and estimate completion time
            avg_episode_time = np.mean(metrics_data['episode_times'][-100:]) if metrics_data['episode_times'] else 0
            remaining_episodes = MAX_EPISODES - episode - 1
            estimated_completion_time = avg_episode_time * remaining_episodes

            metrics_data['avg_episode_time'] = avg_episode_time
            metrics_data['estimated_completion_time'] = estimated_completion_time
            metrics_data['estimated_runtime'] = time.time() - metrics_data['start_time']

            # Check if car crashed
            if env.car.crashed:
                metrics_data['crash_locations'].append((env.car.x, env.car.y, episode))
                # Keep only most recent 50 crash locations
                if len(metrics_data['crash_locations']) > 50:
                    metrics_data['crash_locations'].pop(0)

        # Adaptive reporting - less frequent as training progresses
        if episode < 100 or episode % 10 == 0:
            current_lr = agent.scheduler.get_last_lr()[0]
            print(f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {np.mean(recent_rewards):.2f}, "
                  f"Laps = {env.car.laps_completed}, Steps = {steps}, LR = {current_lr:.6f}, "
                  f"NN Update Time = {avg_nn_update_time:.4f}s")

        # Save model periodically
        if episode > 0 and episode % 10 == 0:
            agent.save()

    # Save final model
    agent.save()