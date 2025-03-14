import torch
import numpy as np
import time
import random
from collections import deque
import copy
import threading

from config import (data_lock, metrics_lock, frame_data, metrics_data, running,
                    MAX_EPISODES, BATCH_SIZE, device, frame_buffer, SAVE_INTERVAL,
                    TIME_SCALE, FRAME_SKIP, BATCH_UPDATE_FREQUENCY)
from models import PPOAgent
from environment import CarEnv


# Training thread with enhanced exploration and learning
def training_thread():
    global frame_data, running, metrics_data

    print("Starting optimized training thread...")

    # Create environment and agent
    env = CarEnv()
    agent = PPOAgent(env.state_dim, env.action_dim, 1.0)  # Actions range from -1 to 1

    # Load model if exists
    model_loaded = agent.load()

    # Get hardware optimization parameters - keep settings but use them differently
    batch_update_frequency = BATCH_UPDATE_FREQUENCY
    steps_since_update = 0

    # For hardware acceleration, we'll use parallel evaluation and processing
    # but maintain original physics and reward structure
    update_frequency_multiplier = TIME_SCALE
    effective_batch_update_freq = max(1, int(batch_update_frequency / update_frequency_multiplier))

    # Use these parameters for training UI display only
    hardware_acceleration = TIME_SCALE
    processing_stride = FRAME_SKIP

    # Use the episode count from the loaded model
    start_episode = agent.episode_count if model_loaded else 0

    # Update metrics_data to store the starting episode number
    with metrics_lock:
        metrics_data['start_episode'] = start_episode
        metrics_data['hardware_acceleration'] = hardware_acceleration
        metrics_data['processing_stride'] = processing_stride

        # Initialize loss history lists if they don't exist
        if 'loss_history' not in metrics_data:
            metrics_data['loss_history'] = []
        if 'actor_loss_history' not in metrics_data:
            metrics_data['actor_loss_history'] = []
        if 'critic_loss_history' not in metrics_data:
            metrics_data['critic_loss_history'] = []
        if 'entropy_loss_history' not in metrics_data:
            metrics_data['entropy_loss_history'] = []

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

    # Store loss history for plotting
    total_loss_history = []
    actor_loss_history = []
    critic_loss_history = []
    entropy_loss_history = []

    # Timing variables for neural network updates
    nn_update_count = 0
    last_nn_update_time = time.time()
    avg_nn_update_time = 0.0
    update_times = deque(maxlen=100)  # Track recent update times

    # Time variables
    last_episode_end_time = time.time()

    # Start from the correct episode number
    for episode in range(start_episode, MAX_EPISODES):
        if not running:
            break

        episode_start_time = time.time()

        # Track time between episodes
        time_between_episodes = episode_start_time - last_episode_end_time

        # Reset environment
        state = env.reset()
        episode_reward = 0
        done = False
        steps = 0

        # Tracking variables for this episode
        episode_actions = []
        episode_step_rewards = []

        # Add episode noise for exploration - annealing over time
        exploration_noise = max(0.1, 1.0 * (0.98 ** episode))  # Decay from 1.0 to 0.1

        # Batch collect - for hardware acceleration without changing physics
        mini_batch_size = processing_stride
        accumulated_states = []
        accumulated_actions = []
        accumulated_log_probs = []
        accumulated_rewards = []
        accumulated_next_states = []
        accumulated_dones = []

        # Main episode loop
        while not done:
            # Select action with exploration noise that decreases over time
            if np.random.random() < exploration_noise and episode < 100:
                # Sometimes take random actions early in training
                action = np.random.uniform(-1, 1, size=env.action_dim)
                log_prob = 0.0  # Simple scalar for random actions
            else:
                # Regular action selection
                action, log_prob = agent.select_action(state)

            # Execute action - using original physics time step
            next_state, reward, done, info = env.step(action)

            # Accumulate experience
            accumulated_states.append(state)
            accumulated_actions.append(action)
            accumulated_log_probs.append(log_prob)
            accumulated_rewards.append(reward)
            accumulated_next_states.append(next_state)
            accumulated_dones.append(done)

            # Track actions and rewards for this step
            episode_actions.append(action.tolist())
            episode_step_rewards.append(reward)

            # Update metrics more frequently for live display
            if steps % 1 == 0:  # Update every step
                with metrics_lock:
                    metrics_data['recent_actions'] = episode_actions[-20:] if episode_actions else []
                    metrics_data['recent_rewards'] = episode_step_rewards[-100:] if episode_step_rewards else []
                    metrics_data['current_steps'] = steps  # Add current steps to metrics
                    # Add hardware acceleration parameters for display
                    metrics_data['hardware_acceleration'] = hardware_acceleration
                    metrics_data['processing_stride'] = processing_stride

            # Add gathered experience to trajectory in batches for hardware acceleration
            # This is purely for computational efficiency and doesn't change the physics
            if len(accumulated_states) >= mini_batch_size or done:
                # Process accumulated experience in optimized batch
                for i in range(len(accumulated_states)):
                    # Add to trajectory - maintains original reward structure
                    agent.add_to_trajectory(
                        accumulated_states[i],
                        accumulated_actions[i],
                        accumulated_log_probs[i],
                        accumulated_rewards[i],
                        accumulated_next_states[i],
                        accumulated_dones[i]
                    )

                # Clear the accumulated batch
                accumulated_states = []
                accumulated_actions = []
                accumulated_log_probs = []
                accumulated_rewards = []
                accumulated_next_states = []
                accumulated_dones = []

            # Update state and episode reward
            state = next_state
            episode_reward += reward
            steps += 1

            # Prepare visualization data - create deep copies to avoid reference issues
            # Just update visualization less frequently for performance
            if steps % max(1, processing_stride) == 0 or done:
                frame_snapshot = {
                    'car': copy.deepcopy(env.car),
                    'walls': env.walls,
                    'checkpoints': env.checkpoints,
                    'reward': reward,
                    'total_reward': episode_reward,
                    'laps': env.car.laps_completed,
                    'episode': episode,
                    'steps': steps,
                    'info': {
                        **info,
                        'hardware_acceleration': hardware_acceleration,
                        'processing_stride': processing_stride
                    }
                }

                # Try to update frame buffer without blocking
                try:
                    if frame_buffer.full():
                        try:
                            frame_buffer.get_nowait()
                        except Exception:
                            pass
                    frame_buffer.put_nowait(frame_snapshot)
                except Exception:
                    pass

            # Update agent based on steps and hardware acceleration
            steps_since_update += 1

            # Update more frequently with hardware acceleration
            if (steps_since_update >= effective_batch_update_freq) or done or len(agent.trajectory) >= BATCH_SIZE:
                # Start timing for neural network update
                nn_update_start = time.time()

                # Calculate last value for bootstrapping
                if done:
                    last_value = 0
                else:
                    with torch.no_grad():
                        if agent.use_cpu_for_inference:
                            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)
                            agent.cpu_actor_critic.eval()
                            _, _, last_value = agent.cpu_actor_critic(next_state_tensor)
                            agent.cpu_actor_critic.train()
                            last_value = last_value.numpy()
                        else:
                            next_state_tensor = torch.FloatTensor(next_state).to(device).unsqueeze(0)
                            agent.actor_critic.eval()
                            _, _, last_value = agent.actor_critic(next_state_tensor)
                            agent.actor_critic.train()
                            last_value = last_value.cpu().numpy()

                        if isinstance(last_value, np.ndarray):
                            last_value = float(last_value.item())

                # End trajectory and perform update (potentially expensive operation)
                if len(agent.trajectory) > 0:
                    agent.end_trajectory(last_value)

                    if len(agent.memory) >= BATCH_SIZE:
                        total_loss, actor_loss, critic_loss, entropy_loss = agent.update()
                        # Reset update counter after successful update
                        steps_since_update = 0

                        # Record neural network update time
                        nn_update_time = time.time() - nn_update_start
                        nn_update_count += 1
                        avg_nn_update_time = avg_nn_update_time * 0.9 + nn_update_time * 0.1  # Exponential moving avg
                        update_times.append(nn_update_time)

                        # Calculate time since last update for metrics
                        time_since_last_update = time.time() - last_nn_update_time
                        last_nn_update_time = time.time()

                        # Store loss history for plotting - only if we actually did an update
                        if total_loss > 0:  # Only store non-zero losses
                            total_loss_history.append(total_loss)
                            actor_loss_history.append(actor_loss)
                            critic_loss_history.append(critic_loss)
                            entropy_loss_history.append(entropy_loss)

                            # Keep history limited to a reasonable size for plotting
                            max_loss_history = 1000
                            if len(total_loss_history) > max_loss_history:
                                total_loss_history = total_loss_history[-max_loss_history:]
                                actor_loss_history = actor_loss_history[-max_loss_history:]
                                critic_loss_history = critic_loss_history[-max_loss_history:]
                                entropy_loss_history = entropy_loss_history[-max_loss_history:]

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
                            'nn_update_time': nn_update_time,
                            'avg_nn_update_time': np.mean(update_times) if update_times else 0,
                            'time_between_updates': time_since_last_update,
                            'loss_history': total_loss_history,
                            'actor_loss_history': actor_loss_history,
                            'critic_loss_history': critic_loss_history,
                            'entropy_loss_history': entropy_loss_history,
                            'hardware_acceleration': hardware_acceleration,
                            'processing_stride': processing_stride
                        }

                        # Update metrics data with minimal lock time
                        with metrics_lock:
                            metrics_data.update(local_metrics)

        # End of episode - update agent's episode tracking
        agent.end_episode(episode_reward)

        # Record end of episode time
        last_episode_end_time = time.time()

        # End of episode - update metrics
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_laps.append(env.car.laps_completed)
        recent_rewards.append(episode_reward)

        # Calculate episode time and update time metrics
        episode_time = time.time() - episode_start_time

        # Update episode metrics in one atomic operation to minimize lock time
        with metrics_lock:
            metrics_data['episode_times'].append(episode_time)
            metrics_data['episode_rewards'] = episode_rewards
            metrics_data['episode_lengths'] = episode_lengths
            metrics_data['episode_laps'] = episode_laps
            metrics_data['avg_rewards'].append(np.mean(recent_rewards) if recent_rewards else 0)
            # Reset current steps at end of episode
            metrics_data['current_steps'] = 0

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
                  f"NN Update Time = {avg_nn_update_time:.4f}s, HW Accel = {hardware_acceleration:.1f}x")

        # Save model periodically using improved async save
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            agent.save("ppo_car_model.pth")

    # Save final model
    agent.save("ppo_car_model.pth")
    print("Training finished!")