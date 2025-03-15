import torch
import numpy as np
import time
import random
from collections import deque
import copy
import threading

from config import (data_lock, metrics_lock, frame_data, metrics_data, running,
                    MAX_EPISODES, BATCH_SIZE, device, frame_buffer, SAVE_INTERVAL,
                    NUM_CARS)
from models import PPOAgent
from environment import CarEnv, MultiCarEnv


def training_thread():
    global frame_data, running, metrics_data

    print("Starting optimized multi-car training thread...")

    # Create multi-car environment and agent
    env = MultiCarEnv(num_cars=NUM_CARS)
    agent = PPOAgent(env.state_dim, env.action_dim, 1.0)  # Actions range from -1 to 1

    # Load model if exists
    model_loaded = agent.load()

    # Use the episode count from the loaded model
    start_episode = agent.episode_count if model_loaded else 0

    # Update metrics_data to store the starting episode number
    with metrics_lock:
        metrics_data['start_episode'] = start_episode
        metrics_data['num_cars'] = NUM_CARS
        metrics_data['active_cars'] = NUM_CARS
        metrics_data['car_rewards'] = [0] * NUM_CARS
        metrics_data['car_total_rewards'] = [0] * NUM_CARS
        metrics_data['car_laps'] = [0] * NUM_CARS
        metrics_data['main_car_idx'] = 0
        metrics_data['best_car_idx'] = 0

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
        warmup_env = CarEnv()  # Use single car env for warmup
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

        # Reset environment to get initial states for all cars
        states = env.reset()
        episode_rewards_per_car = [0] * NUM_CARS
        dones = [False] * NUM_CARS
        steps = 0

        # Tracking variables for this episode
        episode_actions = []
        episode_step_rewards = []

        # Track which cars are still active
        active_cars = [True] * NUM_CARS
        total_active_cars = NUM_CARS

        # Add episode noise for exploration - annealing over time
        exploration_noise = max(0.1, 1.0 * (0.98 ** episode))  # Decay from 1.0 to 0.1

        # Main episode loop
        while not all(dones) and steps < env.max_steps:
            # Select actions for all cars
            actions = []
            log_probs = []

            for i, state in enumerate(states):
                # Skip cars that are already done
                if dones[i]:
                    actions.append([0, 0])  # Dummy action
                    log_probs.append(0.0)  # Dummy log prob
                    continue

                # Select action with exploration for this car
                if np.random.random() < exploration_noise and episode < 100:
                    # Sometimes take random actions early in training
                    action = np.random.uniform(-1, 1, size=env.action_dim)
                    log_prob = 0.0  # Simple scalar for random actions
                else:
                    # Regular action selection
                    action, log_prob = agent.select_action(state)

                actions.append(action)
                log_probs.append(log_prob)

            # Take a step with all cars
            next_states, rewards, new_dones, infos, all_done = env.step(actions)

            # Update global dones
            for i in range(NUM_CARS):
                if new_dones[i]:
                    dones[i] = True
                    active_cars[i] = False

            total_active_cars = sum(1 for active in active_cars if active)

            steps += 1

            # Store trajectory data for active cars
            for i in range(NUM_CARS):
                if not dones[i] or new_dones[i]:  # Include the last transition for newly done cars
                    agent.add_to_trajectory(states[i], actions[i], log_probs[i], rewards[i], next_states[i],
                                            new_dones[i])

                # Update episode rewards
                episode_rewards_per_car[i] += rewards[i]

            # Track best and main car
            best_car_idx = env.get_best_car_index()
            main_car_idx = env.main_car_idx

            # Occasionally switch main car to best performing
            if steps % 50 == 0:
                main_car_idx = env.set_main_car_to_best()

            # Use the main car for action and reward visualization
            if len(actions) > main_car_idx:
                # Fix for the tolist() error - check the type and convert only if needed
                if hasattr(actions[main_car_idx], 'tolist'):
                    # If it's a numpy array, convert to list
                    episode_actions.append(actions[main_car_idx].tolist())
                else:
                    # If it's already a list, append directly
                    episode_actions.append(actions[main_car_idx])

            if len(rewards) > main_car_idx:
                episode_step_rewards.append(rewards[main_car_idx])

            # Update metrics more frequently for live display
            if steps % 1 == 0:  # Update every step
                with metrics_lock:
                    metrics_data['recent_actions'] = episode_actions[-20:] if episode_actions else []
                    metrics_data['recent_rewards'] = episode_step_rewards[-100:] if episode_step_rewards else []
                    metrics_data['current_steps'] = steps
                    metrics_data['active_cars'] = total_active_cars
                    metrics_data['car_rewards'] = rewards
                    metrics_data['car_total_rewards'] = episode_rewards_per_car
                    metrics_data['car_laps'] = [car.laps_completed for car in env.cars]
                    metrics_data['main_car_idx'] = main_car_idx
                    metrics_data['best_car_idx'] = best_car_idx

            # Update states
            states = next_states

            # Prepare visualization data - create deep copies to avoid reference issues
            if steps % 2 == 0 or all_done:  # Reduce frame updates to lower overhead
                # Create a snapshot of all cars
                car_snapshots = [copy.deepcopy(car) for car in env.cars]

                # Get maximum reward and laps for displaying
                max_reward = max(episode_rewards_per_car)
                max_laps = max(car.laps_completed for car in env.cars)

                frame_snapshot = {
                    'cars': car_snapshots,
                    'main_car_idx': main_car_idx,
                    'best_car_idx': best_car_idx,
                    'walls': env.walls,
                    'checkpoints': env.checkpoints,
                    'rewards': rewards.copy() if rewards else [0] * NUM_CARS,
                    'total_rewards': episode_rewards_per_car.copy(),
                    'laps': [car.laps_completed for car in env.cars],
                    'max_reward': max_reward,
                    'max_laps': max_laps,
                    'episode': episode,
                    'steps': steps,
                    'active_cars': total_active_cars,
                    'info': infos[main_car_idx].copy() if infos and main_car_idx < len(infos) else {}
                }

                # Try to update frame buffer without blocking
                try:
                    if frame_buffer.full():
                        try:
                            frame_buffer.get_nowait()  # Remove oldest frame
                        except Exception:
                            pass
                    frame_buffer.put_nowait(frame_snapshot)
                except Exception:
                    pass  # Never block training for visualization

            # Update agent periodically - adjust frequency based on performance
            if steps % BATCH_SIZE == 0 or all_done:
                # Start timing for neural network update
                nn_update_start = time.time()

                # For active cars, calculate last value
                for i in range(NUM_CARS):
                    if not dones[i]:
                        with torch.no_grad():
                            if agent.use_cpu_for_inference:
                                # CPU inference for faster single samples
                                next_state_tensor = torch.FloatTensor(next_states[i]).unsqueeze(0)
                                agent.cpu_actor_critic.eval()
                                _, _, last_value = agent.cpu_actor_critic(next_state_tensor)
                                agent.cpu_actor_critic.train()
                                last_value = last_value.numpy()
                            else:
                                # GPU inference
                                next_state_tensor = torch.FloatTensor(next_states[i]).to(device).unsqueeze(0)
                                agent.actor_critic.eval()
                                _, _, last_value = agent.actor_critic(next_state_tensor)
                                agent.actor_critic.train()
                                last_value = last_value.cpu().numpy()

                            if isinstance(last_value, np.ndarray):
                                last_value = float(last_value.item())

                            # End trajectory with this value
                            agent.end_trajectory(last_value)

                # Update using all collected experience
                total_loss, actor_loss, critic_loss, entropy_loss = agent.update()

                # Record neural network update time
                nn_update_time = time.time() - nn_update_start
                nn_update_count += 1
                avg_nn_update_time = avg_nn_update_time * 0.9 + nn_update_time * 0.1  # Exponential moving avg
                update_times.append(nn_update_time)

                # Calculate time since last update for metrics
                time_since_last_update = time.time() - last_nn_update_time
                last_nn_update_time = time.time()

                # Store loss history
                if total_loss > 0:
                    total_loss_history.append(total_loss)
                    actor_loss_history.append(actor_loss)
                    critic_loss_history.append(critic_loss)
                    entropy_loss_history.append(entropy_loss)

                    # Keep history limited to a reasonable size
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
                    'entropy_loss_history': entropy_loss_history
                }

                # Update metrics with minimal lock time
                with metrics_lock:
                    metrics_data.update(local_metrics)

        # End of episode - update agent's episode tracking
        # Use the maximum reward across all cars for episode tracking
        max_episode_reward = max(episode_rewards_per_car)
        agent.end_episode(max_episode_reward)

        # Record end of episode time
        last_episode_end_time = time.time()

        # Update metrics with the best car's performance
        max_laps = max(car.laps_completed for car in env.cars)

        # End of episode - update metrics
        episode_rewards.append(max_episode_reward)
        episode_lengths.append(steps)
        episode_laps.append(max_laps)
        recent_rewards.append(max_episode_reward)

        # Calculate episode time and update metrics
        episode_time = time.time() - episode_start_time

        # Calculate average reward across all cars
        avg_reward_per_car = sum(episode_rewards_per_car) / NUM_CARS

        # Update episode metrics atomically
        with metrics_lock:
            metrics_data['episode_times'].append(episode_time)
            metrics_data['episode_rewards'] = episode_rewards
            metrics_data['episode_lengths'] = episode_lengths
            metrics_data['episode_laps'] = episode_laps
            metrics_data['avg_rewards'].append(np.mean(recent_rewards) if recent_rewards else 0)
            metrics_data['current_steps'] = 0
            metrics_data['car_rewards'] = [0] * NUM_CARS
            metrics_data['car_total_rewards'] = [0] * NUM_CARS
            metrics_data['avg_reward_per_car'] = avg_reward_per_car

            # Calculate average episode time and estimate completion
            avg_episode_time = np.mean(metrics_data['episode_times'][-100:]) if metrics_data['episode_times'] else 0
            remaining_episodes = MAX_EPISODES - episode - 1
            estimated_completion_time = avg_episode_time * remaining_episodes

            metrics_data['avg_episode_time'] = avg_episode_time
            metrics_data['estimated_completion_time'] = estimated_completion_time
            metrics_data['estimated_runtime'] = time.time() - metrics_data['start_time']

            # Check for crashed cars
            for i, car in enumerate(env.cars):
                if hasattr(car, 'crashed') and car.crashed:
                    metrics_data['crash_locations'].append((car.x, car.y, episode))
                    # Keep only most recent 50 crash locations
                    if len(metrics_data['crash_locations']) > 50:
                        metrics_data['crash_locations'].pop(0)

        # Adaptive reporting frequency
        if episode < 100 or episode % 10 == 0:
            current_lr = agent.scheduler.get_last_lr()[0]
            print(
                f"Episode {episode}: Max Reward = {max_episode_reward:.2f}, Avg Reward = {np.mean(recent_rewards):.2f}, "
                f"Max Laps = {max_laps}, Steps = {steps}, Cars = {NUM_CARS}, "
                f"LR = {current_lr:.6f}, NN Update Time = {avg_nn_update_time:.4f}s")

        # Save model periodically
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            agent.save("ppo_car_model.pth")

    # Save final model
    agent.save("ppo_car_model.pth")
    print("Training finished!")