import pygame
import time
import math
import numpy as np
import platform
import os
from collections import deque
from queue import Empty

from config import (
    SCREEN_WIDTH, SCREEN_HEIGHT, DASHBOARD_WIDTH, DASHBOARD_HEIGHT,
    COMBINED_WIDTH, COMBINED_HEIGHT, FPS, WHITE, BLACK, RED, GREEN, BLUE,
    YELLOW, GRAY, DARK_GRAY, LIGHT_GRAY, PURPLE, CYAN, MAX_SENSOR_DISTANCE,
    data_lock, metrics_lock, render_queue, frame_data, metrics_data, running, MAX_EPISODES,
    MEMORY_SIZE, frame_buffer
)


# Helper function to draw improved line plots
def draw_line_plot(surface, rect, data, min_val=None, max_val=None, color=WHITE, label=None):
    if not data:
        return

    # Determine min and max values if not provided
    if min_val is None:
        min_val = min(data) if data else 0
    if max_val is None:
        max_val = max(data) if data else 1

    # Ensure we don't divide by zero and add some padding
    if max_val == min_val:
        max_val = min_val + 1

    # Add margin to display area to ensure labels are visible
    margin = 30  # Increased margin for labels
    plot_rect = pygame.Rect(
        rect.left + margin,
        rect.top + 20,
        rect.width - margin - 10,
        rect.height - margin - 10
    )

    # Draw axes
    pygame.draw.line(surface, LIGHT_GRAY, (plot_rect.left, plot_rect.bottom), (plot_rect.right, plot_rect.bottom),
                     1)  # X-axis
    pygame.draw.line(surface, LIGHT_GRAY, (plot_rect.left, plot_rect.top), (plot_rect.left, plot_rect.bottom),
                     1)  # Y-axis

    # Draw label at the top
    if label:
        label_font = pygame.font.SysFont('Arial', 12)
        label_surface = label_font.render(label, True, color)
        surface.blit(label_surface, (rect.left + 5, rect.top + 5))

    # Draw min/max values with better positioning
    value_font = pygame.font.SysFont('Arial', 10)
    min_surface = value_font.render(f"{min_val:.2f}", True, LIGHT_GRAY)
    max_surface = value_font.render(f"{max_val:.2f}", True, LIGHT_GRAY)

    # Position the values with better margins
    surface.blit(min_surface, (plot_rect.left - min_surface.get_width() - 5, plot_rect.bottom - 5))
    surface.blit(max_surface, (plot_rect.left - max_surface.get_width() - 5, plot_rect.top))

    # Draw X-axis labels (episodes/steps)
    if len(data) > 1:
        # Draw at least start, middle and end labels
        start_label = value_font.render("0", True, LIGHT_GRAY)
        mid_label = value_font.render(f"{len(data) // 2}", True, LIGHT_GRAY)
        end_label = value_font.render(f"{len(data) - 1}", True, LIGHT_GRAY)

        surface.blit(start_label, (plot_rect.left, plot_rect.bottom + 5))
        surface.blit(mid_label, (plot_rect.left + plot_rect.width // 2, plot_rect.bottom + 5))
        surface.blit(end_label, (plot_rect.right - end_label.get_width(), plot_rect.bottom + 5))

        # Add x-axis caption
        x_caption = value_font.render("Episodes", True, LIGHT_GRAY)
        surface.blit(x_caption,
                     (plot_rect.left + plot_rect.width // 2 - x_caption.get_width() // 2, plot_rect.bottom + 20))

    # Draw intermediate y-axis grid lines
    for i in range(1, 4):
        y = plot_rect.bottom - (i * plot_rect.height / 4)
        pygame.draw.line(surface, DARK_GRAY, (plot_rect.left, y), (plot_rect.right, y), 1)
        value = min_val + (max_val - min_val) * (i / 4)
        value_label = value_font.render(f"{value:.2f}", True, LIGHT_GRAY)
        surface.blit(value_label, (plot_rect.left - value_label.get_width() - 5, y - value_label.get_height() // 2))

    # Draw the plot line
    points = []
    for i, value in enumerate(data):
        x = plot_rect.left + (i / (len(data) - 1 if len(data) > 1 else 1)) * plot_rect.width
        y = plot_rect.bottom - ((value - min_val) / (max_val - min_val)) * plot_rect.height
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)

    # Draw the last point as a circle with value
    if points:
        last_point = points[-1]
        pygame.draw.circle(surface, color, (int(last_point[0]), int(last_point[1])), 3)

        # Show last value
        last_value = data[-1]
        last_label = value_font.render(f"{last_value:.2f}", True, color)
        surface.blit(last_label, (last_point[0] + 5, last_point[1] - 10))


# Function to get frame data safely
def get_frame_data():
    global latest_frame
    try:
        # Non-blocking approach - try to get latest frame from buffer
        frame = frame_buffer.get_nowait()
        latest_frame = frame  # Update the latest valid frame
        return frame
    except Empty:
        # Return the last valid frame if buffer is empty
        if latest_frame is not None:
            return latest_frame

        # Fall back to lock-based approach if no frames yet
        with data_lock:
            return {k: v for k, v in frame_data.items()}


# Function to get metrics data safely
def get_metrics_data():
    try:
        with metrics_lock:
            # Make a shallow copy of non-list items or small lists
            local_metrics = {
                k: v for k, v in metrics_data.items()
                if not isinstance(v, list) or len(v) < 100
            }

            # Only copy these specific lists if needed
            if 'episode_rewards' in metrics_data:
                local_metrics['episode_rewards'] = metrics_data['episode_rewards'].copy() if metrics_data[
                    'episode_rewards'] else []
            if 'avg_rewards' in metrics_data:
                local_metrics['avg_rewards'] = metrics_data['avg_rewards'].copy() if metrics_data['avg_rewards'] else []
            if 'recent_actions' in metrics_data:
                local_metrics['recent_actions'] = metrics_data['recent_actions'].copy() if metrics_data[
                    'recent_actions'] else []
            if 'recent_rewards' in metrics_data:
                local_metrics['recent_rewards'] = metrics_data['recent_rewards'].copy() if metrics_data[
                    'recent_rewards'] else []

        return local_metrics
    except Exception as e:
        print(f"Error getting metrics data: {e}")
        return {}


# Combined rendering thread that shows both simulation and dashboard
def combined_rendering_thread():
    global frame_data, running, metrics_data, frame_buffer, latest_frame

    try:
        print("Starting combined rendering thread with optimizations...")

        # Initialize pygame
        if not pygame.get_init():
            pygame.init()

        # Create a single large window that shows both components
        combined_screen = pygame.display.set_mode((COMBINED_WIDTH, COMBINED_HEIGHT))
        pygame.display.set_caption("Car Reinforcement Learning with Dashboard")
        clock = pygame.time.Clock()
        font = pygame.font.SysFont('Arial', 16)
        title_font = pygame.font.SysFont('Arial', 20, bold=True)
        regular_font = pygame.font.SysFont('Arial', 14)
        small_font = pygame.font.SysFont('Arial', 12)

        print(f"Created combined window: {COMBINED_WIDTH}x{COMBINED_HEIGHT}")

        # Car image setup - precompute this once
        car_img = pygame.Surface((30, 15), pygame.SRCALPHA)
        pygame.draw.rect(car_img, RED, (0, 0, 30, 15), border_radius=3)
        pygame.draw.polygon(car_img, (100, 200, 255), [(15, 3), (22, 3), (22, 12), (15, 12)])
        pygame.draw.rect(car_img, (255, 255, 100), (26, 2, 3, 3))
        pygame.draw.rect(car_img, (255, 255, 100), (26, 10, 3, 3))
        pygame.draw.rect(car_img, (200, 0, 0), (1, 2, 2, 3))
        pygame.draw.rect(car_img, (200, 0, 0), (1, 10, 2, 3))
        pygame.draw.rect(car_img, (30, 30, 30), (5, -1, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (5, 14, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (20, -1, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (20, 14, 5, 2))

        # Dashboard sections with reorganized layout
        dashboard_sections = {
            'training_stats': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 10, 380, 150), 'title': 'Training Statistics'},
            'episode_stats': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 10, 380, 150), 'title': 'Current Episode'},
            'reward_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 170, 380, 180), 'title': 'Episode Reward History'},
            'avg_reward_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 170, 380, 180),
                                'title': 'Average Reward (100 ep)'},
            'loss_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 360, 380, 180), 'title': 'Loss History'},
            'action_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 360, 380, 180), 'title': 'Recent Actions'},
            'sensor_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 550, 770, 230), 'title': 'Sensor Readings'}
        }

        # Track data for plots
        reward_history = []
        loss_history = []
        action_history = []
        sensor_history = []

        # For downsampling larger datasets
        downsample_factor = 1
        max_plot_points = 100

        # Define the global latest_frame variable
        latest_frame = None

        # Frame timing variables
        target_fps = 60  # Increase target FPS for smoother rendering
        target_frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        # Performance monitoring
        render_times = deque(maxlen=100)
        update_times = deque(maxlen=100)

        # Precompute expensive assets
        precomputed_track_surface = None
        track_hash = None  # For detecting if track needs redrawing

        # Precompute sensors angles
        sensor_angles = [(i * 2 * math.pi / 8) % (2 * math.pi) for i in range(8)]
        sensor_names = ["Front", "FR", "Right", "RR", "Rear", "RL", "Left", "FL"]

        print("Starting render loop")
        while running:
            # Calculate elapsed time since last frame
            current_time = time.time()
            frame_delta = current_time - last_frame_time

            # If it's not time for a new frame yet, yield to other threads
            if frame_delta < target_frame_time:
                # Sleep for a short time to avoid busy waiting
                time.sleep(min(0.001, target_frame_time - frame_delta))
                continue

            # We're ready for a new frame
            start_time = time.time()
            last_frame_time = current_time

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Clear the entire screen
            combined_screen.fill(BLACK)

            # Get frame data without blocking (using our optimized function)
            local_frame_data = get_frame_data()

            # Get metrics data without blocking (using our optimized function)
            local_metrics_data = get_metrics_data()

            # Only proceed with rendering if we have valid data
            if local_frame_data:
                # Extract data outside the lock
                car = local_frame_data.get('car')
                walls = local_frame_data.get('walls')
                checkpoints = local_frame_data.get('checkpoints')
                reward = local_frame_data.get('reward', 0)
                total_reward = local_frame_data.get('total_reward', 0)
                laps = local_frame_data.get('laps', 0)
                episode = local_frame_data.get('episode', 0)
                info = local_frame_data.get('info', {})

                current_metrics = local_metrics_data or {}
                current_frame = local_frame_data

                # Draw simulation part (left side)
                # ===================================
                simulation_surface = combined_screen.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

                # Only recompute track surface if needed
                current_track_hash = hash(str(walls)) if walls else None
                if precomputed_track_surface is None or track_hash != current_track_hash:
                    # Track surface needs redrawing
                    track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

                    if walls:
                        # Fill the track area with a dark gray
                        pygame.draw.rect(track_surface, (30, 30, 30), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

                        # Draw the actual track in a slightly lighter gray
                        if len(walls) >= 32:  # If using our new oval track with inner/outer walls
                            # Extract points from walls for the outer and inner boundaries
                            outer_points = [(walls[i][0], walls[i][1]) for i in
                                            range(16)]  # First 16 segments are outer
                            inner_points = [(walls[i][0], walls[i][1]) for i in range(16, 32)]  # Next 16 are inner

                            # Draw the track surface
                            pygame.draw.polygon(track_surface, (50, 50, 50), outer_points)
                            pygame.draw.polygon(track_surface, (30, 30, 30), inner_points)

                    precomputed_track_surface = track_surface
                    track_hash = current_track_hash

                # Use the precomputed track surface
                simulation_surface.blit(precomputed_track_surface, (0, 0))

                # Draw walls - could precompute these too if they're static
                if walls:
                    for wall in walls:
                        pygame.draw.line(simulation_surface, WHITE, (wall[0], wall[1]), (wall[2], wall[3]), 2)

                # Draw checkpoints - these change color but positions are static
                if checkpoints:
                    # Draw the track's center line connecting checkpoints
                    if len(checkpoints) > 1:
                        # Create a closed loop by connecting the last checkpoint to the first
                        for i in range(len(checkpoints)):
                            next_i = (i + 1) % len(checkpoints)
                            pygame.draw.line(simulation_surface, (80, 80, 80), checkpoints[i], checkpoints[next_i], 1)

                    # Draw the actual checkpoint markers
                    for i, cp in enumerate(checkpoints):
                        # Current checkpoint is green, next checkpoint is yellow, others are brighter gray
                        if car and i == car.checkpoint_idx:
                            checkpoint_color = GREEN
                            radius = 8
                        elif car and i == (car.checkpoint_idx + 1) % len(checkpoints):
                            checkpoint_color = YELLOW
                            radius = 6
                        else:
                            checkpoint_color = (120, 120, 120)  # Brighter gray to be visible
                            radius = 4

                        pygame.draw.circle(simulation_surface, checkpoint_color, cp, radius)

                # Draw car
                if car:
                    # Create a rotated copy of the car image
                    angle_degrees = -math.degrees(car.angle)  # Pygame uses opposite rotation direction
                    rotated_car = pygame.transform.rotate(car_img, angle_degrees)
                    # Get the rect of the rotated image and position it at the car's position
                    car_rect = rotated_car.get_rect(center=(car.x, car.y))
                    simulation_surface.blit(rotated_car, car_rect.topleft)

                    # Draw sensors
                    for i in range(len(car.sensor_readings)):
                        # Calculate sensor angle
                        sensor_angle = (car.angle + sensor_angles[i]) % (2 * math.pi)
                        # Calculate endpoint based on reading
                        end_x = car.x + math.cos(sensor_angle) * car.sensor_readings[i]
                        end_y = car.y + math.sin(sensor_angle) * car.sensor_readings[i]
                        # Draw line
                        pygame.draw.line(simulation_surface, GREEN, (car.x, car.y), (end_x, end_y), 1)

                # Draw simulation stats
                avg_frame_time = sum(render_times) / max(1, len(render_times))
                avg_update_time = sum(update_times) / max(1, len(update_times))

                nn_update_time = current_metrics.get('nn_update_time', 0)
                time_between_updates = current_metrics.get('time_between_updates', 0)
                gpu_util = current_metrics.get('gpu_util', 0)

                info_text = [
                    f"Episode: {episode}",
                    f"Reward: {reward:.2f}",
                    f"Total Reward: {total_reward:.2f}",
                    f"Laps: {laps}",
                    f"Speed: {car.speed:.2f}" if car else "Speed: 0.00",
                    f"Distance: {car.total_distance:.2f}" if car else "Distance: 0.00",
                    f"Time Alive: {car.time_alive}" if car else "Time Alive: 0",
                    f"Direction Alignment: {info.get('direction_alignment', 0):.2f}" if info else "Direction Alignment: 0.00",
                    f"FPS: {clock.get_fps():.1f}",
                    f"Render Time: {avg_frame_time * 1000:.1f}ms",
                    f"Update Time: {avg_update_time * 1000:.1f}ms",
                    f"NN Update Time: {nn_update_time * 1000:.1f}ms",
                    f"Time Between NN Updates: {time_between_updates:.2f}s",
                    f"GPU Util: {gpu_util}%" if gpu_util else ""
                ]

                for i, text in enumerate(info_text):
                    text_surface = font.render(text, True, WHITE)
                    simulation_surface.blit(text_surface, (10, 10 + i * 20))

                # Draw a divider between simulation and dashboard
                pygame.draw.line(combined_screen, WHITE, (SCREEN_WIDTH, 0), (SCREEN_WIDTH, COMBINED_HEIGHT), 2)

                # Draw dashboard part (right side)
                # ===================================
                # Draw sections
                for section_id, section in dashboard_sections.items():
                    # Draw section background
                    pygame.draw.rect(combined_screen, DARK_GRAY, section['rect'])
                    pygame.draw.rect(combined_screen, LIGHT_GRAY, section['rect'], 1)

                    # Draw section title
                    title_surface = title_font.render(section['title'], True, WHITE)
                    combined_screen.blit(title_surface, (section['rect'].x + 10, section['rect'].y + 5))

                    # Draw section content based on section ID
                    if section_id == 'training_stats':
                        # Get the starting episode (from loaded model if applicable)
                        start_episode = current_metrics.get('start_episode', 0)
                        current_episodes = len(current_metrics.get('episode_rewards', []))
                        total_episodes = start_episode + current_episodes

                        # Training statistics in two columns for better readability
                        col1_stats = [
                            f"Episodes: {total_episodes}/{MAX_EPISODES}",
                            f"Updates: {current_metrics.get('updates_performed', 0)}",
                            f"Memory: {current_metrics.get('memory_usage', 0)}/{MEMORY_SIZE}",
                            f"Total Laps: {sum(current_metrics.get('episode_laps', [0]))}"
                        ]

                        # Format total training time
                        total_time = current_metrics.get('total_training_time', 0)
                        current_time = time.time() - current_metrics.get('start_time', time.time())
                        all_time = total_time + current_time

                        days, remainder = divmod(int(all_time), 86400)
                        hours, remainder = divmod(remainder, 3600)
                        minutes, seconds = divmod(remainder, 60)

                        time_str = ""
                        if days > 0:
                            time_str += f"{days}d "
                        if hours > 0 or days > 0:
                            time_str += f"{hours}h "
                        time_str += f"{minutes}m {seconds}s"

                        col2_stats = [
                            f"Total Train Time: {time_str}",
                            f"Avg Reward: {np.mean(current_metrics.get('episode_rewards', [])[-100:]) if current_metrics.get('episode_rewards', []) else 0:.2f}",
                            f"Avg Ep Time: {current_metrics.get('avg_episode_time', 0):.2f}s",
                            f"Est. Completion: {time.strftime('%H:%M:%S', time.gmtime(current_metrics.get('estimated_completion_time', 0)))}"
                        ]

                        # Column 1
                        for i, stat in enumerate(col1_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            combined_screen.blit(stat_surface,
                                                 (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                        # Column 2
                        for i, stat in enumerate(col2_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            combined_screen.blit(stat_surface,
                                                 (section['rect'].x + 200, section['rect'].y + 30 + i * 18))

                    elif section_id == 'episode_stats':

                        # Current episode statistics in two columns with added step count

                        col1_stats = [

                            f"Episode: {current_frame.get('episode', 0)}",

                            f"Steps: {current_frame.get('steps', 0)}",  # Added steps count

                            f"Reward: {current_frame.get('reward', 0):.2f}",

                            f"Total Reward: {current_frame.get('total_reward', 0):.2f}",

                            f"Laps: {current_frame.get('laps', 0)}"

                        ]

                        col2_stats = [

                            f"LR: {current_metrics.get('learning_rate', 0):.6f}",

                            f"Entropy: {current_metrics.get('entropy_coef', 0):.4f}",

                            f"Loss: {current_metrics.get('last_loss', 0):.4f}",

                            f"A/C Loss: {current_metrics.get('actor_loss', 0):.2f}/{current_metrics.get('critic_loss', 0):.2f}",

                            f"Update Time: {current_metrics.get('avg_nn_update_time', 0) * 1000:.1f}ms"
                            # Display nn update time in ms

                        ]

                        # Column 1

                        for i, stat in enumerate(col1_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)

                            combined_screen.blit(stat_surface,

                                                 (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                        # Column 2

                        for i, stat in enumerate(col2_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)

                            combined_screen.blit(stat_surface,

                                                 (section['rect'].x + 200, section['rect'].y + 30 + i * 18))

                    elif section_id == 'reward_plot':
                        # Reward history plot
                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Update reward history data
                        if current_metrics.get('episode_rewards', []):
                            reward_history = current_metrics['episode_rewards']

                            # Downsample if too many points
                            if len(reward_history) > max_plot_points:
                                downsample_factor = max(1, len(reward_history) // max_plot_points)
                                reward_history = reward_history[::downsample_factor]

                            # Draw plot
                            draw_line_plot(combined_screen, plot_rect, reward_history,
                                           min_val=min(reward_history) if reward_history else 0,
                                           max_val=max(reward_history) if reward_history else 1,
                                           color=GREEN, label="Episode Reward")

                    elif section_id == 'loss_plot':

                        # Loss history

                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,

                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Draw actor and critic loss if available

                        if current_metrics.get('loss_history', []):

                            # Get loss history data

                            loss_history = current_metrics.get('loss_history', [])

                            actor_loss_history = current_metrics.get('actor_loss_history', [])

                            critic_loss_history = current_metrics.get('critic_loss_history', [])

                            # Downsample if too many points

                            if len(loss_history) > max_plot_points:
                                downsample_factor = max(1, len(loss_history) // max_plot_points)

                                loss_history = loss_history[::downsample_factor]

                            if len(actor_loss_history) > max_plot_points:
                                downsample_factor = max(1, len(actor_loss_history) // max_plot_points)

                                actor_loss_history = actor_loss_history[::downsample_factor]

                            if len(critic_loss_history) > max_plot_points:
                                downsample_factor = max(1, len(critic_loss_history) // max_plot_points)

                                critic_loss_history = critic_loss_history[::downsample_factor]

                            # Draw total loss plot

                            if loss_history:
                                draw_line_plot(combined_screen, plot_rect, loss_history,

                                               color=YELLOW, label="Total Loss")

                            # Draw actor loss plot

                            if actor_loss_history:
                                draw_line_plot(combined_screen, plot_rect, actor_loss_history,

                                               color=RED, label="Actor Loss")

                            # Draw critic loss plot

                            if critic_loss_history:
                                draw_line_plot(combined_screen, plot_rect, critic_loss_history,

                                               color=BLUE, label="Critic Loss")

                        else:

                            # Draw placeholder

                            text = small_font.render("Loss data will appear here as training progresses", True, WHITE)

                            combined_screen.blit(text, (plot_rect.x + 10, plot_rect.y + plot_rect.height // 2))


                    elif section_id == 'action_plot':

                        # Improved recent actions plot

                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,

                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Update action history - more responsive to recent actions

                        if current_metrics.get('recent_actions', []):

                            # Extract acceleration and steering

                            if len(current_metrics['recent_actions']) > 0 and len(
                                    current_metrics['recent_actions'][0]) >= 2:

                                accel_actions = [a[0] for a in current_metrics['recent_actions']]

                                steer_actions = [a[1] for a in current_metrics['recent_actions']]

                                # Add a background rect to separate accel and steering

                                half_height = plot_rect.height // 2

                                upper_rect = pygame.Rect(plot_rect.left, plot_rect.top, plot_rect.width, half_height)

                                lower_rect = pygame.Rect(plot_rect.left, plot_rect.top + half_height, plot_rect.width,

                                                         half_height)

                                # Fill background with slight color to distinguish areas

                                pygame.draw.rect(combined_screen, (0, 40, 40), upper_rect)

                                pygame.draw.rect(combined_screen, (40, 0, 40), lower_rect)

                                # Draw zero lines for reference

                                zero_y_accel = upper_rect.top + upper_rect.height // 2

                                zero_y_steer = lower_rect.top + lower_rect.height // 2

                                pygame.draw.line(combined_screen, (100, 100, 100),

                                                 (upper_rect.left, zero_y_accel),

                                                 (upper_rect.right, zero_y_accel), 1)

                                pygame.draw.line(combined_screen, (100, 100, 100),

                                                 (lower_rect.left, zero_y_steer),

                                                 (lower_rect.right, zero_y_steer), 1)

                                # Add labels

                                label_font = pygame.font.SysFont('Arial', 12)

                                accel_label = label_font.render("Acceleration (Fwd/Rev)", True, CYAN)

                                steer_label = label_font.render("Steering (Left/Right)", True, PURPLE)

                                combined_screen.blit(accel_label, (upper_rect.left + 5, upper_rect.top + 5))

                                combined_screen.blit(steer_label, (lower_rect.left + 5, lower_rect.top + 5))

                                # Draw points for acceleration in upper half

                                accel_points = []

                                for i, accel in enumerate(accel_actions):
                                    x = upper_rect.left + (i / (

                                        len(accel_actions) - 1 if len(accel_actions) > 1 else 1)) * upper_rect.width

                                    # Map -1 to 1 range to the upper rect height

                                    y = zero_y_accel - (accel * upper_rect.height / 2)

                                    accel_points.append((x, y))

                                if len(accel_points) > 1:
                                    pygame.draw.lines(combined_screen, CYAN, False, accel_points, 2)

                                # Draw points for steering in lower half

                                steer_points = []

                                for i, steer in enumerate(steer_actions):
                                    x = lower_rect.left + (i / (

                                        len(steer_actions) - 1 if len(steer_actions) > 1 else 1)) * lower_rect.width

                                    # Map -1 to 1 range to the lower rect height

                                    y = zero_y_steer - (steer * lower_rect.height / 2)

                                    steer_points.append((x, y))

                                if len(steer_points) > 1:
                                    pygame.draw.lines(combined_screen, PURPLE, False, steer_points, 2)

                                # Draw the last values

                                if accel_points:
                                    pygame.draw.circle(combined_screen, CYAN,

                                                       (int(accel_points[-1][0]), int(accel_points[-1][1])), 4)

                                    last_accel = accel_actions[-1]

                                    last_accel_label = label_font.render(f"{last_accel:.2f}", True, CYAN)

                                    combined_screen.blit(last_accel_label,

                                                         (accel_points[-1][0] + 5, accel_points[-1][1] - 10))

                                if steer_points:
                                    pygame.draw.circle(combined_screen, PURPLE,

                                                       (int(steer_points[-1][0]), int(steer_points[-1][1])), 4)

                                    last_steer = steer_actions[-1]

                                    last_steer_label = label_font.render(f"{last_steer:.2f}", True, PURPLE)

                                    combined_screen.blit(last_steer_label,

                                                         (steer_points[-1][0] + 5, steer_points[-1][1] - 10))

                                # Add explanatory legends

                                small_font = pygame.font.SysFont('Arial', 10)

                                accel_pos = small_font.render("Forward", True, LIGHT_GRAY)

                                accel_neg = small_font.render("Reverse", True, LIGHT_GRAY)

                                steer_pos = small_font.render("Right", True, LIGHT_GRAY)

                                steer_neg = small_font.render("Left", True, LIGHT_GRAY)

                                # Position the legends

                                combined_screen.blit(accel_pos, (upper_rect.left + 5, upper_rect.top + 5))

                                combined_screen.blit(accel_neg, (upper_rect.left + 5, zero_y_accel + 5))

                                combined_screen.blit(steer_pos, (lower_rect.left + 5, lower_rect.top + 5))

                                combined_screen.blit(steer_neg, (lower_rect.left + 5, zero_y_steer + 5))

                        else:

                            # If no action data, show a message

                            text = small_font.render("Action data will appear here as the agent acts", True, WHITE)

                            combined_screen.blit(text, (plot_rect.x + 10, plot_rect.y + plot_rect.height // 2))

                    elif section_id == 'avg_reward_plot':
                        # Average reward history plot
                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Get average rewards data
                        if current_metrics.get('avg_rewards', []):
                            avg_rewards = current_metrics['avg_rewards']

                            # Downsample if too many points
                            if len(avg_rewards) > max_plot_points:
                                downsample_factor = max(1, len(avg_rewards) // max_plot_points)
                                avg_rewards = avg_rewards[::downsample_factor]

                            # Draw plot with custom min/max for better visualization
                            min_val = min(avg_rewards) if avg_rewards else 0
                            max_val = max(avg_rewards) if avg_rewards else 1

                            # Add a small margin to the max value for better visibility
                            value_range = max_val - min_val
                            min_val = max(0, min_val - value_range * 0.1)
                            max_val = max_val + value_range * 0.1

                            draw_line_plot(combined_screen, plot_rect, avg_rewards,
                                           min_val=min_val, max_val=max_val,
                                           color=CYAN, label="100-Episode Moving Avg")

                    elif section_id == 'sensor_plot':
                        # Improved sensor readings visualization
                        if car and hasattr(car, 'sensor_readings'):
                            sensor_readings = car.sensor_readings

                            # Use left side for radar chart
                            radar_rect = pygame.Rect(
                                section['rect'].x + 20,
                                section['rect'].y + 40,
                                section['rect'].height - 60,  # Make it square
                                section['rect'].height - 60
                            )

                            # Use right side for bar graph
                            bar_rect = pygame.Rect(
                                radar_rect.right + 50,
                                section['rect'].y + 40,
                                section['rect'].right - radar_rect.right - 70,
                                section['rect'].height - 60
                            )

                            # Create an improved radar chart
                            center_x = radar_rect.centerx
                            center_y = radar_rect.centery
                            radius = min(radar_rect.width, radar_rect.height) // 2 - 10

                            # Normalize readings
                            max_reading = MAX_SENSOR_DISTANCE
                            normalized_readings = [min(1.0, r / max_reading) for r in sensor_readings]

                            # Draw sensor names and directions
                            name_font = pygame.font.SysFont('Arial', 12)

                            # Draw concentric circles for distance reference
                            for i in range(4):
                                r = radius * (i + 1) / 4
                                pygame.draw.circle(combined_screen, DARK_GRAY, (center_x, center_y), int(r), 1)
                                # Add distance label
                                if i > 0:
                                    dist_label = name_font.render(f"{int(MAX_SENSOR_DISTANCE * i / 4)}", True,
                                                                  LIGHT_GRAY)
                                    combined_screen.blit(dist_label, (center_x - 8, center_y - int(r) - 10))

                            # Draw sensor lines and labels
                            for i in range(len(normalized_readings)):
                                angle = sensor_angles[i]
                                # Draw line from center to edge
                                end_x = center_x + radius * math.cos(angle)
                                end_y = center_y + radius * math.sin(angle)
                                pygame.draw.line(combined_screen, DARK_GRAY, (center_x, center_y), (end_x, end_y), 1)

                                # Draw sensor label at outer edge
                                label_x = center_x + (radius + 15) * math.cos(angle)
                                label_y = center_y + (radius + 15) * math.sin(angle)
                                label = name_font.render(sensor_names[i], True, WHITE)
                                # Center the text around the point
                                label_rect = label.get_rect(center=(label_x, label_y))
                                combined_screen.blit(label, label_rect)

                                # Draw actual sensor reading point
                                r = radius * (1 - normalized_readings[i])
                                point_x = center_x + r * math.cos(angle)
                                point_y = center_y + r * math.sin(angle)

                                # Draw a colored circle showing reading
                                color_intensity = int(255 * (1 - normalized_readings[i]))
                                point_color = (255, color_intensity, color_intensity)
                                pygame.draw.circle(combined_screen, point_color, (int(point_x), int(point_y)), 6)

                            # Connect the points to form a better visualization
                            points = []
                            for i in range(len(normalized_readings)):
                                angle = sensor_angles[i]
                                r = radius * (1 - normalized_readings[i])
                                point_x = center_x + r * math.cos(angle)
                                point_y = center_y + r * math.sin(angle)
                                points.append((int(point_x), int(point_y)))

                            if len(points) > 2:
                                # Create a semi-transparent effect by layering colors
                                for alpha in range(3):
                                    alpha_factor = (3 - alpha) / 3
                                    alpha_color = (
                                        int(180 * alpha_factor),
                                        int(20 * alpha_factor),
                                        int(20 * alpha_factor)
                                    )
                                    alpha_points = []
                                    for i, point in enumerate(points):
                                        angle = sensor_angles[i]
                                        r = radius * (1 - normalized_readings[i] * (1 - alpha * 0.2))
                                        px = center_x + r * math.cos(angle)
                                        py = center_y + r * math.sin(angle)
                                        alpha_points.append((int(px), int(py)))

                                    # Draw filled and outline
                                    pygame.draw.polygon(combined_screen, alpha_color, alpha_points, 0)

                                # Final outline
                                pygame.draw.polygon(combined_screen, RED, points, 2)

                            # Add a legend
                            legend_font = pygame.font.SysFont('Arial', 14)
                            legend_text = legend_font.render("Sensor Proximity Radar (Red = Close, White = Far)", True,
                                                             WHITE)
                            combined_screen.blit(legend_text,
                                                 (center_x - legend_text.get_width() // 2, section['rect'].y + 15))

                            # Draw bar chart on the right side
                            bar_width = (bar_rect.width) / len(sensor_readings)
                            bar_font = pygame.font.SysFont('Arial', 11)

                            # Draw axis
                            pygame.draw.line(combined_screen, WHITE,
                                             (bar_rect.left, bar_rect.bottom),
                                             (bar_rect.right, bar_rect.bottom), 1)

                            # Title for bar chart
                            bar_title = legend_font.render("Distance Readings", True, WHITE)
                            combined_screen.blit(bar_title,
                                                 (
                                                     bar_rect.centerx - bar_title.get_width() // 2,
                                                     section['rect'].y + 15))

                            # Y-axis labels and grid lines
                            for i in range(5):
                                y_pos = bar_rect.bottom - (i * bar_rect.height // 4)
                                y_value = (i * MAX_SENSOR_DISTANCE // 4)
                                y_label = bar_font.render(f"{y_value}", True, LIGHT_GRAY)
                                combined_screen.blit(y_label, (bar_rect.left - y_label.get_width() - 5, y_pos - 7))

                                # Grid line
                                pygame.draw.line(combined_screen, DARK_GRAY,
                                                 (bar_rect.left, y_pos),
                                                 (bar_rect.right, y_pos), 1)

                            # Draw bars
                            for i, reading in enumerate(sensor_readings):
                                x = bar_rect.left + i * bar_width
                                bar_height = (reading / MAX_SENSOR_DISTANCE) * bar_rect.height

                                # Determine color based on proximity (red for close, green for far)
                                proximity_factor = 1 - (reading / MAX_SENSOR_DISTANCE)
                                bar_color = (
                                    int(255 * proximity_factor),
                                    int(255 * (1 - proximity_factor)),
                                    0
                                )

                                pygame.draw.rect(combined_screen, bar_color,
                                                 (x, bar_rect.bottom - bar_height,
                                                  bar_width - 2, bar_height))

                                # Add sensor name below bar
                                name_label = bar_font.render(sensor_names[i], True, LIGHT_GRAY)
                                label_x = x + bar_width / 2 - name_label.get_width() / 2
                                combined_screen.blit(name_label, (label_x, bar_rect.bottom + 5))

                                # Add value above bar
                                value_label = bar_font.render(f"{int(reading)}", True, WHITE)
                                value_x = x + bar_width / 2 - value_label.get_width() / 2
                                value_y = bar_rect.bottom - bar_height - value_label.get_height() - 2
                                if value_y < bar_rect.top:  # Ensure label isn't too high
                                    value_y = bar_rect.top
                                combined_screen.blit(value_label, (value_x, value_y))

            # Update display
            pygame.display.flip()

            # Record render time for performance monitoring
            render_time = time.time() - start_time
            render_times.append(render_time)

            # Record frame time for FPS calculation
            clock.tick(target_fps)

    except Exception as e:
        print(f"Error in combined rendering thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Combined rendering thread exiting")
        pygame.quit()


# Rendering thread for simulation only (separate window mode)
def rendering_thread():
    global frame_data, running, latest_frame

    try:
        print("Starting optimized rendering thread...")

        # Initialize pygame
        if not pygame.get_init():
            pygame.init()

        # Position the main window
        if platform.system() == 'Windows':
            # Windows specific setup
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"{DASHBOARD_WIDTH + 100},50"
        else:
            os.environ['SDL_VIDEO_WINDOW_POS'] = "%d,%d" % (DASHBOARD_WIDTH + 100, 50)

        # Create the main window
        try:
            screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Car Reinforcement Learning")
            clock = pygame.time.Clock()
            print("Main window created successfully")
        except pygame.error as e:
            print(f"Error creating main window: {e}")
            return

        # Create a more detailed car image
        car_img = pygame.Surface((30, 15), pygame.SRCALPHA)

        # Main body
        pygame.draw.rect(car_img, RED, (0, 0, 30, 15), border_radius=3)

        # Windshield
        pygame.draw.polygon(car_img, (100, 200, 255), [(15, 3), (22, 3), (22, 12), (15, 12)])

        # Front headlights
        pygame.draw.rect(car_img, (255, 255, 100), (26, 2, 3, 3))
        pygame.draw.rect(car_img, (255, 255, 100), (26, 10, 3, 3))

        # Rear lights
        pygame.draw.rect(car_img, (200, 0, 0), (1, 2, 2, 3))
        pygame.draw.rect(car_img, (200, 0, 0), (1, 10, 2, 3))

        # Wheels (just hints of them)
        pygame.draw.rect(car_img, (30, 30, 30), (5, -1, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (5, 14, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (20, -1, 5, 2))
        pygame.draw.rect(car_img, (30, 30, 30), (20, 14, 5, 2))

        # Precompute expensive assets
        precomputed_track_surface = None
        track_hash = None

        # Precompute sensors angles
        sensor_angles = [(i * 2 * math.pi / 8) % (2 * math.pi) for i in range(8)]

        # Frame timing variables
        target_fps = 60
        target_frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        # Performance monitoring
        render_times = deque(maxlen=100)

        print("Starting separate render loop")
        while running:
            # Calculate elapsed time since last frame
            current_time = time.time()
            frame_delta = current_time - last_frame_time

            # If it's not time for a new frame yet, yield to other threads
            if frame_delta < target_frame_time:
                # Sleep for a short time to avoid busy waiting
                time.sleep(min(0.001, target_frame_time - frame_delta))
                continue

            # We're ready for a new frame
            start_time = time.time()
            last_frame_time = current_time

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Get frame data safely without blocking
            local_frame_data = get_frame_data()

            # Background - track surface
            # Only recompute track surface if needed
            if local_frame_data and local_frame_data.get('walls'):
                walls = local_frame_data.get('walls')
                current_track_hash = hash(str(walls)) if walls else None

                if precomputed_track_surface is None or track_hash != current_track_hash:
                    # Track surface needs redrawing
                    track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

                    if walls:
                        # Fill the track area with a dark gray
                        pygame.draw.rect(track_surface, (30, 30, 30), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

                        # Draw the actual track in a slightly lighter gray
                        if len(walls) >= 32:  # If using oval track with inner/outer walls
                            # Extract points from walls for the outer and inner boundaries
                            outer_points = [(walls[i][0], walls[i][1]) for i in range(16)]
                            inner_points = [(walls[i][0], walls[i][1]) for i in range(16, 32)]

                            # Draw the track surface
                            pygame.draw.polygon(track_surface, (50, 50, 50), outer_points)
                            pygame.draw.polygon(track_surface, (30, 30, 30), inner_points)

                    precomputed_track_surface = track_surface
                    track_hash = current_track_hash

                # Use the precomputed track surface
                screen.blit(precomputed_track_surface, (0, 0))
            else:
                # Just fill with black if no track data
                screen.fill(BLACK)

            if local_frame_data:
                car = local_frame_data.get('car')
                walls = local_frame_data.get('walls')
                checkpoints = local_frame_data.get('checkpoints')
                reward = local_frame_data.get('reward', 0)
                total_reward = local_frame_data.get('total_reward', 0)
                laps = local_frame_data.get('laps', 0)
                episode = local_frame_data.get('episode', 0)
                info = local_frame_data.get('info', {})

                # Draw walls
                if walls:
                    for wall in walls:
                        pygame.draw.line(screen, WHITE, (wall[0], wall[1]), (wall[2], wall[3]), 2)

                # Draw checkpoints
                if checkpoints:
                    # Draw the track's center line connecting checkpoints
                    if len(checkpoints) > 1:
                        # Create a closed loop by connecting the last checkpoint to the first
                        for i in range(len(checkpoints)):
                            next_i = (i + 1) % len(checkpoints)
                            pygame.draw.line(screen, (80, 80, 80), checkpoints[i], checkpoints[next_i], 1)

                    # Draw the actual checkpoint markers
                    for i, cp in enumerate(checkpoints):
                        # Current checkpoint is green, next checkpoint is yellow, others are brighter gray
                        if car and i == car.checkpoint_idx:
                            checkpoint_color = GREEN
                            radius = 8
                        elif car and i == (car.checkpoint_idx + 1) % len(checkpoints):
                            checkpoint_color = YELLOW
                            radius = 6
                        else:
                            checkpoint_color = (120, 120, 120)  # Brighter gray to be visible
                            radius = 4

                        pygame.draw.circle(screen, checkpoint_color, cp, radius)

                # Draw car
                if car:
                    # Create a rotated copy of the car image
                    angle_degrees = -math.degrees(car.angle)  # Pygame uses opposite rotation direction
                    rotated_car = pygame.transform.rotate(car_img, angle_degrees)
                    # Get the rect of the rotated image and position it at the car's position
                    car_rect = rotated_car.get_rect(center=(car.x, car.y))
                    screen.blit(rotated_car, car_rect.topleft)

                    # Draw sensors
                    for i in range(len(car.sensor_readings)):
                        # Calculate sensor angle
                        sensor_angle = (car.angle + sensor_angles[i]) % (2 * math.pi)
                        # Calculate endpoint based on reading
                        end_x = car.x + math.cos(sensor_angle) * car.sensor_readings[i]
                        end_y = car.y + math.sin(sensor_angle) * car.sensor_readings[i]
                        # Draw line
                        pygame.draw.line(screen, GREEN, (car.x, car.y), (end_x, end_y), 1)

                # Draw stats
                font = pygame.font.SysFont('Arial', 16)
                avg_render_time = sum(render_times) / max(1, len(render_times))

                info_text = [
                    f"Episode: {episode}",
                    f"Reward: {reward:.2f}",
                    f"Total Reward: {total_reward:.2f}",
                    f"Laps: {laps}",
                    f"Speed: {car.speed:.2f}" if car else "Speed: 0.00",
                    f"Distance: {car.total_distance:.2f}" if car else "Distance: 0.00",
                    f"Time Alive: {car.time_alive}" if car else "Time Alive: 0",
                    f"Direction Alignment: {info.get('direction_alignment', 0):.2f}" if info else "Direction Alignment: 0.00",
                    f"FPS: {clock.get_fps():.1f}",
                    f"Render Time: {avg_render_time * 1000:.1f}ms"
                ]

                for i, text in enumerate(info_text):
                    text_surface = font.render(text, True, WHITE)
                    screen.blit(text_surface, (10, 10 + i * 20))

            # Update display
            pygame.display.flip()

            # Record render time for performance monitoring
            render_time = time.time() - start_time
            render_times.append(render_time)

            # Cap at target FPS
            clock.tick(target_fps)
    except Exception as e:
        print(f"Error in rendering thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Rendering thread exiting")
        pygame.quit()


# Dashboard thread for metrics only (separate window mode)
def dashboard_thread():
    global frame_data, running

    try:
        print("Dashboard thread starting with optimizations...")

        # Wait a bit to ensure the main window is created first (if needed)
        time.sleep(0.5)

        # Clear any existing display mode
        if pygame.display.get_surface():
            current_w, current_h = pygame.display.get_surface().get_size()
            print(f"Existing display surface found: {current_w}x{current_h}")

        # Set window position for dashboard
        if platform.system() == 'Windows':
            # Windows implementation
            import ctypes
            user32 = ctypes.windll.user32
            screen_width, screen_height = user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
            os.environ['SDL_VIDEO_WINDOW_POS'] = f"50,50"
            print(f"Setting window position on Windows, screen size: {screen_width}x{screen_height}")
        else:
            # Unix implementation
            os.environ['SDL_VIDEO_WINDOW_POS'] = "50,50"

        # Create dashboard window
        print("Creating dashboard window...")
        dashboard = pygame.display.set_mode((DASHBOARD_WIDTH, DASHBOARD_HEIGHT), pygame.RESIZABLE)
        pygame.display.set_caption("RL Training Dashboard")
        print(f"Dashboard window created: {DASHBOARD_WIDTH}x{DASHBOARD_HEIGHT}")
        clock = pygame.time.Clock()

        # Fonts
        title_font = pygame.font.SysFont('Arial', 20, bold=True)
        regular_font = pygame.font.SysFont('Arial', 14)
        small_font = pygame.font.SysFont('Arial', 12)

        # Dashboard sections with reorganized layout (matching combined rendering)
        sections = {
            'training_stats': {'rect': pygame.Rect(10, 10, 380, 150), 'title': 'Training Statistics'},
            'episode_stats': {'rect': pygame.Rect(400, 10, 380, 150), 'title': 'Current Episode'},
            'reward_plot': {'rect': pygame.Rect(10, 170, 380, 180), 'title': 'Episode Reward History'},
            'avg_reward_plot': {'rect': pygame.Rect(400, 170, 380, 180), 'title': 'Average Reward (100 ep)'},
            'loss_plot': {'rect': pygame.Rect(10, 360, 380, 180), 'title': 'Loss History'},
            'action_plot': {'rect': pygame.Rect(400, 360, 380, 180), 'title': 'Recent Actions'},
            'sensor_plot': {'rect': pygame.Rect(10, 550, 770, 230), 'title': 'Sensor Readings'}
        }

        # For downsampling larger datasets
        max_plot_points = 100

        # Frame timing variables
        target_fps = 30  # Lower FPS for dashboard is fine
        target_frame_time = 1.0 / target_fps
        last_frame_time = time.time()

        # Performance monitoring
        render_times = deque(maxlen=100)

        while running:
            # Calculate elapsed time since last frame
            current_time = time.time()
            frame_delta = current_time - last_frame_time

            # If it's not time for a new frame yet, yield to other threads
            if frame_delta < target_frame_time:
                # Sleep for a short time to avoid busy waiting
                time.sleep(min(0.001, target_frame_time - frame_delta))
                continue

            # We're ready for a new frame
            start_time = time.time()
            last_frame_time = current_time

            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                elif event.type == pygame.VIDEORESIZE:
                    # Resize the dashboard window
                    dashboard = pygame.display.set_mode((event.w, event.h), pygame.RESIZABLE)

            # Clear dashboard
            dashboard.fill(BLACK)

            # Get current metrics and frame data safely
            current_metrics = get_metrics_data()
            current_frame = get_frame_data()

            # Only render if we have valid data
            if current_metrics and current_frame:
                # Draw sections
                for section_id, section in sections.items():
                    # Draw section background
                    pygame.draw.rect(dashboard, DARK_GRAY, section['rect'])
                    pygame.draw.rect(dashboard, LIGHT_GRAY, section['rect'], 1)

                    # Draw section title
                    title_surface = title_font.render(section['title'], True, WHITE)
                    dashboard.blit(title_surface, (section['rect'].x + 10, section['rect'].y + 5))

                    # Draw each section content based on section type
                    if section_id == 'training_stats':
                        # Training statistics in two columns for better readability
                        start_episode = current_metrics.get('start_episode', 0)
                        current_episodes = len(current_metrics.get('episode_rewards', []))
                        total_episodes = start_episode + current_episodes

                        col1_stats = [
                            f"Episodes: {total_episodes}/{MAX_EPISODES}",
                            f"Updates: {current_metrics.get('updates_performed', 0)}",
                            f"Memory: {current_metrics.get('memory_usage', 0)}/{MEMORY_SIZE}",
                            f"Total Laps: {sum(current_metrics.get('episode_laps', [0]))}",
                            f"GPU Util: {current_metrics.get('gpu_util', 0)}%"
                        ]

                        # Format total training time
                        total_time = current_metrics.get('total_training_time', 0)
                        current_time = time.time() - current_metrics.get('start_time', time.time())
                        all_time = total_time + current_time

                        days, remainder = divmod(int(all_time), 86400)
                        hours, remainder = divmod(remainder, 3600)
                        minutes, seconds = divmod(remainder, 60)

                        time_str = ""
                        if days > 0:
                            time_str += f"{days}d "
                        if hours > 0 or days > 0:
                            time_str += f"{hours}h "
                        time_str += f"{minutes}m {seconds}s"

                        col2_stats = [
                            f"Total Train Time: {time_str}",
                            f"Avg Reward: {np.mean(current_metrics.get('episode_rewards', [])[-100:]) if current_metrics.get('episode_rewards', []) else 0:.2f}",
                            f"Avg Ep Time: {current_metrics.get('avg_episode_time', 0):.2f}s",
                            f"Est. Completion: {time.strftime('%H:%M:%S', time.gmtime(current_metrics.get('estimated_completion_time', 0)))}",
                            f"NN Update Time: {current_metrics.get('nn_update_time', 0) * 1000:.1f}ms"
                        ]

                        # Column 1
                        for i, stat in enumerate(col1_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            dashboard.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                        # Column 2
                        for i, stat in enumerate(col2_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            dashboard.blit(stat_surface, (section['rect'].x + 200, section['rect'].y + 30 + i * 18))

                    elif section_id == 'episode_stats':
                        # Current episode statistics
                        col1_stats = [
                            f"Episode: {current_frame.get('episode', 0)}",
                            f"Reward: {current_frame.get('reward', 0):.2f}",
                            f"Total Reward: {current_frame.get('total_reward', 0):.2f}",
                            f"Laps: {current_frame.get('laps', 0)}"
                        ]

                        col2_stats = [
                            f"LR: {current_metrics.get('learning_rate', 0):.6f}",
                            f"Entropy: {current_metrics.get('entropy_coef', 0):.4f}",
                            f"Loss: {current_metrics.get('last_loss', 0):.4f}",
                            f"A/C Loss: {current_metrics.get('actor_loss', 0):.2f}/{current_metrics.get('critic_loss', 0):.2f}"
                        ]

                        # Column 1
                        for i, stat in enumerate(col1_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            dashboard.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                        # Column 2
                        for i, stat in enumerate(col2_stats):
                            stat_surface = regular_font.render(stat, True, WHITE)
                            dashboard.blit(stat_surface, (section['rect'].x + 200, section['rect'].y + 30 + i * 18))

                    elif section_id == 'reward_plot':
                        # Reward history plot
                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Update reward history data
                        if current_metrics.get('episode_rewards', []):
                            reward_history = current_metrics['episode_rewards']

                            # Downsample if too many points
                            if len(reward_history) > max_plot_points:
                                downsample_factor = max(1, len(reward_history) // max_plot_points)
                                reward_history = reward_history[::downsample_factor]

                            # Draw plot
                            draw_line_plot(dashboard, plot_rect, reward_history,
                                           min_val=min(reward_history) if reward_history else 0,
                                           max_val=max(reward_history) if reward_history else 1,
                                           color=GREEN, label="Episode Reward")

                    # Implement other section rendering similarly...
                    # Other visualization sections (use the implementations from combined_rendering_thread)
                    elif section_id == 'avg_reward_plot':
                        # Draw the average reward plot
                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                                section['rect'].width - 20, section['rect'].height - 40)

                        if current_metrics.get('avg_rewards', []):
                            avg_rewards = current_metrics['avg_rewards']

                            # Downsample if needed
                            if len(avg_rewards) > max_plot_points:
                                downsample_factor = max(1, len(avg_rewards) // max_plot_points)
                                avg_rewards = avg_rewards[::downsample_factor]

                            # Get min/max for better visualization
                            min_val = min(avg_rewards) if avg_rewards else 0
                            max_val = max(avg_rewards) if avg_rewards else 1

                            # Add margin for better visibility
                            value_range = max_val - min_val
                            min_val = max(0, min_val - value_range * 0.1)
                            max_val = max_val + value_range * 0.1

                            draw_line_plot(dashboard, plot_rect, avg_rewards,
                                           min_val=min_val, max_val=max_val,
                                           color=CYAN, label="100-Episode Moving Avg")

                    elif section_id == 'action_plot':
                        # Reuse the action plot implementation from the combined renderer
                        plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                                section['rect'].width - 20, section['rect'].height - 40)

                        # Check if we have action data
                        if current_metrics.get('recent_actions', []):
                            # Extract acceleration and steering
                            if len(current_metrics['recent_actions']) > 0 and len(
                                    current_metrics['recent_actions'][0]) >= 2:
                                accel_actions = [a[0] for a in current_metrics['recent_actions']]
                                steer_actions = [a[1] for a in current_metrics['recent_actions']]

                                # Create separate plot areas for acceleration and steering
                                half_height = plot_rect.height // 2
                                upper_rect = pygame.Rect(plot_rect.left, plot_rect.top, plot_rect.width, half_height)
                                lower_rect = pygame.Rect(plot_rect.left, plot_rect.top + half_height, plot_rect.width,
                                                         half_height)

                                # Fill backgrounds
                                pygame.draw.rect(dashboard, (0, 40, 40), upper_rect)
                                pygame.draw.rect(dashboard, (40, 0, 40), lower_rect)

                                # Continue with action plot implementation...
                                # (same as in combined renderer)
                                # Draw zero lines
                                zero_y_accel = upper_rect.top + upper_rect.height // 2
                                zero_y_steer = lower_rect.top + lower_rect.height // 2

                                pygame.draw.line(dashboard, (100, 100, 100),
                                                 (upper_rect.left, zero_y_accel),
                                                 (upper_rect.right, zero_y_accel), 1)
                                pygame.draw.line(dashboard, (100, 100, 100),
                                                 (lower_rect.left, zero_y_steer),
                                                 (lower_rect.right, zero_y_steer), 1)

                                # Add labels
                                label_font = pygame.font.SysFont('Arial', 12)
                                accel_label = label_font.render("Acceleration (Fwd/Rev)", True, CYAN)
                                steer_label = label_font.render("Steering (Left/Right)", True, PURPLE)

                                dashboard.blit(accel_label, (upper_rect.left + 5, upper_rect.top + 5))
                                dashboard.blit(steer_label, (lower_rect.left + 5, lower_rect.top + 5))

                                # Add plotting code for accel_actions and steer_actions
                                # (similar to the implementation in combined renderer)

            # Record render time for performance
            render_time = time.time() - start_time
            render_times.append(render_time)

            # Update display and cap framerate
            pygame.display.flip()
            clock.tick(target_fps)

    except Exception as e:
        print(f"Dashboard thread error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Dashboard thread exiting")