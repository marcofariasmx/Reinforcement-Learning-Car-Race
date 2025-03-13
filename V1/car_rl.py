import pygame
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import time
import math
import os
import pickle
import json
import threading
import copy
from threading import Lock
from collections import deque, namedtuple
from queue import Queue, Empty
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Constants
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 1000
TARGET_FPS = 60  # Target FPS for smooth rendering
MAX_FRAME_TIME = 1.0 / TARGET_FPS  # Maximum time per frame
TRAINING_SPEED = 5  # Training iterations per frame (higher = faster training)
MAX_TRAINING_SPEED = 20  # Maximum allowed training speed
BATCH_SIZE_MULTIPLIER = 2  # Multiply default batch size to process more experiences at once
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
LIGHT_BLUE = (173, 216, 230)

# UI layout constants
INFO_PANEL_WIDTH = 350
INFO_PANEL_HEIGHT = 320  # Increased height for better spacing
STATS_WIDTH = 500  # Reduced from 600 for a smaller stats box
STATS_HEIGHT = 400  # Reduced from 450 for a smaller stats box

# Checkpoint saving
CHECKPOINT_DIR = "checkpoints"
SAVE_FREQUENCY = 10  # Save every N episodes

# RL parameters
BATCH_SIZE = 128
GAMMA = 0.99  # Discount factor
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10
MEMORY_SIZE = 20000
LEARNING_RATE = 0.001

# Raceway parameters
TRACK_WIDTH = 100
CHECKPOINT_RADIUS = 15
NUM_CHECKPOINTS = 20

# Car parameters
CAR_LENGTH = 20
CAR_WIDTH = 10
MAX_SPEED = 5
MAX_STEERING = 0.1
SENSOR_LENGTH = 200
NUM_SENSORS = 8

# Threaded training flag
USE_THREADED_TRAINING = True  # Set to False to disable threaded training

# Experience replay memory
Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
        self.lock = Lock()  # Add thread lock for memory access

    def push(self, *args):
        with self.lock:  # Thread-safe push operation
            self.memory.append(Experience(*args))

    def sample(self, batch_size):
        with self.lock:  # Thread-safe sample operation
            # Create a copy of the memory to avoid mutation during iteration
            memory_snapshot = list(self.memory)
        return random.sample(memory_snapshot, batch_size)

    def __len__(self):
        with self.lock:  # Thread-safe length check
            return len(self.memory)

    def get_serializable_memory(self):
        """Convert memory to a serializable format (lists)"""
        serialized = []
        with self.lock:  # Thread-safe memory access
            memory_snapshot = list(self.memory)

        for experience in memory_snapshot:
            serialized.append({
                'state': experience.state.tolist() if isinstance(experience.state, np.ndarray) else experience.state,
                'action': experience.action,
                'next_state': experience.next_state.tolist() if isinstance(experience.next_state,
                                                                           np.ndarray) else experience.next_state,
                'reward': experience.reward,
                'done': experience.done
            })
        return serialized

    def load_from_serialized(self, serialized_memory):
        """Restore memory from serialized format"""
        with self.lock:  # Thread-safe memory access
            self.memory.clear()
            for exp_dict in serialized_memory:
                state = np.array(exp_dict['state']) if isinstance(exp_dict['state'], list) else exp_dict['state']
                next_state = np.array(exp_dict['next_state']) if isinstance(exp_dict['next_state'], list) else exp_dict[
                    'next_state']
                self.memory.append(Experience(
                    state=state,
                    action=exp_dict['action'],
                    next_state=next_state,
                    reward=exp_dict['reward'],
                    done=exp_dict['done']
                ))


# Neural network for Q-learning
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Car class
class Car:
    def __init__(self, x, y, angle):
        self.x = x
        self.y = y
        self.angle = angle  # In radians
        self.speed = 0
        self.steering = 0
        self.acceleration = 0
        self.sensor_distances = np.zeros(NUM_SENSORS)
        self.checkpoint_reached = 0
        self.alive = True
        self.distance_traveled = 0
        self.last_position = (x, y)
        self.time_alive = 0
        # Added for visual smoothness
        self.prev_x = x
        self.prev_y = y
        self.visual_x = x
        self.visual_y = y
        self.visual_angle = angle
        self.lerp_factor = 0.3  # Visual interpolation factor

    def update_visual_position(self, delta_time):
        # Interpolate visual position for smooth rendering
        self.visual_x += (self.x - self.visual_x) * self.lerp_factor * delta_time * 60
        self.visual_y += (self.y - self.visual_y) * self.lerp_factor * delta_time * 60

        # Handle angle interpolation accounting for circular nature
        angle_diff = self.angle - self.visual_angle
        # Normalize angle difference to [-pi, pi]
        if angle_diff > math.pi:
            angle_diff -= 2 * math.pi
        elif angle_diff < -math.pi:
            angle_diff += 2 * math.pi
        self.visual_angle += angle_diff * self.lerp_factor * delta_time * 60

    def get_corners(self, use_visual=True):
        # Use visual position for rendering, actual position for logic
        x = self.visual_x if use_visual else self.x
        y = self.visual_y if use_visual else self.y
        angle = self.visual_angle if use_visual else self.angle

        cos_angle = math.cos(angle)
        sin_angle = math.sin(angle)

        # Define the car corners relative to center
        corners = [
            (CAR_LENGTH / 2, CAR_WIDTH / 2),
            (CAR_LENGTH / 2, -CAR_WIDTH / 2),
            (-CAR_LENGTH / 2, -CAR_WIDTH / 2),
            (-CAR_LENGTH / 2, CAR_WIDTH / 2)
        ]

        # Rotate and translate corners
        rotated_corners = []
        for dx, dy in corners:
            x_rot = dx * cos_angle - dy * sin_angle
            y_rot = dx * sin_angle + dy * cos_angle
            rotated_corners.append((x + x_rot, y + y_rot))

        return rotated_corners

    def get_sensor_lines(self, use_visual=True):
        # Use visual position for rendering, actual position for logic
        x = self.visual_x if use_visual else self.x
        y = self.visual_y if use_visual else self.y
        angle = self.visual_angle if use_visual else self.angle

        angles = [angle + i * (2 * math.pi / NUM_SENSORS) for i in range(NUM_SENSORS)]
        lines = []

        for i, sensor_angle in enumerate(angles):
            end_x = x + math.cos(sensor_angle) * self.sensor_distances[i]
            end_y = y + math.sin(sensor_angle) * self.sensor_distances[i]
            lines.append(((x, y), (end_x, end_y)))

        return lines

    def update(self, action, track, checkpoints):
        # Store previous position
        self.prev_x = self.x
        self.prev_y = self.y

        # Decode action (0: accelerate, 1: brake, 2: steer left, 3: steer right)
        if action == 0:
            self.acceleration = 0.1
        elif action == 1:
            self.acceleration = -0.1
        elif action == 2:
            self.steering = -MAX_STEERING
        elif action == 3:
            self.steering = MAX_STEERING

        # Update speed
        self.speed += self.acceleration
        self.speed = max(-MAX_SPEED / 2, min(MAX_SPEED, self.speed))

        # Apply friction
        self.speed *= 0.95

        # Update angle based on steering and speed
        self.angle += self.steering * self.speed

        # Update position
        self.last_position = (self.x, self.y)
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Calculate distance traveled
        dx = self.x - self.last_position[0]
        dy = self.y - self.last_position[1]
        self.distance_traveled += math.sqrt(dx * dx + dy * dy)

        # Update sensors
        self.update_sensors(track)

        # Check for checkpoint crossing
        next_checkpoint = (self.checkpoint_reached + 1) % len(checkpoints)
        if self.distance_to_point(checkpoints[next_checkpoint]) < CHECKPOINT_RADIUS:
            self.checkpoint_reached = next_checkpoint
            return 10  # Reward for reaching checkpoint

        # Check for collision with track boundary
        if self.is_collision(track):
            self.alive = False
            return -10  # Penalty for collision

        # Small penalty for time passing (encourages finding efficient path)
        self.time_alive += 1
        return -0.01

    def update_sensors(self, track):
        for i in range(NUM_SENSORS):
            angle = self.angle + i * (2 * math.pi / NUM_SENSORS)
            self.sensor_distances[i] = self.get_distance_to_wall(angle, track)

    def get_distance_to_wall(self, angle, track):
        # Cast a ray from the car's position in the given angle direction
        # and find the distance to the nearest wall
        dx = math.cos(angle)
        dy = math.sin(angle)

        for i in range(1, SENSOR_LENGTH):
            x = int(self.x + i * dx)
            y = int(self.y + i * dy)

            # Check if the point is inside the track
            if not self.is_point_in_track((x, y), track):
                return i

        return SENSOR_LENGTH

    def is_point_in_track(self, point, track):
        # Check if a point is inside the track
        x, y = point
        if x < 0 or x >= SCREEN_WIDTH or y < 0 or y >= SCREEN_HEIGHT:
            return False

        # For a simple track represented as an image, check the color
        if track[int(y), int(x)] == 0:  # Assuming 0 represents wall
            return False

        return True

    def is_collision(self, track):
        corners = self.get_corners(use_visual=False)  # Use actual position for collision detection
        for corner in corners:
            if not self.is_point_in_track(corner, track):
                return True
        return False

    def distance_to_point(self, point):
        return math.sqrt((self.x - point[0]) ** 2 + (self.y - point[1]) ** 2)

    def get_state(self):
        # State includes normalized sensor distances, speed, and angle
        state = np.append(self.sensor_distances / SENSOR_LENGTH,
                          [self.speed / MAX_SPEED,
                           math.sin(self.angle),
                           math.cos(self.angle)])
        return state


# Environment class
class RaceEnvironment:
    def __init__(self, load_checkpoint=None):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
        pygame.display.set_caption("Reinforcement Learning for Racing Cars")
        self.clock = pygame.time.Clock()

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(CHECKPOINT_DIR):
            os.makedirs(CHECKPOINT_DIR)

        self.track = self.create_track()
        self.checkpoints = self.create_checkpoints()

        # Create the car
        self.car = Car(self.checkpoints[0][0], self.checkpoints[0][1], 0)

        # Initialize RL components
        self.state_size = NUM_SENSORS + 3  # Sensor readings + speed + sin/cos of angle
        self.action_size = 4  # Accelerate, brake, steer left, steer right

        self.policy_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net = DQN(self.state_size, self.action_size).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999))
        self.memory = ReplayMemory(MEMORY_SIZE)

        self.epsilon = EPSILON_START
        self.episode = 0
        self.steps_done = 0
        self.best_reward = float('-inf')
        self.fps = 0  # For tracking performance
        self.actual_fps = 0  # More accurate FPS counter for display

        # Statistics tracking
        self.episode_rewards = []
        self.episode_lengths = []
        self.checkpoint_counts = []
        self.epsilon_values = []

        # For visualization
        self.stats_fig = Figure(figsize=(5, 4), dpi=100)

        # For threaded training
        self.training_thread = None
        self.training_queue = Queue()
        self.result_queue = Queue()
        self.training_active = False
        self.training_paused = False
        self.last_state = None
        self.total_reward = 0
        self.episode_length = 0
        self.render_stats_cache = None
        self.last_stats_update = 0

        # Render state
        self.track_surface = None  # Cache the track surface
        self.next_render_time = 0
        self.delta_time = 0
        self.last_frame_time = time.time()
        self.frame_times = deque(maxlen=30)  # For FPS calculation
        self.fps_lock = Lock()  # Lock for FPS calculation

        # Load checkpoint if specified
        if load_checkpoint:
            self.load_checkpoint(load_checkpoint)

        # Prerender the track
        self.prerender_track()

    def prerender_track(self):
        """Prerender the track to a surface for faster rendering"""
        self.track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                if self.track[y, x] == 0:
                    self.track_surface.set_at((x, y), (80, 80, 80))  # Darker gray for track boundaries
                else:
                    self.track_surface.set_at((x, y), (220, 220, 220))  # Light gray for track

    def create_track(self):
        # Create a simple track as a binary matrix (0 = wall, 1 = track)
        track = np.ones((SCREEN_HEIGHT, SCREEN_WIDTH))

        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        major_axis = 400  # Fixed size for track major axis
        minor_axis = 300  # Fixed size for track minor axis

        for y in range(SCREEN_HEIGHT):
            for x in range(SCREEN_WIDTH):
                # Calculate distance from center
                dx = x - center_x
                dy = y - center_y

                # Compute normalized distance for ellipse
                normalized_dist = math.sqrt((dx / major_axis) ** 2 + (dy / minor_axis) ** 2)

                # Track boundaries (outer and inner)
                if normalized_dist > 0.9 or normalized_dist < 0.5:
                    track[y, x] = 0

        return track

    def create_checkpoints(self):
        # Create checkpoints along the track with fixed distances from center
        checkpoints = []
        center_x, center_y = SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2
        major_axis = 400 * 0.7  # 70% of the track major axis (positioned between inner and outer walls)
        minor_axis = 300 * 0.7  # 70% of the track minor axis

        for i in range(NUM_CHECKPOINTS):
            angle = 2 * math.pi * i / NUM_CHECKPOINTS
            x = center_x + major_axis * math.cos(angle)
            y = center_y + minor_axis * math.sin(angle)
            checkpoints.append((x, y))

        return checkpoints

    def reset(self):
        # Reset the car
        start_checkpoint = self.checkpoints[0]
        self.car = Car(start_checkpoint[0], start_checkpoint[1], 0)
        return self.car.get_state()

    def step(self, action):
        # Take an action and return new state, reward, done
        reward = self.car.update(action, self.track, self.checkpoints)
        done = not self.car.alive
        return self.car.get_state(), reward, done

    def select_action(self, state):
        # Epsilon-greedy action selection
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state_tensor)
                return q_values.max(1)[1].item()
        else:
            return random.randrange(self.action_size)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        # Use larger batch size for faster learning
        actual_batch_size = min(BATCH_SIZE * BATCH_SIZE_MULTIPLIER, len(self.memory))
        experiences = self.memory.sample(actual_batch_size)
        batch = Experience(*zip(*experiences))

        non_final_mask = torch.tensor([not d for d in batch.done], dtype=torch.bool).to(device)

        # Fix for slow tensor creation - convert list to numpy array first
        non_final_states_list = [s for s, d in zip(batch.next_state, batch.done) if not d]
        if non_final_states_list:
            non_final_next_states = torch.FloatTensor(np.array(non_final_states_list)).to(device)
        else:
            non_final_next_states = torch.FloatTensor([]).to(device)

        # Convert other lists to numpy arrays before creating tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(np.array(batch.action)).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(np.array(batch.reward)).to(device)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states
        next_state_values = torch.zeros(actual_batch_size, device=device)
        with torch.no_grad():
            if non_final_next_states.size(0) > 0:  # Check if tensor is not empty
                next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0]

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Use a larger gradient clipping value for faster updates
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-2, 2)
        self.optimizer.step()

    def update_stats(self, total_reward, episode_length, checkpoints_reached):
        """Thread-safe update of statistics"""
        try:
            # Create thread-safe copies for updating
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(episode_length)
            self.checkpoint_counts.append(checkpoints_reached)
            self.epsilon_values.append(self.epsilon)
        except Exception as e:
            print(f"Error updating stats: {e}")

    def draw_track(self):
        # Draw the pre-rendered track
        self.screen.blit(self.track_surface, (0, 0))

        # Draw checkpoints with better visibility
        for i, checkpoint in enumerate(self.checkpoints):
            if i == (self.car.checkpoint_reached + 1) % len(self.checkpoints):
                # Next checkpoint is bright green with thicker outline
                pygame.draw.circle(self.screen, (0, 200, 0), (int(checkpoint[0]), int(checkpoint[1])),
                                   CHECKPOINT_RADIUS, 3)
                # Add a smaller inner circle for better visibility
                pygame.draw.circle(self.screen, (100, 255, 100), (int(checkpoint[0]), int(checkpoint[1])),
                                   CHECKPOINT_RADIUS // 2, 0)
            else:
                # Regular checkpoints are yellow with thinner outline
                pygame.draw.circle(self.screen, (200, 200, 0), (int(checkpoint[0]), int(checkpoint[1])),
                                   CHECKPOINT_RADIUS, 2)

    def draw_car(self):
        # Update visual position for smooth rendering
        self.car.update_visual_position(self.delta_time)

        # Draw the car with better visuals
        corners = self.car.get_corners(use_visual=True)  # Use interpolated visual position

        # Draw a filled car shape in bright red with dark outline
        pygame.draw.polygon(self.screen, (220, 30, 30), corners, 0)  # Fill
        pygame.draw.polygon(self.screen, (100, 10, 10), corners, 2)  # Outline

        # Draw direction indicator (front of car)
        front_x = self.car.visual_x + math.cos(self.car.visual_angle) * (CAR_LENGTH / 2)
        front_y = self.car.visual_y + math.sin(self.car.visual_angle) * (CAR_LENGTH / 2)
        pygame.draw.circle(self.screen, (255, 255, 255), (int(front_x), int(front_y)), 3, 0)

        # Draw sensors with enhanced visibility
        sensor_lines = self.car.get_sensor_lines(use_visual=True)
        for i, line in enumerate(sensor_lines):
            # Gradient color based on sensor distance (red to green)
            dist_ratio = self.car.sensor_distances[i] / SENSOR_LENGTH
            r = int(255 * (1 - dist_ratio))
            g = int(200 * dist_ratio)
            b = 50
            sensor_color = (r, g, b)

            # Draw thicker lines for better visibility with transparency effect
            pygame.draw.line(self.screen, sensor_color, line[0], line[1], 2)
            # Draw small circle at end of each sensor
            pygame.draw.circle(self.screen, sensor_color, (int(line[1][0]), int(line[1][1])), 3, 0)

    def draw_stats(self):
        # Only update the stats figure periodically to improve performance
        current_time = time.time()
        if self.render_stats_cache is None or current_time - self.last_stats_update > 1.0:
            try:
                # Make the figure smaller and more compact
                # Reduce the size from the original
                smaller_width = int(STATS_WIDTH * 0.7)
                smaller_height = int(STATS_HEIGHT * 0.7)

                self.stats_fig = Figure(figsize=(smaller_width / 100, smaller_height / 100), dpi=100)
                self.stats_fig.set_facecolor((0.9, 0.9, 0.9))  # Light gray background

                if len(self.episode_rewards) > 0:
                    # Compact spacing between subplots to make it smaller but still readable
                    subplot_params = {
                        'left': 0.15,  # Left margin
                        'right': 0.9,  # Right margin
                        'bottom': 0.15,  # Bottom margin
                        'top': 0.9,  # Top margin
                        'wspace': 0.4,  # Horizontal space between plots
                        'hspace': 0.6  # Vertical space between plots
                    }
                    self.stats_fig.subplots_adjust(**subplot_params)

                    # Smaller font sizes for a more compact display
                    title_size = 12
                    tick_size = 8

                    # Reward plot
                    ax1 = self.stats_fig.add_subplot(221)
                    ax1.plot(self.episode_rewards, color='green', linewidth=2)
                    ax1.set_title('Episode Rewards', fontsize=title_size, fontweight='bold')
                    ax1.set_facecolor((0.95, 0.95, 0.95))
                    ax1.grid(True, linestyle='--', alpha=0.7)
                    ax1.tick_params(labelsize=tick_size)

                    # Episode length plot
                    ax2 = self.stats_fig.add_subplot(222)
                    ax2.plot(self.episode_lengths, color='blue', linewidth=2)
                    ax2.set_title('Episode Lengths', fontsize=title_size, fontweight='bold')
                    ax2.set_facecolor((0.95, 0.95, 0.95))
                    ax2.grid(True, linestyle='--', alpha=0.7)
                    ax2.tick_params(labelsize=tick_size)

                    # Checkpoints plot
                    ax3 = self.stats_fig.add_subplot(223)
                    ax3.plot(self.checkpoint_counts, color='purple', linewidth=2)
                    ax3.set_title('Checkpoints', fontsize=title_size, fontweight='bold')
                    ax3.set_facecolor((0.95, 0.95, 0.95))
                    ax3.grid(True, linestyle='--', alpha=0.7)
                    ax3.tick_params(labelsize=tick_size)

                    # Epsilon plot
                    ax4 = self.stats_fig.add_subplot(224)
                    ax4.plot(self.epsilon_values, color='red', linewidth=2)
                    ax4.set_title('Epsilon', fontsize=title_size, fontweight='bold')
                    ax4.set_facecolor((0.95, 0.95, 0.95))
                    ax4.grid(True, linestyle='--', alpha=0.7)
                    ax4.tick_params(labelsize=tick_size)

                    # Convert figure to pygame surface
                    canvas = FigureCanvas(self.stats_fig)
                    canvas.draw()
                    renderer = canvas.get_renderer()
                    raw_data = renderer.tostring_rgb()
                    size = canvas.get_width_height()

                    surf = pygame.image.fromstring(raw_data, size, "RGB")
                    self.render_stats_cache = surf
                    self.last_stats_update = current_time
            except Exception as e:
                print(f"Error drawing stats: {e}")
                return  # Skip if there's an error

        # Position the stats in the bottom right corner with appropriate margins
        if self.render_stats_cache:
            stats_width = self.render_stats_cache.get_width()
            stats_height = self.render_stats_cache.get_height()

            # Bottom right corner positioning with margins
            stats_x = SCREEN_WIDTH - stats_width - 40  # 40px margin from right edge
            stats_y = SCREEN_HEIGHT - stats_height - 30  # 30px margin from bottom edge

            # Semi-transparent background for better visibility
            bg_surface = pygame.Surface((stats_width + 10, stats_height + 10))
            bg_surface.set_alpha(180)
            bg_surface.fill((240, 240, 240))
            self.screen.blit(bg_surface, (stats_x - 5, stats_y - 5))

            # Add a border around the stats for better separation
            pygame.draw.rect(self.screen, (50, 50, 50),
                             (stats_x - 5, stats_y - 5,
                              stats_width + 10,
                              stats_height + 10), 2)

            self.screen.blit(self.render_stats_cache, (stats_x, stats_y))

    def display_info(self):
        """Improved info panel display with proper spacing"""
        try:
            # Position the info panel in the top-right corner
            panel_x = SCREEN_WIDTH - INFO_PANEL_WIDTH - 40
            panel_y = 40

            # Semi-transparent background
            info_panel = pygame.Surface((INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT))
            info_panel.set_alpha(220)
            info_panel.fill((240, 240, 240))  # Light gray background
            self.screen.blit(info_panel, (panel_x, panel_y))

            # Add border
            pygame.draw.rect(self.screen, (100, 100, 100),
                             (panel_x, panel_y, INFO_PANEL_WIDTH, INFO_PANEL_HEIGHT), 2)

            # Fonts
            title_font = pygame.font.SysFont('Arial', 26, bold=True)
            info_font = pygame.font.SysFont('Arial', 24)
            perf_font = pygame.font.SysFont('Arial', 24, bold=True)
            help_font = pygame.font.SysFont('Arial', 22, italic=True)

            # Title
            title_text = title_font.render('TRAINING INFO', True, (50, 50, 150))
            self.screen.blit(title_text, (panel_x + 20, panel_y + 15))

            # Calculate lines based on available episode data
            line_y = panel_y + 55  # Starting y-position
            line_height = 35  # Consistent spacing between lines

            if len(self.episode_rewards) > 0:
                # Training metrics
                text_episode = info_font.render(f'Episode: {self.episode}', True, (0, 0, 0))
                self.screen.blit(text_episode, (panel_x + 20, line_y))
                line_y += line_height

                text_reward = info_font.render(f'Last Reward: {self.episode_rewards[-1]:.2f}', True, (0, 100, 0))
                self.screen.blit(text_reward, (panel_x + 20, line_y))
                line_y += line_height

                text_best = info_font.render(f'Best Reward: {self.best_reward:.2f}', True, (150, 0, 0))
                self.screen.blit(text_best, (panel_x + 20, line_y))
                line_y += line_height

                text_checkpoints = info_font.render(f'Checkpoints: {self.checkpoint_counts[-1]}', True, (0, 0, 0))
                self.screen.blit(text_checkpoints, (panel_x + 20, line_y))
                line_y += line_height

                text_epsilon = info_font.render(f'Epsilon: {self.epsilon:.4f}', True, (0, 0, 0))
                self.screen.blit(text_epsilon, (panel_x + 20, line_y))
                line_y += line_height
            else:
                # Starting message
                text_start = info_font.render('Training starting...', True, (0, 0, 0))
                self.screen.blit(text_start, (panel_x + 20, line_y))
                line_y += line_height

            # Performance info
            text_device = info_font.render(f'Device: {device}', True, (100, 0, 100))
            self.screen.blit(text_device, (panel_x + 20, line_y))
            line_y += line_height

            text_speed = perf_font.render(f'Speed: {TRAINING_SPEED}x', True, (0, 0, 150))
            self.screen.blit(text_speed, (panel_x + 20, line_y))
            line_y += line_height

            # FPS display
            if hasattr(self, 'actual_fps') and self.actual_fps > 0:
                text_fps = perf_font.render(f'FPS: {self.actual_fps:.1f}', True, (150, 50, 50))
                self.screen.blit(text_fps, (panel_x + 20, line_y))
                line_y += line_height

            # Separator line
            separator_y = panel_y + INFO_PANEL_HEIGHT - 35
            pygame.draw.line(self.screen, (180, 180, 180),
                             (panel_x + 10, separator_y),
                             (panel_x + INFO_PANEL_WIDTH - 10, separator_y), 2)

            # Controls help
            text_controls = help_font.render('S=Save, ↑↓=Speed, P=Pause', True, (100, 100, 100))
            self.screen.blit(text_controls, (panel_x + 20, panel_y + INFO_PANEL_HEIGHT - 30))

        except Exception as e:
            print(f"Error in display_info: {e}")

    def save_checkpoint(self, episode, total_reward, is_best=False):
        """Save training state to checkpoint file"""
        try:
            checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_ep{episode}.pt")
            best_path = os.path.join(CHECKPOINT_DIR, "best_model.pt")

            # Prepare checkpoint data
            checkpoint = {
                'episode': episode,
                'policy_net': self.policy_net.state_dict(),
                'target_net': self.target_net.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epsilon': self.epsilon,
                'memory': self.memory.get_serializable_memory(),
                'episode_rewards': copy.deepcopy(self.episode_rewards),
                'episode_lengths': copy.deepcopy(self.episode_lengths),
                'checkpoint_counts': copy.deepcopy(self.checkpoint_counts),
                'epsilon_values': copy.deepcopy(self.epsilon_values),
                'best_reward': self.best_reward
            }

            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)
            print(f"Checkpoint saved at episode {episode}")

            # Save best model
            if is_best:
                torch.save(checkpoint, best_path)
                print(f"Best model saved with reward {total_reward:.2f}")
        except Exception as e:
            print(f"Error saving checkpoint: {e}")

    def load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint file"""
        if not os.path.exists(checkpoint_path):
            print(f"Checkpoint file {checkpoint_path} not found")
            return False

        print(f"Loading checkpoint from {checkpoint_path}")

        # Load checkpoint file
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Load model and optimizer state
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])

        # Load training state
        self.episode = checkpoint['episode']
        self.epsilon = checkpoint['epsilon']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']
        self.checkpoint_counts = checkpoint['checkpoint_counts']
        self.epsilon_values = checkpoint['epsilon_values']
        self.best_reward = checkpoint['best_reward']

        # Load memory
        self.memory.load_from_serialized(checkpoint['memory'])

        print(f"Loaded checkpoint from episode {self.episode}")
        return True

    def training_worker(self):
        """Worker function that runs in a separate thread for training"""
        print("Training thread started")

        # Training loop
        while self.training_active:
            if self.training_paused:
                time.sleep(0.01)  # Sleep briefly to avoid busy-waiting
                continue

            try:
                # Get training task from queue with timeout
                task = self.training_queue.get(timeout=0.1)

                if task['command'] == 'step':
                    state = task['state']
                    # Select and perform action
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)

                    # Store transition in memory
                    self.memory.push(state, action, next_state, reward, done)

                    # Optimize model
                    self.optimize_model()

                    # Send results back to main thread
                    self.result_queue.put({
                        'command': 'step_result',
                        'next_state': next_state,
                        'reward': reward,
                        'done': done,
                        'action': action
                    })

                elif task['command'] == 'update_target':
                    # Update target network
                    self.target_net.load_state_dict(self.policy_net.state_dict())
                    self.result_queue.put({'command': 'target_updated'})

                elif task['command'] == 'reset':
                    # Reset environment
                    new_state = self.reset()
                    self.result_queue.put({
                        'command': 'reset_result',
                        'state': new_state
                    })

                elif task['command'] == 'exit':
                    break

            except Empty:
                # Queue was empty, just continue
                pass
            except Exception as e:
                print(f"Error in training thread: {e}")
                self.result_queue.put({
                    'command': 'error',
                    'message': str(e)
                })

        print("Training thread stopped")

    def start_training_thread(self):
        """Start the training thread"""
        if self.training_thread is None or not self.training_thread.is_alive():
            self.training_active = True
            self.training_thread = threading.Thread(target=self.training_worker)
            self.training_thread.daemon = True  # Thread will exit when main program exits
            self.training_thread.start()
            print("Training thread started")

    def stop_training_thread(self):
        """Stop the training thread gracefully"""
        if self.training_thread and self.training_thread.is_alive():
            self.training_queue.put({'command': 'exit'})
            self.training_active = False
            self.training_thread.join(timeout=1.0)  # Wait for thread to finish
            print("Training thread stopped")

    def toggle_pause(self):
        """Toggle pause state for training"""
        self.training_paused = not self.training_paused
        print(f"Training {'paused' if self.training_paused else 'resumed'}")

    def threaded_train(self, num_episodes=1000, start_episode=0):
        """Main training loop that uses a separate thread for RL computations"""
        global TRAINING_SPEED  # Global declaration moved to beginning of function

        self.episode = start_episode
        self.start_training_thread()

        # Send initial reset command
        self.training_queue.put({'command': 'reset'})

        # Wait for reset result
        while True:
            try:
                result = self.result_queue.get(timeout=1.0)
                if result['command'] == 'reset_result':
                    self.last_state = result['state']
                    break
            except Empty:
                print("Waiting for environment reset...")

        running = True
        self.total_reward = 0
        self.episode_length = 0
        waiting_for_result = False

        # For FPS tracking
        frame_start_time = time.time()
        frame_count = 0

        while running and self.episode < num_episodes:
            current_time = time.time()
            self.delta_time = current_time - self.last_frame_time
            self.last_frame_time = current_time

            # Update FPS calculation
            with self.fps_lock:
                self.frame_times.append(self.delta_time)
                if len(self.frame_times) > 0:
                    # Make a copy to avoid modification during iteration
                    times_copy = list(self.frame_times)
                    avg_frame_time = sum(times_copy) / len(times_copy)
                    self.actual_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0

            # Process events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:  # Save checkpoint
                        self.save_checkpoint(self.episode, self.total_reward)
                    elif event.key == pygame.K_UP:  # Increase training speed
                        TRAINING_SPEED = min(MAX_TRAINING_SPEED, TRAINING_SPEED + 5)
                        print(f"Training speed: {TRAINING_SPEED}x")
                    elif event.key == pygame.K_DOWN:  # Decrease training speed
                        TRAINING_SPEED = max(1, TRAINING_SPEED - 5)
                        print(f"Training speed: {TRAINING_SPEED}x")
                    elif event.key == pygame.K_p:  # Pause/resume training
                        self.toggle_pause()

            if not running:
                break

            # Check for results from training thread
            if waiting_for_result:
                try:
                    result = self.result_queue.get_nowait()
                    if result['command'] == 'step_result':
                        # Process the result
                        next_state = result['next_state']
                        reward = result['reward']
                        done = result['done']

                        self.total_reward += reward
                        self.episode_length += 1
                        self.last_state = next_state

                        if done or self.episode_length >= 1000:
                            # End of episode
                            print(f"Episode {self.episode}: Reward={self.total_reward:.2f}, "
                                  f"Length={self.episode_length}, Checkpoints={self.car.checkpoint_reached}, "
                                  f"Epsilon={self.epsilon:.4f}")

                            # Update statistics
                            self.update_stats(self.total_reward, self.episode_length, self.car.checkpoint_reached)

                            # Decrease epsilon
                            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

                            # Save checkpoint periodically
                            if self.episode % SAVE_FREQUENCY == 0:
                                self.save_checkpoint(self.episode, self.total_reward)

                            # Save best model
                            if self.total_reward > self.best_reward:
                                self.best_reward = self.total_reward
                                self.save_checkpoint(self.episode, self.total_reward, is_best=True)

                            # Update target network
                            if self.episode % TARGET_UPDATE == 0:
                                self.training_queue.put({'command': 'update_target'})

                            # Reset for next episode
                            self.episode += 1
                            self.total_reward = 0
                            self.episode_length = 0

                            # Reset environment for next episode
                            self.training_queue.put({'command': 'reset'})
                            waiting_for_result = False
                        else:
                            # Continue with next step
                            waiting_for_result = False

                    elif result['command'] == 'reset_result':
                        self.last_state = result['state']
                        waiting_for_result = False

                    elif result['command'] == 'error':
                        print(f"Error from training thread: {result['message']}")
                        waiting_for_result = False

                except Empty:
                    # No result yet, continue rendering
                    pass

            # Send step command to training thread if not already waiting
            if not waiting_for_result and not self.training_paused:
                # Send multiple training steps based on TRAINING_SPEED
                for _ in range(TRAINING_SPEED):
                    self.training_queue.put({
                        'command': 'step',
                        'state': self.last_state
                    })
                    waiting_for_result = True
                    break  # Only send one step command at a time

            # Always render at consistent FPS regardless of training state
            self.render()

            # Maintain target FPS
            frame_time = time.time() - frame_start_time
            if frame_time < MAX_FRAME_TIME:
                time.sleep(MAX_FRAME_TIME - frame_time)

            # Update frame tracking
            frame_count += 1
            if frame_count >= 30:  # Update FPS display every 30 frames
                self.fps = frame_count / (time.time() - frame_start_time)
                frame_start_time = time.time()
                frame_count = 0

        # Clean up
        self.stop_training_thread()

    def train(self, num_episodes=1000, start_episode=0):
        """Main training loop with fixed time step for consistent FPS"""
        global TRAINING_SPEED  # Global declaration at beginning of function

        if USE_THREADED_TRAINING:
            return self.threaded_train(num_episodes, start_episode)

        if start_episode > 0:
            self.episode = start_episode

        # For tracking FPS
        last_fps_update = time.time()
        frame_count = 0

        # Enable PyTorch optimizations
        torch.backends.cudnn.benchmark = True

        for ep in range(self.episode, num_episodes):
            self.episode = ep

            # Reset environment
            state = self.reset()
            total_reward = 0
            episode_length = 0

            # Time tracking for fixed time step
            last_time = time.time()
            accumulated_time = 0.0

            # Training state
            done = False
            training_debt = 0  # Track how many training iterations we owe

            while not done:
                # Calculate time delta and accumulate time
                current_time = time.time()
                frame_time = current_time - last_time
                last_time = current_time
                self.delta_time = frame_time  # Store for smooth animations

                # Accumulate time and add to training debt
                accumulated_time += frame_time
                training_debt += TRAINING_SPEED  # Add desired training steps per frame

                # Always process events to keep the UI responsive
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        # Save before exit
                        self.save_checkpoint(self.episode, total_reward)
                        pygame.quit()
                        return
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_s:  # Save checkpoint
                            self.save_checkpoint(self.episode, total_reward)
                        elif event.key == pygame.K_UP:  # Increase speed
                            TRAINING_SPEED = min(MAX_TRAINING_SPEED, TRAINING_SPEED + 5)
                            print(f"Training speed: {TRAINING_SPEED}x")
                        elif event.key == pygame.K_DOWN:  # Decrease speed
                            TRAINING_SPEED = max(1, TRAINING_SPEED - 5)
                            print(f"Training speed: {TRAINING_SPEED}x")

                # Process training based on accumulated debt
                training_steps = min(int(training_debt), 20)  # Limit max steps per frame
                training_debt -= training_steps

                for _ in range(training_steps):
                    # Select and perform action
                    action = self.select_action(state)
                    next_state, reward, done = self.step(action)
                    total_reward += reward

                    # Store transition in memory
                    self.memory.push(state, action, next_state, reward, done)

                    # Move to next state
                    state = next_state

                    # Optimize model
                    self.optimize_model()

                    episode_length += 1

                    # Break if episode is done
                    if done or episode_length >= 1000:
                        break

                # Update FPS counter
                frame_count += 1
                if current_time - last_fps_update > 1.0:  # Update every second
                    self.actual_fps = frame_count / (current_time - last_fps_update)
                    frame_count = 0
                    last_fps_update = current_time

                # Always render to maintain consistent FPS
                self.render()

                # Maintain target FPS
                frame_end_time = time.time()
                frame_duration = frame_end_time - current_time
                if frame_duration < MAX_FRAME_TIME:
                    time.sleep(MAX_FRAME_TIME - frame_duration)

                if done:
                    break

            # Update target network
            if self.episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Decrease epsilon
            self.epsilon = max(EPSILON_END, self.epsilon * EPSILON_DECAY)

            # Update statistics
            self.update_stats(total_reward, episode_length, self.car.checkpoint_reached)

            print(f"Episode {self.episode}: Reward={total_reward:.2f}, Length={episode_length}, "
                  f"Checkpoints={self.car.checkpoint_reached}, Epsilon={self.epsilon:.4f}")

            # Save checkpoint periodically
            if self.episode % SAVE_FREQUENCY == 0:
                self.save_checkpoint(self.episode, total_reward)

            # Save best model
            if total_reward > self.best_reward:
                self.best_reward = total_reward
                self.save_checkpoint(self.episode, total_reward, is_best=True)

    def render(self):
        """Render the current state of the environment"""
        try:
            # Set background color
            self.screen.fill((210, 210, 210))

            # Draw track and car
            self.draw_track()
            self.draw_car()

            # Draw title at top
            title_font = pygame.font.SysFont('Arial', 36, bold=True)
            title_text = title_font.render("2D Car Reinforcement Learning", True, (20, 20, 120))
            title_rect = title_text.get_rect(centerx=SCREEN_WIDTH // 2, y=30)

            # Add title background
            title_bg = pygame.Surface((title_rect.width + 30, title_rect.height + 16))
            title_bg.set_alpha(230)
            title_bg.fill((240, 240, 240))
            self.screen.blit(title_bg, (title_rect.x - 15, title_rect.y - 8))
            self.screen.blit(title_text, title_rect)

            # Add border around title
            pygame.draw.rect(self.screen, (100, 100, 150),
                             (title_rect.x - 15, title_rect.y - 8,
                              title_rect.width + 30, title_rect.height + 16), 2)

            # Draw info panel
            self.display_info()

            # Draw statistics every 10 episodes
            if len(self.episode_rewards) >= 10:
                self.draw_stats()

            # Update display
            pygame.display.flip()

        except Exception as e:
            print(f"Error in render: {e}")


# Run the simulation
if __name__ == "__main__":
    import argparse
    import os.path

    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Car Racing with Reinforcement Learning')
    parser.add_argument('--load', type=str, help='Load checkpoint file')
    parser.add_argument('--episodes', type=int, default=1000, help='Number of episodes to train')
    parser.add_argument('--speed', type=int, default=5, help='Training speed multiplier (1-20)')
    parser.add_argument('--no-threaded', action='store_true', help='Disable threaded training')
    parser.add_argument('--batch-size-mult', type=int, default=2, help='Batch size multiplier')
    parser.add_argument('--fast', action='store_true', help='Enable faster training mode (speed=10)')
    parser.add_argument('--turbo', action='store_true', help='Enable turbo training mode (speed=20)')
    parser.add_argument('--fps', type=int, default=60, help='Target FPS (30-120)')
    parser.add_argument('--no-best', action='store_true', help='Do not auto-load best model')
    args = parser.parse_args()

    # Set training speed and render frequency
    if args.turbo:
        TRAINING_SPEED = 20
        BATCH_SIZE_MULTIPLIER = 4
    elif args.fast:
        TRAINING_SPEED = 10
        BATCH_SIZE_MULTIPLIER = 3
    else:
        TRAINING_SPEED = max(1, min(MAX_TRAINING_SPEED, args.speed))
        BATCH_SIZE_MULTIPLIER = max(1, args.batch_size_mult)

    # Set target FPS
    TARGET_FPS = max(30, min(120, args.fps))
    MAX_FRAME_TIME = 1.0 / TARGET_FPS

    # Set threaded training flag
    USE_THREADED_TRAINING = not args.no_threaded

    print(f"Training Speed: {TRAINING_SPEED}x | Target FPS: {TARGET_FPS} | " +
          f"Threaded: {USE_THREADED_TRAINING} | Batch Size Mult: {BATCH_SIZE_MULTIPLIER}x")
    print(f"Device: {device}")

    # Ensure Pygame is initialized
    if pygame.get_init() == False:
        pygame.init()

    # Set PyTorch optimization flags for better performance
    if torch.cuda.is_available():
        # CUDA optimization flags
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

        # Print CUDA memory info
        print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f}GB")
        torch.cuda.empty_cache()

    # Create environment and load checkpoint if specified
    try:
        # Check if specific checkpoint was requested
        if args.load:
            env = RaceEnvironment(load_checkpoint=args.load)
            start_episode = env.episode + 1
            print(f"Loaded checkpoint from episode {env.episode} (as specified)")
        # Auto-load best model if available and not disabled
        elif not args.no_best and os.path.exists(os.path.join(CHECKPOINT_DIR, "best_model.pt")):
            env = RaceEnvironment(load_checkpoint=os.path.join(CHECKPOINT_DIR, "best_model.pt"))
            start_episode = env.episode + 1
            print(f"Automatically loaded best model from episode {env.episode} with reward {env.best_reward:.2f}")
        else:
            env = RaceEnvironment()
            start_episode = 0
            print("Starting new training session (no checkpoint loaded)")

        env.train(num_episodes=args.episodes, start_episode=start_episode)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        if hasattr(env, 'episode_rewards') and len(env.episode_rewards) > 0:
            env.save_checkpoint(env.episode, env.episode_rewards[-1])
    except Exception as e:
        print(f"\nAn error occurred: {e}")
        # Try to save checkpoint even if there's an error
        if hasattr(env, 'episode_rewards') and len(env.episode_rewards) > 0:
            env.save_checkpoint(env.episode, env.episode_rewards[-1])
            print("Checkpoint saved despite error")
    finally:
        # Clean up pygame
        if pygame.get_init():
            pygame.quit()
        print("\nTraining session ended.")