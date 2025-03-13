#!/usr/bin/env python3
import os
import math
import random
import time
import threading
import queue
import sys
import numpy as np
from collections import deque
import pygame
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import argparse

# Set environment variables to help with pygame display issues
os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Platform-specific settings
import platform

if platform.system() == 'Linux':
    # For Linux systems
    os.environ['SDL_VIDEODRIVER'] = 'x11'
elif platform.system() == 'Windows':
    # For Windows systems - don't set SDL_VIDEODRIVER to let pygame auto-detect
    pass
elif platform.system() == 'Darwin':
    # For macOS systems
    os.environ['SDL_VIDEODRIVER'] = 'cocoa'

# Try to avoid pygame window creation errors
if 'DISPLAY' not in os.environ and platform.system() == 'Linux':
    os.environ['DISPLAY'] = ':0'

# Print versions for debugging
print(f"Python version: {sys.version}")
print(f"Pygame version: {pygame.version.ver}")
print(f"PyTorch version: {torch.__version__}")
print(f"NumPy version: {np.__version__}")

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
SCREEN_WIDTH = 1200
SCREEN_HEIGHT = 800
DASHBOARD_WIDTH = 800
DASHBOARD_HEIGHT = 600
COMBINED_WIDTH = SCREEN_WIDTH + DASHBOARD_WIDTH  # Combined window width
COMBINED_HEIGHT = max(SCREEN_HEIGHT, DASHBOARD_HEIGHT)  # Combined window height
FPS = 60
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (100, 100, 100)
DARK_GRAY = (50, 50, 50)
LIGHT_GRAY = (200, 200, 200)
PURPLE = (128, 0, 128)
CYAN = (0, 255, 255)

# RL Parameters
MEMORY_SIZE = 10000
BATCH_SIZE = 128
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
MAX_SPEED = 5.0  # Default max speed (adjustable)
LEARNING_RATE = 3e-4
PPO_EPOCHS = 10
PPO_EPSILON = 0.2
SENSOR_COUNT = 8  # Number of distance sensors
MAX_SENSOR_DISTANCE = 200  # Maximum sensor detection distance
SAVE_INTERVAL = 10  # Save model every N episodes

# Global variables for communication between threads
frame_data = {
    'car': None,
    'walls': None,
    'checkpoints': None,
    'reward': 0,
    'total_reward': 0,
    'laps': 0,
    'episode': 0,
    'info': {}
}

# Training parameters
MAX_EPISODES = 10000  # Total episodes to train for

# Training metrics for dashboard
metrics_data = {
    'episode_rewards': [],  # List of episode rewards
    'episode_lengths': [],  # List of episode lengths (steps)
    'episode_laps': [],  # List of completed laps per episode
    'avg_rewards': [],  # Moving average of rewards
    'episode_times': [],  # Time taken for each episode (in seconds)
    'avg_episode_time': 0,  # Average time per episode
    'estimated_completion_time': 0,  # Estimated time to complete all episodes
    'learning_rate': LEARNING_RATE,
    'epsilon': PPO_EPSILON,
    'last_loss': 0.0,  # Last training loss
    'actor_loss': 0.0,  # Actor component loss
    'critic_loss': 0.0,  # Critic component loss
    'entropy_loss': 0.0,  # Entropy component loss
    'recent_actions': [],  # Recent actions taken
    'recent_rewards': [],  # Recent rewards received
    'training_step': 0,  # Current training step
    'updates_performed': 0,  # Number of network updates
    'crash_locations': [],  # Where cars crashed (x,y,episode)
    'checkpoint_times': {},  # Time to reach each checkpoint
    'memory_usage': 0,  # Current replay buffer usage
    'estimated_runtime': 0,  # Estimated training time
    'start_time': time.time()  # When training started
}

data_lock = threading.Lock()
render_queue = queue.Queue(maxsize=1)
running = True


# Neural network for the PPO Agent
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(ActorCritic, self).__init__()

        # Shared layers
        self.layer1 = nn.Linear(state_dim, 256)
        self.layer2 = nn.Linear(256, 256)

        # Actor (policy) layers
        self.mean_layer = nn.Linear(256, action_dim)
        self.log_std_layer = nn.Linear(256, action_dim)

        # Critic (value) layers
        self.value_layer = nn.Linear(256, 1)

        self.max_action = max_action

    def forward(self, state):
        x = F.relu(self.layer1(state))
        x = F.relu(self.layer2(x))

        # Actor: Mean and standard deviation for action distribution
        action_mean = torch.tanh(self.mean_layer(x)) * self.max_action
        action_log_std = self.log_std_layer(x)
        action_log_std = torch.clamp(action_log_std, min=-20, max=2)
        action_std = torch.exp(action_log_std)

        # Critic: State value
        value = self.value_layer(x)

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


# PPO Agent
class PPOAgent:
    def __init__(self, state_dim, action_dim, max_action):
        self.actor_critic = ActorCritic(state_dim, action_dim, max_action).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=LEARNING_RATE)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action

        self.memory = deque(maxlen=MEMORY_SIZE)
        self.trajectory = []

    def select_action(self, state, evaluation=False):
        state = torch.FloatTensor(state).to(device).unsqueeze(0)

        with torch.no_grad():
            if evaluation:
                action = self.actor_critic.get_action(state, evaluation=True)
                return action.cpu().data.numpy().flatten()
            else:
                action, log_prob = self.actor_critic.get_action(state)
                return action.cpu().data.numpy().flatten(), log_prob.cpu().data.numpy()

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
                _, _, value_t = self.actor_critic(state_t)
                value_t = value_t.cpu().data.numpy()[0, 0]

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
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).to(device)
        returns = torch.FloatTensor(np.array(returns)).to(device).unsqueeze(1)
        advantages = torch.FloatTensor(np.array(advantages)).to(device).unsqueeze(1)

        # Track losses
        total_loss = 0
        actor_loss_total = 0
        critic_loss_total = 0
        entropy_loss_total = 0

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
            entropy_loss = -0.01 * entropy  # Encourage exploration

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

        # Return average losses
        return (
            total_loss / PPO_EPOCHS,
            actor_loss_total / PPO_EPOCHS,
            critic_loss_total / PPO_EPOCHS,
            entropy_loss_total / PPO_EPOCHS
        )

    def save(self, filename="ppo_car_model.pth"):
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filename)
        print(f"Model saved to {filename}")

    def load(self, filename="ppo_car_model.pth"):
        if os.path.isfile(filename):
            checkpoint = torch.load(filename, map_location=device)
            self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print(f"Model loaded from {filename}")
            return True
        return False


# Car environment
class Car:
    def __init__(self, x, y, width=30, height=15):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.angle = 0  # Angle in radians
        self.speed = 0
        self.max_speed = MAX_SPEED
        self.acceleration = 0
        self.steering = 0
        self.sensor_readings = [MAX_SENSOR_DISTANCE] * SENSOR_COUNT
        self.checkpoint_idx = 0  # Current checkpoint index
        self.laps_completed = 0
        self.time_alive = 0
        self.total_distance = 0
        self.prev_position = (x, y)
        self.crashed = False

    def get_corners(self):
        # Calculate the corners of the car based on current position and rotation
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        # These are the corner offsets from center (before rotation)
        half_width = self.width / 2
        half_height = self.height / 2
        corners = [
            (-half_width, -half_height),  # Front left
            (half_width, -half_height),  # Front right
            (half_width, half_height),  # Rear right
            (-half_width, half_height),  # Rear left
        ]

        # Rotate and translate the corners
        rotated_corners = []
        for corner_x, corner_y in corners:
            # Rotate
            x_rotated = corner_x * cos_angle - corner_y * sin_angle
            y_rotated = corner_x * sin_angle + corner_y * cos_angle
            # Translate
            x_final = self.x + x_rotated
            y_final = self.y + y_rotated
            rotated_corners.append((x_final, y_final))

        return rotated_corners

    def reset(self, x, y, angle=0):
        self.x = x
        self.y = y
        self.angle = angle  # Set initial angle in radians
        self.speed = 0
        self.acceleration = 0
        self.steering = 0
        self.sensor_readings = [MAX_SENSOR_DISTANCE] * SENSOR_COUNT
        self.checkpoint_idx = 0
        self.laps_completed = 0
        self.time_alive = 0
        self.total_distance = 0
        self.prev_position = (x, y)
        self.crashed = False

    def update(self, walls):
        # Update angle based on steering
        self.angle += self.steering * (self.speed / self.max_speed) * 0.1
        self.angle %= 2 * math.pi  # Keep angle between 0 and 2π

        # Update speed based on acceleration
        self.speed += self.acceleration * 0.1

        # Apply friction
        self.speed *= 0.95

        # Clamp speed
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))  # Allow reverse but at lower max speed

        # Store previous position for movement calculation
        self.prev_position = (self.x, self.y)

        # Update position
        self.x += math.cos(self.angle) * self.speed
        self.y += math.sin(self.angle) * self.speed

        # Calculate traveled distance
        dx = self.x - self.prev_position[0]
        dy = self.y - self.prev_position[1]
        distance_moved = math.sqrt(dx * dx + dy * dy)

        # Only add to total distance if moving forward
        if self.speed > 0:
            self.total_distance += distance_moved

        # Increment time alive
        self.time_alive += 1

        # Check for collision
        self.crashed = self.check_collision(walls)

        # Update sensors
        self.update_sensors(walls)

        return not self.crashed

    def check_collision(self, walls):
        car_corners = self.get_corners()

        # Check each car edge against each wall
        for i in range(4):
            car_p1 = car_corners[i]
            car_p2 = car_corners[(i + 1) % 4]

            for wall in walls:
                wall_p1 = (wall[0], wall[1])
                wall_p2 = (wall[2], wall[3])

                if self.line_intersection(car_p1, car_p2, wall_p1, wall_p2):
                    return True

        return False

    def update_sensors(self, walls):
        # Define sensor angles relative to car's direction
        sensor_angles = [
            0,  # Front
            math.pi / 4,  # Front-right
            math.pi / 2,  # Right
            3 * math.pi / 4,  # Rear-right
            math.pi,  # Rear
            5 * math.pi / 4,  # Rear-left
            3 * math.pi / 2,  # Left
            7 * math.pi / 4  # Front-left
        ]

        for i, rel_angle in enumerate(sensor_angles):
            # Calculate absolute angle
            angle = (self.angle + rel_angle) % (2 * math.pi)

            # Calculate sensor end point (at maximum distance)
            end_x = self.x + math.cos(angle) * MAX_SENSOR_DISTANCE
            end_y = self.y + math.sin(angle) * MAX_SENSOR_DISTANCE

            # Find closest intersection with any wall
            min_dist = MAX_SENSOR_DISTANCE

            for wall in walls:
                wall_p1 = (wall[0], wall[1])
                wall_p2 = (wall[2], wall[3])

                intersection = self.line_segment_intersection((self.x, self.y), (end_x, end_y), wall_p1, wall_p2)

                if intersection:
                    dx = intersection[0] - self.x
                    dy = intersection[1] - self.y
                    dist = math.sqrt(dx * dx + dy * dy)
                    min_dist = min(min_dist, dist)

            self.sensor_readings[i] = min_dist

    def check_checkpoint(self, checkpoints):
        # Check if car has reached the next checkpoint
        next_checkpoint = checkpoints[self.checkpoint_idx]
        dx = self.x - next_checkpoint[0]
        dy = self.y - next_checkpoint[1]
        distance = math.sqrt(dx * dx + dy * dy)

        if distance < 50:  # Checkpoint radius
            self.checkpoint_idx = (self.checkpoint_idx + 1) % len(checkpoints)

            # If we've completed a full lap
            if self.checkpoint_idx == 0:
                self.laps_completed += 1
                return True

            return True  # Reached a checkpoint

        return False  # No checkpoint reached

    def get_state(self):
        # Normalize sensor readings
        normalized_sensors = [reading / MAX_SENSOR_DISTANCE for reading in self.sensor_readings]

        # Normalized speed
        normalized_speed = self.speed / self.max_speed

        # Append additional state information
        state = normalized_sensors + [normalized_speed, math.sin(self.angle), math.cos(self.angle)]

        return np.array(state, dtype=np.float32)

    @staticmethod
    def line_intersection(p1, p2, p3, p4):
        # Check if two line segments intersect (used for collision detection)
        def ccw(a, b, c):
            return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])

        return ccw(p1, p3, p4) != ccw(p2, p3, p4) and ccw(p1, p2, p3) != ccw(p1, p2, p4)

    @staticmethod
    def line_segment_intersection(p1, p2, p3, p4):
        # Find the intersection point of two line segments (used for sensors)
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = p3
        x4, y4 = p4

        # Calculate denominator
        den = (y4 - y3) * (x2 - x1) - (x4 - x3) * (y2 - y1)

        if den == 0:
            return None  # Lines are parallel

        ua = ((x4 - x3) * (y1 - y3) - (y4 - y3) * (x1 - x3)) / den
        ub = ((x2 - x1) * (y1 - y3) - (y2 - y1) * (x1 - x3)) / den

        # If the intersection is within both line segments
        if 0 <= ua <= 1 and 0 <= ub <= 1:
            x = x1 + ua * (x2 - x1)
            y = y1 + ua * (y2 - y1)
            return (x, y)

        return None


# Map creation - walls are defined as line segments (start_x, start_y, end_x, end_y)
def create_map():
    # Create a simple oval track - much more consistent and clean

    # Track parameters
    center_x, center_y = 600, 400  # Center of the screen
    outer_width, outer_height = 700, 500  # Oval dimensions for outer track
    inner_width, inner_height = 400, 250  # Oval dimensions for inner track
    segments = 16  # Number of segments to approximate the oval (higher = smoother)

    outer_track = []
    inner_track = []
    checkpoints = []

    # Create the oval track using line segments
    for i in range(segments):
        angle1 = 2 * math.pi * i / segments
        angle2 = 2 * math.pi * (i + 1) / segments

        # Outer track points
        x1_outer = center_x + outer_width / 2 * math.cos(angle1)
        y1_outer = center_y + outer_height / 2 * math.sin(angle1)
        x2_outer = center_x + outer_width / 2 * math.cos(angle2)
        y2_outer = center_y + outer_height / 2 * math.sin(angle2)

        # Inner track points
        x1_inner = center_x + inner_width / 2 * math.cos(angle1)
        y1_inner = center_y + inner_height / 2 * math.sin(angle1)
        x2_inner = center_x + inner_width / 2 * math.cos(angle2)
        y2_inner = center_y + inner_height / 2 * math.sin(angle2)

        # Add line segments
        outer_track.append((x1_outer, y1_outer, x2_outer, y2_outer))
        inner_track.append((x1_inner, y1_inner, x2_inner, y2_inner))

        # Add checkpoint at every segment (more checkpoints for visibility)
        checkpoint_x = center_x + (outer_width / 2 + inner_width / 2) / 2 * math.cos(angle1)
        checkpoint_y = center_y + (outer_height / 2 + inner_height / 2) / 2 * math.sin(angle1)
        checkpoints.append((checkpoint_x, checkpoint_y))

    # All walls
    all_walls = outer_track + inner_track
    return all_walls, checkpoints


# Environment class
class CarEnv:
    def __init__(self):
        self.walls, self.checkpoints = create_map()

        # Start the car in a safe position on the right side of the track
        # Calculate a starting position that's definitely inside the track
        center_x, center_y = 600, 400
        start_x = center_x + 550 / 2  # A bit more than halfway from center to outer wall on the right
        start_y = center_y  # At the horizontal middle
        self.car = Car(start_x, start_y)

        self.done = False
        self.steps = 0
        self.max_steps = 2000
        self.reward = 0
        self.total_reward = 0
        self.prev_checkpoint_time = 0
        self.last_progress = 0

        # State and action dimensions
        self.state_dim = SENSOR_COUNT + 3  # Sensors + speed + sin(angle) + cos(angle)
        self.action_dim = 2  # Acceleration and steering

    def reset(self):
        # Make sure car starts at the same safe position
        center_x, center_y = 600, 400
        start_x = center_x + 550 / 2
        start_y = center_y
        # Set car pointing tangent to the track (to the left)
        start_angle = math.pi
        self.car.reset(start_x, start_y, start_angle)

        self.done = False
        self.steps = 0
        self.reward = 0
        self.total_reward = 0
        self.prev_checkpoint_time = 0
        self.last_progress = 0
        return self.car.get_state()

    def step(self, action):
        # Apply action (acceleration, steering)
        self.car.acceleration = action[0] * 2  # Scale action to reasonable acceleration
        self.car.steering = action[1]

        # Update car
        alive = self.car.update(self.walls)
        self.steps += 1

        # Initialize reward (no default time penalty)
        reward = 0.0

        # Check if car has reached a checkpoint
        checkpoint_reached = self.car.check_checkpoint(self.checkpoints)

        # Get the next checkpoint and calculate direction to it
        next_checkpoint_idx = self.car.checkpoint_idx
        next_checkpoint = self.checkpoints[next_checkpoint_idx]

        # Calculate vector from car to next checkpoint
        dx = next_checkpoint[0] - self.car.x
        dy = next_checkpoint[1] - self.car.y
        distance_to_checkpoint = math.sqrt(dx * dx + dy * dy)

        # Calculate car's forward direction vector
        forward_x = math.cos(self.car.angle)
        forward_y = math.sin(self.car.angle)

        # Calculate dot product to see if car is pointing toward checkpoint
        # Normalize the checkpoint direction vector
        if distance_to_checkpoint > 0:
            checkpoint_dx = dx / distance_to_checkpoint
            checkpoint_dy = dy / distance_to_checkpoint
            direction_alignment = forward_x * checkpoint_dx + forward_y * checkpoint_dy
        else:
            direction_alignment = 1  # Prevent division by zero

        # Calculate reward
        if not alive:
            # Penalize crashing, but scale based on progress
            # Cars that crash after making substantial progress should be penalized less
            progress_factor = min(1.0, self.car.total_distance / 500)  # Normalize progress up to 500 distance units
            crash_penalty = -5.0 * (1.0 - progress_factor)  # Penalty ranges from -5 to 0 based on progress
            reward += crash_penalty
            self.done = True
        elif self.steps >= self.max_steps:
            # Timeout - mild penalty
            reward += -2.0
            self.done = True
        elif checkpoint_reached:
            # Significantly reward checkpoint progress
            checkpoint_reward = 20.0  # Base reward for reaching any checkpoint
            reward += checkpoint_reward

            # Add a modest time bonus, but not so large that it dominates
            # This still rewards faster completion but won't overshadow checkpoint progress
            time_to_checkpoint = self.steps - self.prev_checkpoint_time
            self.prev_checkpoint_time = self.steps

            # Cap the time bonus to avoid excessive punishment for exploration
            speed_bonus = 10.0 / max(10, time_to_checkpoint) * 10  # Cap bonus between 0 and 10
            reward += speed_bonus

            # Extra reward for completing a lap
            if self.car.checkpoint_idx == 0:
                reward += 100.0  # Increased from 50 to make lap completion more significant
        else:
            # Reward for making forward progress toward next checkpoint
            progress = self.car.total_distance
            progress_diff = progress - self.last_progress
            self.last_progress = progress

            # Increase progress reward significantly
            if progress_diff > 0:
                # Multiply by alignment to reward moving toward checkpoint
                reward += progress_diff * 0.5 * max(0, direction_alignment)  # Increased from 0.1 to 0.5

            # Smaller penalty for very slow movement or reversing
            if progress_diff < 0.01:
                reward -= 0.05  # Reduced from 0.1

            # Increase alignment reward
            reward += direction_alignment * 0.05  # Increased from 0.01

            # Reward smooth driving (less steering changes)
            smoothness = 1.0 - min(1.0, abs(self.car.steering) * 2)
            reward += smoothness * 0.05  # Increased from 0.01

            # Increased proximity reward for approaching checkpoint
            reward += 0.5 / max(10, distance_to_checkpoint)  # Increased from 0.01

        self.reward = reward
        self.total_reward += reward

        # Return next state, reward, done flag, and info
        next_state = self.car.get_state()
        info = {
            'laps': self.car.laps_completed,
            'checkpoints': self.car.checkpoint_idx,
            'distance': self.car.total_distance,
            'speed': self.car.speed,
            'steps': self.steps,
            'direction_alignment': direction_alignment
        }

        return next_state, reward, self.done, info


# Helper function to draw line plots
def draw_line_plot(surface, rect, data, min_val=None, max_val=None, color=WHITE, label=None):
    if not data:
        return

    # Determine min and max values if not provided
    if min_val is None:
        min_val = min(data)
    if max_val is None:
        max_val = max(data)

    # Ensure we don't divide by zero
    if max_val == min_val:
        max_val = min_val + 1

    # Draw axes
    pygame.draw.line(surface, LIGHT_GRAY, (rect.left, rect.bottom), (rect.right, rect.bottom), 1)  # X-axis
    pygame.draw.line(surface, LIGHT_GRAY, (rect.left, rect.top), (rect.left, rect.bottom), 1)  # Y-axis

    # Draw label
    if label:
        label_font = pygame.font.SysFont('Arial', 12)
        label_surface = label_font.render(label, True, color)
        surface.blit(label_surface, (rect.left + 5, rect.top + 5))

    # Draw min/max values
    value_font = pygame.font.SysFont('Arial', 10)
    min_surface = value_font.render(f"{min_val:.2f}", True, LIGHT_GRAY)
    max_surface = value_font.render(f"{max_val:.2f}", True, LIGHT_GRAY)
    surface.blit(min_surface, (rect.left - 25, rect.bottom - 10))
    surface.blit(max_surface, (rect.left - 25, rect.top))

    # Draw the plot line
    points = []
    for i, value in enumerate(data):
        x = rect.left + (i / (len(data) - 1 if len(data) > 1 else 1)) * rect.width
        y = rect.bottom - ((value - min_val) / (max_val - min_val)) * rect.height
        points.append((x, y))

    if len(points) > 1:
        pygame.draw.lines(surface, color, False, points, 2)

    # Draw the last point as a circle
    if points:
        pygame.draw.circle(surface, color, (int(points[-1][0]), int(points[-1][1])), 3)


# Training thread
def training_thread():
    global frame_data, running, metrics_data

    # Create environment and agent
    env = CarEnv()
    agent = PPOAgent(env.state_dim, env.action_dim, 1.0)  # Actions range from -1 to 1

    # Load model if exists
    model_loaded = agent.load()

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

        while not done:
            # Select action
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
                        _, _, last_value = agent.actor_critic(next_state_tensor)
                        last_value = last_value.cpu().data.numpy()[0, 0]

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

        # End of episode
        episode_rewards.append(episode_reward)
        episode_lengths.append(steps)
        episode_laps.append(env.car.laps_completed)
        recent_rewards.append(episode_reward)

        # Calculate episode time and update time metrics
        episode_time = time.time() - episode_start_time
        metrics_data['episode_times'].append(episode_time)

        # Calculate moving average of rewards
        avg_reward = np.mean(recent_rewards)

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
            metrics_data['avg_rewards'].append(avg_reward)
            metrics_data['avg_episode_time'] = avg_episode_time
            metrics_data['estimated_completion_time'] = estimated_completion_time
            metrics_data['estimated_runtime'] = time.time() - metrics_data['start_time']

        print(
            f"Episode {episode}: Reward = {episode_reward:.2f}, Avg Reward = {avg_reward:.2f}, Laps = {env.car.laps_completed}, Steps = {steps}")

        # Save model periodically
        if episode > 0 and episode % SAVE_INTERVAL == 0:
            agent.save()

    # Save final model
    agent.save()


# Combined rendering thread that shows both simulation and dashboard
def combined_rendering_thread():
    global frame_data, running, metrics_data

    try:
        print("Starting combined rendering thread...")

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

        # Car image setup
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

        # Dashboard sections
        dashboard_sections = {
            'training_stats': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 10, 380, 180), 'title': 'Training Statistics'},
            'episode_stats': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 10, 380, 180), 'title': 'Current Episode'},
            'reward_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 200, 380, 180), 'title': 'Reward History'},
            'loss_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 200, 380, 180), 'title': 'Loss History'},
            'action_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 10, 390, 380, 180), 'title': 'Recent Actions'},
            'sensor_plot': {'rect': pygame.Rect(SCREEN_WIDTH + 400, 390, 380, 180), 'title': 'Sensor Readings'}
        }

        # Track data for plots
        reward_history = []
        loss_history = []
        action_history = []
        sensor_history = []

        # For downsampling larger datasets
        downsample_factor = 1
        max_plot_points = 100

        print("Starting render loop")
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Clear the entire screen
            combined_screen.fill(BLACK)

            # Get current state data (thread-safe)
            with data_lock:
                car = frame_data['car']
                walls = frame_data['walls']
                checkpoints = frame_data['checkpoints']
                reward = frame_data['reward']
                total_reward = frame_data['total_reward']
                laps = frame_data['laps']
                episode = frame_data['episode']
                info = frame_data['info']
                current_metrics = metrics_data.copy()
                current_frame = frame_data.copy()

            # Draw simulation part (left side)
            # ===================================
            simulation_surface = combined_screen.subsurface(pygame.Rect(0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

            # Track surface
            track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

            if walls:
                # Fill the track area with a dark gray
                pygame.draw.rect(track_surface, (30, 30, 30), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

                # Draw the actual track in a slightly lighter gray
                if len(walls) >= 32:  # If using our new oval track with inner/outer walls
                    # Extract points from walls for the outer and inner boundaries
                    outer_points = [(walls[i][0], walls[i][1]) for i in range(16)]  # First 16 segments are outer
                    inner_points = [(walls[i][0], walls[i][1]) for i in range(16, 32)]  # Next 16 are inner

                    # Draw the track surface
                    pygame.draw.polygon(track_surface, (50, 50, 50), outer_points)
                    pygame.draw.polygon(track_surface, (30, 30, 30), inner_points)

            simulation_surface.blit(track_surface, (0, 0))

            # Draw walls
            if walls:
                for wall in walls:
                    pygame.draw.line(simulation_surface, WHITE, (wall[0], wall[1]), (wall[2], wall[3]), 2)

            # Draw checkpoints
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
                for i in range(SENSOR_COUNT):
                    # Calculate sensor angle
                    sensor_angle = (car.angle + i * math.pi / 4) % (2 * math.pi)
                    # Calculate endpoint based on reading
                    end_x = car.x + math.cos(sensor_angle) * car.sensor_readings[i]
                    end_y = car.y + math.sin(sensor_angle) * car.sensor_readings[i]
                    # Draw line
                    pygame.draw.line(simulation_surface, GREEN, (car.x, car.y), (end_x, end_y), 1)

            # Draw simulation stats
            info_text = [
                f"Episode: {episode}",
                f"Reward: {reward:.2f}",
                f"Total Reward: {total_reward:.2f}",
                f"Laps: {laps}",
                f"Speed: {car.speed:.2f}" if car else "Speed: 0.00",
                f"Distance: {car.total_distance:.2f}" if car else "Distance: 0.00",
                f"Time Alive: {car.time_alive}" if car else "Time Alive: 0",
                f"Direction Alignment: {info.get('direction_alignment', 0):.2f}" if info else "Direction Alignment: 0.00",
                f"FPS: {clock.get_fps():.1f}"
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
                    # Training statistics
                    stats = [
                        f"Episodes: {len(current_metrics['episode_rewards'])}/{MAX_EPISODES}",
                        f"Updates: {current_metrics['updates_performed']}",
                        f"Memory Usage: {current_metrics['memory_usage']}/{MEMORY_SIZE}",
                        f"Total Laps: {sum(current_metrics['episode_laps'])}",
                        f"Avg Reward: {np.mean(current_metrics['episode_rewards'][-100:]) if current_metrics['episode_rewards'] else 0:.2f}",
                        f"Avg Episode Time: {current_metrics['avg_episode_time']:.2f}s",
                        f"Est. Completion: {time.strftime('%H:%M:%S', time.gmtime(current_metrics['estimated_completion_time']))}",
                        f"Runtime: {time.strftime('%H:%M:%S', time.gmtime(current_metrics['estimated_runtime']))}",
                        f"Using: {device}"
                    ]

                    for i, stat in enumerate(stats):
                        stat_surface = regular_font.render(stat, True, WHITE)
                        combined_screen.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                elif section_id == 'episode_stats':
                    # Current episode statistics
                    stats = [
                        f"Episode: {current_frame['episode']}",
                        f"Reward: {current_frame['reward']:.2f}",
                        f"Total Reward: {current_frame['total_reward']:.2f}",
                        f"Laps: {current_frame['laps']}",
                        f"Last Loss: {current_metrics['last_loss']:.4f}",
                        f"Actor Loss: {current_metrics['actor_loss']:.4f}",
                        f"Critic Loss: {current_metrics['critic_loss']:.4f}",
                        f"Entropy Loss: {current_metrics['entropy_loss']:.4f}"
                    ]

                    for i, stat in enumerate(stats):
                        stat_surface = regular_font.render(stat, True, WHITE)
                        combined_screen.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                elif section_id == 'reward_plot':
                    # Reward history plot
                    plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                            section['rect'].width - 20, section['rect'].height - 40)

                    # Update reward history data
                    if current_metrics['episode_rewards']:
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
                    if hasattr(current_metrics, 'actor_loss_history'):
                        draw_line_plot(combined_screen, plot_rect, current_metrics['actor_loss_history'][-100:],
                                       color=RED, label="Actor Loss")
                        draw_line_plot(combined_screen, plot_rect, current_metrics['critic_loss_history'][-100:],
                                       color=BLUE, label="Critic Loss")
                    else:
                        # Draw placeholder
                        text = small_font.render("Loss data will appear here as training progresses", True, WHITE)
                        combined_screen.blit(text, (plot_rect.x + 10, plot_rect.y + plot_rect.height // 2))

                elif section_id == 'action_plot':
                    # Recent actions plot
                    plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                            section['rect'].width - 20, section['rect'].height - 40)

                    # Update action history
                    if current_metrics['recent_actions']:
                        # Extract acceleration and steering
                        if len(current_metrics['recent_actions']) > 0 and len(
                                current_metrics['recent_actions'][0]) >= 2:
                            accel_actions = [a[0] for a in current_metrics['recent_actions']]
                            steer_actions = [a[1] for a in current_metrics['recent_actions']]

                            # Draw the actions
                            draw_line_plot(combined_screen, plot_rect, accel_actions,
                                           min_val=-1, max_val=1, color=CYAN, label="Acceleration")
                            draw_line_plot(combined_screen, plot_rect, steer_actions,
                                           min_val=-1, max_val=1, color=PURPLE, label="Steering")

                elif section_id == 'sensor_plot':
                    # Sensor readings visualization
                    if current_frame['car'] and hasattr(current_frame['car'], 'sensor_readings'):
                        sensor_readings = current_frame['car'].sensor_readings

                        # Create a radar chart
                        center_x = section['rect'].x + section['rect'].width // 2
                        center_y = section['rect'].y + section['rect'].height // 2
                        radius = min(section['rect'].width, section['rect'].height) // 2 - 20

                        # Normalize readings
                        max_reading = MAX_SENSOR_DISTANCE
                        normalized_readings = [min(1.0, r / max_reading) for r in sensor_readings]

                        # Draw radar chart background
                        for i in range(3):
                            r = radius * (i + 1) / 3
                            pygame.draw.circle(combined_screen, DARK_GRAY, (center_x, center_y), int(r), 1)

                        # Draw sensor lines
                        for i in range(len(normalized_readings)):
                            angle = i * 2 * math.pi / len(normalized_readings)
                            end_x = center_x + radius * math.cos(angle)
                            end_y = center_y + radius * math.sin(angle)
                            pygame.draw.line(combined_screen, DARK_GRAY, (center_x, center_y), (end_x, end_y), 1)

                            # Draw sensor reading
                            r = radius * (1 - normalized_readings[i])
                            point_x = center_x + r * math.cos(angle)
                            point_y = center_y + r * math.sin(angle)
                            pygame.draw.circle(combined_screen, RED, (int(point_x), int(point_y)), 4)

                        # Connect the points
                        points = []
                        for i in range(len(normalized_readings)):
                            angle = i * 2 * math.pi / len(normalized_readings)
                            r = radius * (1 - normalized_readings[i])
                            point_x = center_x + r * math.cos(angle)
                            point_y = center_y + r * math.sin(angle)
                            points.append((int(point_x), int(point_y)))

                        if len(points) > 2:
                            # Draw filled polygon with alpha would be ideal, but pygame doesn't support alpha in this context
                            # We'll use a solid color instead
                            pygame.draw.polygon(combined_screen, (100, 0, 0), points, 0)  # Filled
                            pygame.draw.polygon(combined_screen, RED, points, 1)  # Outline

            # Update display
            pygame.display.flip()

            # Cap at 60 FPS
            clock.tick(FPS)

    except Exception as e:
        print(f"Error in combined rendering thread: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("Combined rendering thread exiting")
        pygame.quit()


# Rendering thread
def rendering_thread():
    global frame_data, running

    try:
        print("Rendering thread starting...")

        # Initialize pygame (only once)
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

        print("Starting render loop")
        while running:
            # Handle events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

            # Background - track surface
            track_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)

            # Get current state data (thread-safe)
            with data_lock:
                car = frame_data['car']
                walls = frame_data['walls']
                checkpoints = frame_data['checkpoints']
                reward = frame_data['reward']
                total_reward = frame_data['total_reward']
                laps = frame_data['laps']
                episode = frame_data['episode']
                info = frame_data['info']

            if walls:
                # Fill the track area with a dark gray
                pygame.draw.rect(track_surface, (30, 30, 30), (0, 0, SCREEN_WIDTH, SCREEN_HEIGHT))

                # Draw the actual track in a slightly lighter gray
                if len(walls) >= 32:  # If using our new oval track with inner/outer walls
                    # Extract points from walls for the outer and inner boundaries
                    outer_points = [(walls[i][0], walls[i][1]) for i in range(16)]  # First 16 segments are outer
                    inner_points = [(walls[i][0], walls[i][1]) for i in range(16, 32)]  # Next 16 are inner

                    # Draw the track surface
                    pygame.draw.polygon(track_surface, (50, 50, 50), outer_points)
                    pygame.draw.polygon(track_surface, (30, 30, 30), inner_points)

            screen.blit(track_surface, (0, 0))

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
                for i in range(SENSOR_COUNT):
                    # Calculate sensor angle
                    sensor_angle = (car.angle + i * math.pi / 4) % (2 * math.pi)
                    # Calculate endpoint based on reading
                    end_x = car.x + math.cos(sensor_angle) * car.sensor_readings[i]
                    end_y = car.y + math.sin(sensor_angle) * car.sensor_readings[i]
                    # Draw line
                    pygame.draw.line(screen, GREEN, (car.x, car.y), (end_x, end_y), 1)

            # Draw stats
            font = pygame.font.SysFont('Arial', 16)
            info_text = [
                f"Episode: {episode}",
                f"Reward: {reward:.2f}",
                f"Total Reward: {total_reward:.2f}",
                f"Laps: {laps}",
                f"Speed: {car.speed:.2f}" if car else "Speed: 0.00",
                f"Distance: {car.total_distance:.2f}" if car else "Distance: 0.00",
                f"Time Alive: {car.time_alive}" if car else "Time Alive: 0",
                f"Direction Alignment: {info.get('direction_alignment', 0):.2f}" if info else "Direction Alignment: 0.00",
                f"FPS: {clock.get_fps():.1f}"
            ]

            for i, text in enumerate(info_text):
                text_surface = font.render(text, True, WHITE)
                screen.blit(text_surface, (10, 10 + i * 20))

            # Update display
            pygame.display.flip()

            # Cap at 60 FPS
            clock.tick(FPS)
    except Exception as e:
        print(f"Error in rendering thread: {e}")
    finally:
        print("Rendering thread exiting")
        pygame.quit()


# Dashboard thread
def dashboard_thread():
    global frame_data, running, metrics_data

    try:
        print("Dashboard thread starting...")

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

        # Dashboard sections
        sections = {
            'training_stats': {'rect': pygame.Rect(10, 10, 380, 180), 'title': 'Training Statistics'},
            'episode_stats': {'rect': pygame.Rect(400, 10, 380, 180), 'title': 'Current Episode'},
            'reward_plot': {'rect': pygame.Rect(10, 200, 380, 180), 'title': 'Reward History'},
            'loss_plot': {'rect': pygame.Rect(400, 200, 380, 180), 'title': 'Loss History'},
            'action_plot': {'rect': pygame.Rect(10, 390, 380, 180), 'title': 'Recent Actions'},
            'sensor_plot': {'rect': pygame.Rect(400, 390, 380, 180), 'title': 'Sensor Readings'}
        }

        # Track data for plots
        reward_history = []
        loss_history = []
        action_history = []
        sensor_history = []

        # For downsampling larger datasets
        downsample_factor = 1
        max_plot_points = 100

        while running:
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

            # Get current metrics (thread-safe)
            with data_lock:
                current_metrics = metrics_data.copy()
                current_frame = frame_data.copy()

            # Draw sections
            for section_id, section in sections.items():
                # Draw section background
                pygame.draw.rect(dashboard, DARK_GRAY, section['rect'])
                pygame.draw.rect(dashboard, LIGHT_GRAY, section['rect'], 1)

                # Draw section title
                title_surface = title_font.render(section['title'], True, WHITE)
                dashboard.blit(title_surface, (section['rect'].x + 10, section['rect'].y + 5))

                # Draw section content based on section ID
                if section_id == 'training_stats':
                    # Training statistics
                    stats = [
                        f"Episodes: {len(current_metrics['episode_rewards'])}",
                        f"Updates: {current_metrics['updates_performed']}",
                        f"Memory Usage: {current_metrics['memory_usage']}/{MEMORY_SIZE}",
                        f"Total Laps: {sum(current_metrics['episode_laps'])}",
                        f"Avg Reward: {np.mean(current_metrics['episode_rewards'][-100:]) if current_metrics['episode_rewards'] else 0:.2f}",
                        f"Learning Rate: {current_metrics['learning_rate']:.6f}",
                        f"Runtime: {time.strftime('%H:%M:%S', time.gmtime(current_metrics['estimated_runtime']))}",
                        f"Using: {device}"
                    ]

                    for i, stat in enumerate(stats):
                        stat_surface = regular_font.render(stat, True, WHITE)
                        dashboard.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                elif section_id == 'episode_stats':
                    # Current episode statistics
                    stats = [
                        f"Episode: {current_frame['episode']}",
                        f"Reward: {current_frame['reward']:.2f}",
                        f"Total Reward: {current_frame['total_reward']:.2f}",
                        f"Laps: {current_frame['laps']}",
                        f"Last Loss: {current_metrics['last_loss']:.4f}",
                        f"Actor Loss: {current_metrics['actor_loss']:.4f}",
                        f"Critic Loss: {current_metrics['critic_loss']:.4f}",
                        f"Entropy Loss: {current_metrics['entropy_loss']:.4f}"
                    ]

                    for i, stat in enumerate(stats):
                        stat_surface = regular_font.render(stat, True, WHITE)
                        dashboard.blit(stat_surface, (section['rect'].x + 15, section['rect'].y + 30 + i * 18))

                elif section_id == 'reward_plot':
                    # Reward history plot
                    plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                            section['rect'].width - 20, section['rect'].height - 40)

                    # Update reward history data
                    if current_metrics['episode_rewards']:
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

                elif section_id == 'loss_plot':
                    # Loss history
                    plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                            section['rect'].width - 20, section['rect'].height - 40)

                    # Draw actor and critic loss if available
                    if hasattr(current_metrics, 'actor_loss_history'):
                        draw_line_plot(dashboard, plot_rect, current_metrics['actor_loss_history'][-100:],
                                       color=RED, label="Actor Loss")
                        draw_line_plot(dashboard, plot_rect, current_metrics['critic_loss_history'][-100:],
                                       color=BLUE, label="Critic Loss")
                    else:
                        # Draw placeholder
                        text = small_font.render("Loss data will appear here as training progresses", True, WHITE)
                        dashboard.blit(text, (plot_rect.x + 10, plot_rect.y + plot_rect.height // 2))

                elif section_id == 'action_plot':
                    # Recent actions plot
                    plot_rect = pygame.Rect(section['rect'].x + 10, section['rect'].y + 30,
                                            section['rect'].width - 20, section['rect'].height - 40)

                    # Update action history
                    if current_metrics['recent_actions']:
                        # Extract acceleration and steering
                        if len(current_metrics['recent_actions']) > 0 and len(
                                current_metrics['recent_actions'][0]) >= 2:
                            accel_actions = [a[0] for a in current_metrics['recent_actions']]
                            steer_actions = [a[1] for a in current_metrics['recent_actions']]

                            # Draw the actions
                            draw_line_plot(dashboard, plot_rect, accel_actions,
                                           min_val=-1, max_val=1, color=CYAN, label="Acceleration")
                            draw_line_plot(dashboard, plot_rect, steer_actions,
                                           min_val=-1, max_val=1, color=PURPLE, label="Steering")

                elif section_id == 'sensor_plot':
                    # Sensor readings visualization
                    if current_frame['car'] and hasattr(current_frame['car'], 'sensor_readings'):
                        sensor_readings = current_frame['car'].sensor_readings

                        # Create a radar chart
                        center_x = section['rect'].x + section['rect'].width // 2
                        center_y = section['rect'].y + section['rect'].height // 2
                        radius = min(section['rect'].width, section['rect'].height) // 2 - 20

                        # Normalize readings
                        max_reading = MAX_SENSOR_DISTANCE
                        normalized_readings = [min(1.0, r / max_reading) for r in sensor_readings]

                        # Draw radar chart background
                        for i in range(3):
                            r = radius * (i + 1) / 3
                            pygame.draw.circle(dashboard, DARK_GRAY, (center_x, center_y), int(r), 1)

                        # Draw sensor lines
                        for i in range(len(normalized_readings)):
                            angle = i * 2 * math.pi / len(normalized_readings)
                            end_x = center_x + radius * math.cos(angle)
                            end_y = center_y + radius * math.sin(angle)
                            pygame.draw.line(dashboard, DARK_GRAY, (center_x, center_y), (end_x, end_y), 1)

                            # Draw sensor reading
                            r = radius * (1 - normalized_readings[i])
                            point_x = center_x + r * math.cos(angle)
                            point_y = center_y + r * math.sin(angle)
                            pygame.draw.circle(dashboard, RED, (int(point_x), int(point_y)), 4)

                        # Connect the points
                        points = []
                        for i in range(len(normalized_readings)):
                            angle = i * 2 * math.pi / len(normalized_readings)
                            r = radius * (1 - normalized_readings[i])
                            point_x = center_x + r * math.cos(angle)
                            point_y = center_y + r * math.sin(angle)
                            points.append((int(point_x), int(point_y)))

                        if len(points) > 2:
                            pygame.draw.polygon(dashboard, (255, 0, 0, 128), points, 0)
                            pygame.draw.polygon(dashboard, RED, points, 1)

            # Update display and cap at 30 FPS
            pygame.display.flip()
            clock.tick(30)

    except Exception as e:
        print(f"Dashboard thread error: {e}")
    finally:
        print("Dashboard thread exiting")


def main():
    global running, MAX_SPEED

    # Allow command line arguments to adjust max speed
    parser = argparse.ArgumentParser(description='Car Reinforcement Learning with PPO')
    parser.add_argument('--max_speed', type=float, default=MAX_SPEED,
                        help=f'Maximum car speed (default: {MAX_SPEED})')
    parser.add_argument('--no_gui', action='store_true',
                        help='Run without GUI for faster training')
    parser.add_argument('--headless', action='store_true',
                        help='Run in headless mode (no GUI, training only)')
    parser.add_argument('--separate_windows', action='store_true',
                        help='Use separate windows for simulation and dashboard (not recommended)')
    args = parser.parse_args()

    # Update global max speed
    MAX_SPEED = args.max_speed
    print(f"Running with max speed: {MAX_SPEED}")

    # Headless mode overrides other options
    if args.headless:
        args.no_gui = True

    # Start training thread first
    print("Starting training thread...")
    train_thread = threading.Thread(target=training_thread)
    train_thread.start()
    print("Training thread started")

    # If no GUI elements needed, just wait for training
    if args.no_gui:
        print("Running in headless mode (training only). Press Ctrl+C to stop.")
        try:
            # Just wait for training
            train_thread.join()
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            try:
                train_thread.join(timeout=1.0)
            except:
                pass
        return

    # Initialize pygame here where we know we need it
    try:
        print("Initializing pygame...")
        pygame.init()
        pygame.display.init()
        print("Pygame initialized with driver:", pygame.display.get_driver())
    except Exception as e:
        print(f"Failed to initialize pygame: {e}")
        print("Continuing in headless mode (training only)")
        # Continue with training if GUI fails
        try:
            train_thread.join()
        except KeyboardInterrupt:
            print("Training interrupted by user")
        finally:
            running = False
            train_thread.join(timeout=1.0)
        return

    try:
        # Use the combined window approach by default (unless separate windows requested)
        if args.separate_windows:
            print("Using separate windows for simulation and dashboard (experimental)...")
            # This is the old approach - may not work well on all systems
            render_thread = threading.Thread(target=rendering_thread)
            dash_thread = threading.Thread(target=dashboard_thread)
            render_thread.start()
            dash_thread.start()
            render_thread.join()
            dash_thread.join()
        else:
            # New approach - single window with both components
            print("Starting combined simulation+dashboard window...")
            combined_rendering_thread()
    except Exception as e:
        print(f"Error in GUI: {e}")
        import traceback
        traceback.print_exc()

    # We only get here after windows are closed
    running = False
    print("Windows closed, waiting for training thread to finish...")
    train_thread.join(timeout=2.0)
    print("Exiting")


if __name__ == "__main__":
    main()