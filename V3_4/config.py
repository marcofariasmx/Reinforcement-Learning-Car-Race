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
import platform

# Set environment variables to help with pygame display issues
os.environ['SDL_VIDEO_ALLOW_SCREENSAVER'] = '1'
os.environ['SDL_VIDEO_CENTERED'] = '1'

# Platform-specific settings
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

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Constants
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 800
DASHBOARD_WIDTH = 800
DASHBOARD_HEIGHT = 600
COMBINED_WIDTH = SCREEN_WIDTH + DASHBOARD_WIDTH  # Combined window width
COMBINED_HEIGHT = max(SCREEN_HEIGHT, DASHBOARD_HEIGHT)  # Combined window height
FPS = 60

# Colors
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
MEMORY_SIZE = 50000
BATCH_SIZE = 1024
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
    'entropy_coef': 0.1,  # Current entropy coefficient
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
    'start_time': time.time(),  # When training started
    'total_training_time': 0  # Cumulative time across all training sessions
}

data_lock = threading.Lock()
render_queue = queue.Queue(maxsize=1)
running = True

# Add these lines to the end of config.py
from queue import Queue
# Double buffer for visualization data - completely separate from frame_data
frame_buffer = Queue(maxsize=2)