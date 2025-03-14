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

# Constants (not hardware-dependent)
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


# Function to detect hardware capabilities and set optimal parameters
def detect_hardware_capabilities():
    """Detect hardware capabilities and return optimal parameters"""
    gpu_available = torch.cuda.is_available()
    cpu_count = os.cpu_count() or 1

    gpu_info = {}

    if gpu_available:
        try:
            # Get GPU details
            gpu_name = torch.cuda.get_device_name(0)
            gpu_compute_capability = torch.cuda.get_device_capability(0)
            vram_bytes = torch.cuda.get_device_properties(0).total_memory
            vram_gb = vram_bytes / (1024 ** 3)

            gpu_info = {
                'name': gpu_name,
                'compute_capability': gpu_compute_capability,
                'vram_gb': vram_gb
            }

            print(f"GPU: {gpu_name}, Compute: {gpu_compute_capability}, VRAM: {vram_gb:.2f} GB")

            # Adaptive settings based on GPU
            if vram_gb > 6.0:  # High-end GPU
                return {
                    'batch_size': 2048,
                    'memory_size': 100000,
                    'max_speed': 5.0,
                    'learning_rate': 3e-4,
                    'ppo_epochs': 10,
                    'save_interval': 50,
                    'use_mixed_precision': True,
                    'use_pin_memory': True,
                    'use_gpu_for_inference': False,  # Still better to use CPU for small inference batches
                    'gpu_info': gpu_info,
                    'async_save': True
                }
            else:  # Low VRAM GPU
                return {
                    'batch_size': 1024,
                    'memory_size': 50000,
                    'max_speed': 5.0,
                    'learning_rate': 3e-4,
                    'ppo_epochs': 5,
                    'save_interval': 25,
                    'use_mixed_precision': False,
                    'use_pin_memory': True,
                    'use_gpu_for_inference': False,  # Use CPU for inference to avoid small transfers
                    'gpu_info': gpu_info,
                    'async_save': True
                }
        except Exception as e:
            print(f"Error detecting GPU capabilities: {e}")
            gpu_available = False

    if not gpu_available:
        print(f"Using CPU with {cpu_count} cores")
        # CPU-optimized settings
        return {
            'batch_size': 512,
            'memory_size': 25000,
            'max_speed': 5.0,
            'learning_rate': 3e-4,
            'ppo_epochs': 5,
            'save_interval': 25,
            'use_mixed_precision': False,
            'use_pin_memory': False,
            'use_gpu_for_inference': False,
            'gpu_info': None,
            'async_save': True
        }


# Get hardware-specific settings
hw_settings = detect_hardware_capabilities()

# RL Parameters - initialize with auto-detected settings
MEMORY_SIZE = hw_settings['memory_size']
BATCH_SIZE = hw_settings['batch_size']
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
MAX_SPEED = hw_settings['max_speed']
LEARNING_RATE = hw_settings['learning_rate']
PPO_EPOCHS = hw_settings['ppo_epochs']
PPO_EPSILON = 0.2
SENSOR_COUNT = 8  # Number of distance sensors
MAX_SENSOR_DISTANCE = 200  # Maximum sensor detection distance
SAVE_INTERVAL = hw_settings['save_interval']
USE_MIXED_PRECISION = hw_settings['use_mixed_precision']
USE_PIN_MEMORY = hw_settings['use_pin_memory']
USE_GPU_FOR_INFERENCE = hw_settings['use_gpu_for_inference']
USE_ASYNC_SAVE = hw_settings['async_save']
GPU_INFO = hw_settings['gpu_info']


# Method to update settings from command line args
def update_settings_from_args(args):
    """Update global settings based on command line arguments"""
    global MAX_SPEED, USE_MIXED_PRECISION, USE_GPU_FOR_INFERENCE
    global USE_PIN_MEMORY, USE_ASYNC_SAVE, SAVE_INTERVAL, BATCH_SIZE
    global PPO_EPOCHS, device

    # Only update if explicitly provided
    if args.max_speed is not None:
        MAX_SPEED = args.max_speed
        print(f"Using custom max speed: {MAX_SPEED}")

    if args.batch_size is not None:
        BATCH_SIZE = args.batch_size
        print(f"Using custom batch size: {BATCH_SIZE}")

    if args.save_interval is not None:
        SAVE_INTERVAL = args.save_interval
        print(f"Using custom save interval: {SAVE_INTERVAL}")

    if args.ppo_epochs is not None:
        PPO_EPOCHS = args.ppo_epochs
        print(f"Using custom PPO epochs: {PPO_EPOCHS}")

    if args.gpu_inference:
        USE_GPU_FOR_INFERENCE = True
        print("Enabled GPU for inference")

    if args.cpu_inference:
        USE_GPU_FOR_INFERENCE = False
        print("Disabled GPU for inference")

    if args.mixed_precision and torch.cuda.is_available() and hasattr(torch.cuda, 'amp'):
        USE_MIXED_PRECISION = True
        print("Enabled mixed precision training")
    elif args.no_mixed_precision:
        USE_MIXED_PRECISION = False
        print("Disabled mixed precision training")

    if args.async_save:
        USE_ASYNC_SAVE = True
        print("Enabled asynchronous model saving")
    elif args.sync_save:
        USE_ASYNC_SAVE = False
        print("Disabled asynchronous model saving")

    if args.cpu and torch.cuda.is_available():
        print("Forcing CPU usage despite GPU availability")
        device = torch.device("cpu")


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
    'total_training_time': 0,  # Cumulative time across all training sessions
    'start_episode': 0,  # Starting episode number (for loaded models)
    'gpu_temp': 0,  # GPU temperature
    'gpu_util': 0,  # GPU utilization
    'nn_update_time': 0,  # Time for neural network update
    'time_between_updates': 0,  # Time between neural network updates
    'transfer_time': 0,  # Time for CPU-GPU transfers
}


# Print the auto-detected settings
def print_current_settings():
    """Print all the current settings in a readable format"""
    print("\n===== Current Settings =====")
    print(f"Device: {device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Memory Size: {MEMORY_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print(f"Max Speed: {MAX_SPEED}")
    print(f"PPO Epochs: {PPO_EPOCHS}")
    print(f"Save Interval: {SAVE_INTERVAL}")
    print(f"Mixed Precision: {USE_MIXED_PRECISION}")
    print(f"Pin Memory: {USE_PIN_MEMORY}")
    print(f"GPU for Inference: {USE_GPU_FOR_INFERENCE}")
    print(f"Async Save: {USE_ASYNC_SAVE}")
    if GPU_INFO:
        print(f"GPU: {GPU_INFO['name']}, VRAM: {GPU_INFO['vram_gb']:.2f} GB")
    print("===========================\n")


data_lock = threading.Lock()
metrics_lock = threading.Lock()  # Separate lock for metrics to reduce contention
render_queue = queue.Queue(maxsize=1)
running = True

# Improved thread synchronization with double buffering
# A double buffer Queue is more efficient than a lock for this use case
from queue import Queue, Empty

frame_buffer = Queue(maxsize=4)  # Increased buffer size

# Save queue for asynchronous model saving
save_queue = Queue(maxsize=2)

# Variable to store the latest valid frame for visualization
latest_frame = None