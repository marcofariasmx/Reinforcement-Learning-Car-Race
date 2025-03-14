# Autonomous Car Reinforcement Learning

This project implements a reinforcement learning system where a virtual car learns to navigate a race track using Proximal Policy Optimization (PPO).

## Project Structure

The project is organized into the following files:

- `main.py` - Entry point for the application
- `config.py` - Configuration parameters and global variables
- `models.py` - Neural network models and PPO agent implementation
- `environment.py` - Car simulation environment and track generation
- `visualization.py` - Visualization and dashboard components
- `training.py` - Training loop and logic

## Requirements

- Python 3.7+
- PyTorch
- Pygame
- NumPy

## Installation

```bash
pip install torch pygame numpy
```

## Usage

Run the main script:

```bash
python main.py
```

### Command Line Arguments

- `--max_speed FLOAT` - Set maximum car speed (default: 5.0)
- `--no_gui` - Run without GUI for faster training
- `--headless` - Run in headless mode (training only)
- `--separate_windows` - Use separate windows for simulation and dashboard (experimental)

Example:

```bash
python main.py --max_speed 6.0
```

## Features

- PPO-based self-learning car agent
- Real-time visualization of the training process
- Comprehensive performance metrics
- Adaptive rewards based on progress and driving behavior
- Checkpoints system for tracking laps
- Sensor-based environment perception

## How It Works

1. The car is equipped with distance sensors that detect walls
2. The PPO agent receives the sensor readings and current state
3. The agent decides on acceleration and steering actions
4. Rewards are given for making progress, completing checkpoints and laps
5. The agent learns over time to navigate the track efficiently

## Visualization

The application displays:
- The car simulation with sensor rays
- Training statistics and metrics
- Reward and loss history graphs
- Real-time action visualization
- Sensor reading visualization

## Configuration

Key parameters can be adjusted in `config.py`:

- `MAX_EPISODES` - Total training episodes
- `MEMORY_SIZE` - Replay buffer size
- `BATCH_SIZE` - Mini-batch size for training
- `LEARNING_RATE` - Initial learning rate
- `MAX_SPEED` - Maximum speed of the car
- `SENSOR_COUNT` - Number of distance sensors
