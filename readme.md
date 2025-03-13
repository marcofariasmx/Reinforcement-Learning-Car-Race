# 2D Car Racing Reinforcement Learning

A Python application that showcases reinforcement learning with a car navigating a 2D raceway, visualized from above. The car learns to drive efficiently around the track through Deep Q-Learning implemented with PyTorch.

![Car Racing Reinforcement Learning Demo](https://placeholder-image.png)

## Features

- **Real-time Visualization**: Watch the car learn to drive in real-time with Pygame
- **Deep Q-Learning**: Implements a deep neural network for reinforcement learning
- **GPU Acceleration**: Automatically utilizes CUDA GPU acceleration if available
- **Interactive Environment**: Custom 2D racetrack with dynamic sensor feedback
- **Performance Metrics**: Visualizes learning progress with charts and statistics
- **Checkpoint System**: Car receives rewards for reaching checkpoints around the track

## Requirements

- Python 3.6+
- PyTorch
- Pygame
- NumPy
- Matplotlib

## Installation

### 1. Clone the repository or create a new project

```bash
mkdir car_rl_project
cd car_rl_project
```

### 2. Create a virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

PyTorch will be installed with automatic CUDA detection, enabling GPU acceleration if available:

If GPU Cuda compatible then
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

else
```bash
pip install torch torchvision torchaudio
```
continue with
```bash
pip install pygame==2.5.2 numpy==1.24.3 matplotlib==3.7.3
```


Alternatively, you can use the provided requirements.txt:

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python car_rl.py
```

## How It Works

### Reinforcement Learning Process

1. **Environment**: A 2D oval racetrack with checkpoints
2. **Agent**: The car with sensors that detect walls and track boundaries
3. **State**: Information from the car's sensors, speed, and orientation
4. **Actions**: Accelerate, Brake, Steer Left, Steer Right
5. **Reward System**:
   - +10 points for reaching checkpoints
   - -10 points for colliding with track boundaries
   - -0.01 points per time step (encourages efficiency)

### Neural Network Architecture

The Deep Q-Network (DQN) consists of:
- Input layer: Car sensor readings + speed + orientation (11 inputs)
- Hidden layers: 2 fully-connected layers with 128 neurons each
- Output layer: Q-values for the 4 possible actions

### Learning Mechanisms

- **Experience Replay**: Stores and samples past experiences to stabilize learning
- **Target Network**: Separate network updated periodically to reduce oscillation
- **Epsilon-Greedy Exploration**: Balances exploration and exploitation, gradually reducing randomness

## Customization

You can modify various parameters in the code to change the:

- **Track shape and complexity**
- **Car physics** (speed, steering, etc.)
- **Neural network architecture**
- **Learning hyperparameters** (learning rate, discount factor, etc.)
- **Rewards and penalties**

## Performance Notes

- **GPU Acceleration**: The application automatically detects and uses CUDA-compatible GPUs for neural network training, significantly improving performance.
- **CPU Mode**: Falls back to CPU training if no GPU is available.
- **Display**: Shows which device (GPU/CPU) is being used for training in the simulation UI.

## Visualization

During training, the application displays:
- The car and its sensor readings on the track
- Current episode number and reward
- Number of checkpoints reached
- Learning exploration parameter (epsilon)
- Statistical charts showing learning progress over time

## Troubleshooting

**PyTorch Installation Issues:**
- If you encounter issues with PyTorch installation, visit the [PyTorch website](https://pytorch.org/get-started/locally/) for specific installation instructions for your system.

**Pygame Display Issues:**
- Some systems may require additional dependencies for Pygame. For Ubuntu/Debian: `sudo apt-get install python3-dev libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev`

**Performance Issues:**
- Reduce the screen resolution by modifying SCREEN_WIDTH and SCREEN_HEIGHT constants
- Decrease FPS for slower computers
- If training is too slow even with GPU, consider simplifying the neural network architecture

## License

This project is available under the MIT License. See the LICENSE file for more information.

## Acknowledgments

- This project demonstrates the application of Deep Q-Learning to a continuous control problem
- Inspired by various reinforcement learning research papers and tutorials
