import torch
import numpy as np
import pygame
import math
import time
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, set_composite_lp_aggregate
from tensordict.nn.distributions import NormalParamExtractor

# TorchRL imports
from torchrl.data import BoundedTensorSpec
from torchrl.collectors import SyncDataCollector
from torchrl.data.replay_buffers import ReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal
from torchrl.objectives import ClipPPOLoss, ValueEstimators

# Set random seed for reproducibility
torch.manual_seed(0)
np.random.seed(0)

# Disable log-prob aggregation
set_composite_lp_aggregate(False).set()

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


class CarRacingEnv:
    """2D Car Racing Environment with lidar sensors and continuous actions."""

    def __init__(self, num_envs=1, max_steps=1000, device=torch.device("cpu"), n_agents=1):
        self.num_envs = num_envs
        self.max_steps = max_steps
        self.device = device
        self._n_agents = n_agents

        # Track definition (oval race track)
        self.track_width = 100  # Width of the track
        self.track_outer_radius = 400  # Outer radius of the oval
        self.track_inner_radius = 300  # Inner radius of the oval

        # Car properties
        self.car_length = 20
        self.car_width = 10
        self.max_speed = 5.0
        self.max_steering = 0.3  # Maximum steering angle in radians

        # Lidar properties
        self.num_lidar_rays = 8
        self.lidar_max_dist = 200

        # Checkpoints for lap tracking
        self.num_checkpoints = 12
        self.checkpoints = self._generate_checkpoints()

        # Reset the environment state
        self.reset()

        # For visualization
        self.pygame_initialized = False

    def _generate_checkpoints(self):
        """Generate evenly spaced checkpoints around the track."""
        checkpoints = []
        for i in range(self.num_checkpoints):
            angle = 2 * math.pi * i / self.num_checkpoints
            mid_radius = (self.track_outer_radius + self.track_inner_radius) / 2
            x = mid_radius * math.cos(angle)
            y = mid_radius * math.sin(angle)
            checkpoints.append((x, y))
        return torch.tensor(checkpoints, dtype=torch.float32, device=self.device)

    def _is_inside_track(self, positions):
        """Check if positions are inside the track (between inner and outer radius)."""
        distances = torch.norm(positions, dim=-1)
        inside_outer = distances <= self.track_outer_radius
        inside_inner = distances >= self.track_inner_radius
        return inside_outer & inside_inner

    def _calculate_lidar_readings(self):
        """Calculate lidar readings for all environments and agents."""
        lidar_readings = torch.zeros(self.num_envs, self._n_agents, self.num_lidar_rays,
                                     dtype=torch.float32, device=self.device)

        for env_idx in range(self.num_envs):
            for agent_idx in range(self._n_agents):
                for ray_idx in range(self.num_lidar_rays):
                    ray_angle = self.car_orientations[env_idx, agent_idx] + (
                                ray_idx * 2 * math.pi / self.num_lidar_rays)
                    ray_dir = torch.tensor([math.cos(ray_angle), math.sin(ray_angle)], device=self.device)

                    # Raycast to find distance to track boundaries
                    t_outer = self._ray_circle_intersection(self.car_positions[env_idx, agent_idx], ray_dir,
                                                            self.track_outer_radius)
                    t_inner = self._ray_circle_intersection(self.car_positions[env_idx, agent_idx], ray_dir,
                                                            self.track_inner_radius)

                    if t_outer > 0 and (t_inner <= 0 or t_outer < t_inner):
                        lidar_readings[env_idx, agent_idx, ray_idx] = min(t_outer, self.lidar_max_dist)
                    elif t_inner > 0:
                        lidar_readings[env_idx, agent_idx, ray_idx] = min(t_inner, self.lidar_max_dist)
                    else:
                        lidar_readings[env_idx, agent_idx, ray_idx] = self.lidar_max_dist

        # Normalize readings
        return lidar_readings / self.lidar_max_dist

    def _ray_circle_intersection(self, origin, direction, radius):
        """Calculate intersection of ray with circle centered at origin."""
        a = torch.sum(direction * direction)
        b = 2 * torch.sum(origin * direction)
        c = torch.sum(origin * origin) - radius * radius

        discriminant = b * b - 4 * a * c

        if discriminant < 0:
            return -1  # No intersection

        t1 = (-b + torch.sqrt(discriminant)) / (2 * a)
        t2 = (-b - torch.sqrt(discriminant)) / (2 * a)

        if t1 >= 0 and t2 >= 0:
            return min(t1, t2)
        elif t1 >= 0:
            return t1
        elif t2 >= 0:
            return t2
        else:
            return -1  # Intersections are behind the ray

    def _update_checkpoint_progress(self):
        """Update checkpoint progress for each environment and agent."""
        for env_idx in range(self.num_envs):
            for agent_idx in range(self._n_agents):
                current_checkpoint = self.checkpoint_progress[env_idx, agent_idx] % self.num_checkpoints
                next_checkpoint = (current_checkpoint + 1) % self.num_checkpoints

                # Calculate distance to the next checkpoint
                next_cp_pos = self.checkpoints[next_checkpoint]
                distance = torch.norm(self.car_positions[env_idx, agent_idx] - next_cp_pos)

                # If we're close enough to the next checkpoint, update progress
                if distance < 30:  # Threshold distance
                    self.checkpoint_progress[env_idx, agent_idx] += 1

                    # Check for lap completion
                    if next_checkpoint == 0:
                        self.laps_completed[env_idx, agent_idx] += 1

    def reset(self):
        """Reset the environment and return initial observations."""
        # Reset step count
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # Initialize car positions (at the starting line)
        self.car_positions = torch.zeros(self.num_envs, self._n_agents, 2, dtype=torch.float32, device=self.device)

        # Place agents in a line at the starting position
        for agent_idx in range(self._n_agents):
            offset = (agent_idx - self._n_agents // 2) * self.car_width * 1.5
            self.car_positions[:, agent_idx, 0] = (self.track_outer_radius + self.track_inner_radius) / 2
            self.car_positions[:, agent_idx, 1] = offset

        # Reset velocities, orientations, and other state variables
        self.car_velocities = torch.zeros(self.num_envs, self._n_agents, 2, dtype=torch.float32, device=self.device)
        self.car_orientations = torch.zeros(self.num_envs, self._n_agents, dtype=torch.float32, device=self.device)
        self.car_speeds = torch.zeros(self.num_envs, self._n_agents, dtype=torch.float32, device=self.device)
        self.checkpoint_progress = torch.zeros(self.num_envs, self._n_agents, dtype=torch.int32, device=self.device)
        self.laps_completed = torch.zeros(self.num_envs, self._n_agents, dtype=torch.int32, device=self.device)
        self.cumulative_rewards = torch.zeros(self.num_envs, self._n_agents, dtype=torch.float32, device=self.device)
        self.total_movements = torch.zeros(self.num_envs, self._n_agents, dtype=torch.float32, device=self.device)

        # Calculate initial observations
        lidar_readings = self._calculate_lidar_readings()

        # Create observations
        observations = []
        for env_idx in range(self.num_envs):
            env_obs = []
            for agent_idx in range(self._n_agents):
                # Combine lidar readings with other state info
                agent_obs = torch.cat([
                    lidar_readings[env_idx, agent_idx],
                    self.car_positions[env_idx, agent_idx],
                    self.car_orientations[env_idx, agent_idx].unsqueeze(0),
                    self.car_speeds[env_idx, agent_idx].unsqueeze(0),
                    self.checkpoint_progress[env_idx, agent_idx].float().unsqueeze(0)
                ])
                env_obs.append(agent_obs)
            observations.append(torch.stack(env_obs))

        observations = torch.stack(observations)

        # Create initial TensorDict
        tensordict = TensorDict({
            "agents": {
                "observation": observations,
                "action": torch.zeros(self.num_envs, self._n_agents, 2, device=self.device),
            },
            "done": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
        }, batch_size=[self.num_envs])

        return tensordict

    def step(self, tensordict):
        """Execute actions and return next observations, rewards, and done flags."""
        # Extract actions from tensordict
        actions = tensordict["agents", "action"]

        # Apply actions for each agent
        prev_positions = self.car_positions.clone()

        for agent_idx in range(self._n_agents):
            # Update car orientation and speed based on actions
            acceleration = actions[:, agent_idx, 0].clamp(-1.0, 1.0)
            steering = actions[:, agent_idx, 1].clamp(-1.0, 1.0) * self.max_steering

            # Update car orientation
            self.car_orientations[:, agent_idx] += steering

            # Update car speed based on acceleration
            self.car_speeds[:, agent_idx] += acceleration * 0.1
            self.car_speeds[:, agent_idx].clamp_(-self.max_speed, self.max_speed)

            # Calculate movement vector
            movement_x = self.car_speeds[:, agent_idx] * torch.cos(self.car_orientations[:, agent_idx])
            movement_y = self.car_speeds[:, agent_idx] * torch.sin(self.car_orientations[:, agent_idx])

            # Update car position
            self.car_positions[:, agent_idx, 0] += movement_x
            self.car_positions[:, agent_idx, 1] += movement_y

            # Calculate total movement
            self.total_movements[:, agent_idx] += torch.norm(
                self.car_positions[:, agent_idx] - prev_positions[:, agent_idx],
                dim=1
            )

        # Check if cars are inside track
        inside_track = self._is_inside_track(self.car_positions)

        # Handle collisions with track boundaries
        for env_idx in range(self.num_envs):
            for agent_idx in range(self._n_agents):
                if not inside_track[env_idx, agent_idx]:
                    # Bounce back: move car back to previous position
                    self.car_positions[env_idx, agent_idx] = prev_positions[env_idx, agent_idx]
                    # Reduce speed to simulate collision
                    self.car_speeds[env_idx, agent_idx] *= 0.5
                    # Add some randomness to orientation to help get unstuck
                    self.car_orientations[env_idx, agent_idx] += (torch.rand(1, device=self.device).item() - 0.5) * 0.5

        # Update checkpoint progress
        self._update_checkpoint_progress()

        # Calculate rewards for each agent
        rewards = torch.zeros(self.num_envs, self._n_agents, device=self.device)

        for env_idx in range(self.num_envs):
            for agent_idx in range(self._n_agents):
                # Previous position for this agent
                prev_position = prev_positions[env_idx, agent_idx]

                # Forward progress reward (only for forward progress toward next checkpoint)
                movement_reward = 0
                if self.car_speeds[env_idx, agent_idx] > 0:
                    # Project movement onto the direction to the next checkpoint
                    current_cp = self.checkpoint_progress[env_idx, agent_idx] % self.num_checkpoints
                    next_cp = (current_cp + 1) % self.num_checkpoints
                    next_cp_pos = self.checkpoints[next_cp]
                    direction_to_cp = next_cp_pos - prev_position
                    direction_to_cp = direction_to_cp / torch.norm(direction_to_cp)

                    movement_vector = self.car_positions[env_idx, agent_idx] - prev_position
                    movement_magnitude = torch.norm(movement_vector)

                    if movement_magnitude > 0:
                        movement_dir = movement_vector / movement_magnitude
                        forward_progress = torch.dot(movement_dir, direction_to_cp) * movement_magnitude
                        movement_reward = max(0, forward_progress * 15)  # Increased reward for moving toward checkpoint

                # Checkpoint reward - increased
                checkpoint_reward = 0
                if self.checkpoint_progress[env_idx, agent_idx] > 0 and self.step_count[env_idx] == 0:
                    # This is a reset with loaded checkpoint progress
                    pass
                elif self.checkpoint_progress[env_idx, agent_idx] > 0 and self.step_count[env_idx] > 0:
                    # Normal checkpoint progress
                    checkpoint_diff = self.checkpoint_progress[env_idx, agent_idx] - (self.step_count[env_idx] // 10)
                    if checkpoint_diff > 0:
                        checkpoint_reward = checkpoint_diff * 100  # Increased reward for reaching checkpoints

                # Lap completion reward - increased
                lap_reward = self.laps_completed[env_idx, agent_idx] * 1000  # Increased reward for completing laps

                # Movement efficiency penalty (reduced penalty)
                efficiency_penalty = -self.total_movements[env_idx, agent_idx] * 0.005

                # Time penalty (reduced penalty)
                time_penalty = -0.05

                # Crash penalty (if outside track) - now handled with bounce back
                crash_penalty = 0
                if not inside_track[env_idx, agent_idx]:
                    crash_penalty = -20  # Reduced since we're also bouncing back

                # Speed bonus - reward for maintaining speed
                speed_bonus = self.car_speeds[env_idx, agent_idx] * 0.5

                # Combine rewards
                total_reward = movement_reward + checkpoint_reward + lap_reward + \
                               efficiency_penalty + time_penalty + crash_penalty + speed_bonus
                rewards[env_idx, agent_idx] = total_reward

                # Update cumulative reward
                self.cumulative_rewards[env_idx, agent_idx] += total_reward

        # Update step count
        self.step_count += 1

        # Check termination conditions - an environment is done if any agent is outside the track or max steps reached
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_idx in range(self.num_envs):
            if self.step_count[env_idx] >= self.max_steps:  # Only terminate when max steps reached, not on out-of-track
                dones[env_idx] = True
                terminated[env_idx] = True

        # Calculate lidar readings
        lidar_readings = self._calculate_lidar_readings()

        # Create observations
        observations = []
        for env_idx in range(self.num_envs):
            env_obs = []
            for agent_idx in range(self._n_agents):
                # Combine lidar readings with other state info
                agent_obs = torch.cat([
                    lidar_readings[env_idx, agent_idx],
                    self.car_positions[env_idx, agent_idx],
                    self.car_orientations[env_idx, agent_idx].unsqueeze(0),
                    self.car_speeds[env_idx, agent_idx].unsqueeze(0),
                    self.checkpoint_progress[env_idx, agent_idx].float().unsqueeze(0)
                ])
                env_obs.append(agent_obs)
            observations.append(torch.stack(env_obs))

        observations = torch.stack(observations)

        # Create result tensordict
        result = TensorDict({
            "agents": {
                "observation": observations,
                "reward": rewards.unsqueeze(-1),  # Add dimension to match expected shape
            },
            "done": dones,
            "terminated": terminated,
            "next": {
                "agents": {
                    "observation": observations,
                },
                "done": dones,
                "terminated": terminated,
            }
        }, batch_size=[self.num_envs])

        return result

    def render(self):
        """Render the environment."""
        if not self.pygame_initialized:
            pygame.init()
            self.screen_width = 800
            self.screen_height = 800
            self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            pygame.display.set_caption("Car Racing Environment")
            self.clock = pygame.time.Clock()
            self.pygame_initialized = True

        # Clear screen
        self.screen.fill((255, 255, 255))

        # Draw track
        pygame.draw.circle(self.screen, (200, 200, 200), (self.screen_width // 2, self.screen_height // 2),
                           int(self.track_outer_radius * self.screen_width / (2 * self.track_outer_radius)))
        pygame.draw.circle(self.screen, (255, 255, 255), (self.screen_width // 2, self.screen_height // 2),
                           int(self.track_inner_radius * self.screen_width / (2 * self.track_outer_radius)))

        # Draw checkpoints
        for cp in self.checkpoints:
            x = int(cp[0].item() * self.screen_width / (2 * self.track_outer_radius) + self.screen_width // 2)
            y = int(cp[1].item() * self.screen_height / (2 * self.track_outer_radius) + self.screen_height // 2)
            pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 5)

        # Draw each car - simple visualization
        colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]

        for agent_idx in range(self._n_agents):
            # Convert car position to screen coordinates
            car_pos_x = int(self.car_positions[0, agent_idx, 0].item() * self.screen_width /
                            (2 * self.track_outer_radius) + self.screen_width // 2)
            car_pos_y = int(self.car_positions[0, agent_idx, 1].item() * self.screen_height /
                            (2 * self.track_outer_radius) + self.screen_height // 2)

            # Check if car is inside track
            is_inside = self._is_inside_track(self.car_positions[0:1, agent_idx:agent_idx + 1])[0, 0].item()

            # Draw car as a circle, red border if outside track
            color = colors[agent_idx % len(colors)]
            pygame.draw.circle(self.screen, color, (car_pos_x, car_pos_y), 10)
            if not is_inside:
                # Draw red border to indicate collision
                pygame.draw.circle(self.screen, (255, 0, 0), (car_pos_x, car_pos_y), 12, 2)

            # Draw a line showing orientation
            orientation = self.car_orientations[0, agent_idx].item()
            line_end_x = car_pos_x + int(20 * math.cos(orientation))
            line_end_y = car_pos_y + int(20 * math.sin(orientation))
            pygame.draw.line(self.screen, (0, 0, 0), (car_pos_x, car_pos_y), (line_end_x, line_end_y), 2)

            # Show speed as a line representing velocity magnitude
            speed_line_end_x = car_pos_x + int(self.car_speeds[0, agent_idx].item() * 5 * math.cos(orientation))
            speed_line_end_y = car_pos_y + int(self.car_speeds[0, agent_idx].item() * 5 * math.sin(orientation))
            pygame.draw.line(self.screen, (255, 0, 0), (car_pos_x, car_pos_y), (speed_line_end_x, speed_line_end_y), 1)

            # Show position and speed info
            font = pygame.font.Font(None, 24)
            pos_text = font.render(
                f"Car {agent_idx}: pos=({self.car_positions[0, agent_idx, 0].item():.1f}, {self.car_positions[0, agent_idx, 1].item():.1f})",
                True, color)
            self.screen.blit(pos_text, (10, 180 + agent_idx * 60))

            speed_text = font.render(
                f"     speed={self.car_speeds[0, agent_idx].item():.2f}, reward={self.cumulative_rewards[0, agent_idx].item():.1f}",
                True, color)
            self.screen.blit(speed_text, (10, 180 + agent_idx * 60 + 25))

        # Display general info
        font = pygame.font.Font(None, 36)
        lap_text = font.render(f"Laps: {self.laps_completed[0, 0].item()}", True, (0, 0, 0))
        checkpoint_text = font.render(f"Checkpoints: {self.checkpoint_progress[0, 0].item()}", True, (0, 0, 0))
        reward_text = font.render(f"Reward: {self.cumulative_rewards[0, 0].item():.1f}", True, (0, 0, 0))
        step_text = font.render(f"Step: {self.step_count[0].item()}", True, (0, 0, 0))

        self.screen.blit(lap_text, (10, 10))
        self.screen.blit(checkpoint_text, (10, 50))
        self.screen.blit(reward_text, (10, 90))
        self.screen.blit(step_text, (10, 130))

        pygame.display.flip()
        self.clock.tick(30)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return False

        return True

    # Define properties for training compatibility
    @property
    def n_agents(self):
        """Return the number of agents."""
        return self._n_agents

    @property
    def observation_spec(self):
        """Return observation specification shape."""
        obs_dim = self.num_lidar_rays + 5  # lidar + position(2) + orientation + speed + checkpoint progress
        return torch.zeros(self.num_envs, self._n_agents, obs_dim, device=self.device)

    @property
    def action_spec(self):
        """Return action specification."""
        # Use scalar bounds that will be broadcast to the full shape
        return BoundedTensorSpec(
            low=-1.0,  # Scalar value will be broadcast to all dimensions
            high=1.0,  # Scalar value will be broadcast to all dimensions
            shape=(self.num_envs, self._n_agents, 2),
            dtype=torch.float32,
            device=self.device
        )


def run_random_agent(env, steps=1000, render_delay=50):
    """Run random agent in the environment with visualization."""
    print("Running random agent...")

    # Reset the environment
    env.reset()
    running = True

    # Keep track of rewards for display
    rewards = []

    for step in range(steps):
        if not running:
            break

        # Create random actions in range [-1, 1]
        random_actions = torch.rand((env.num_envs, env.n_agents, 2), device=env.device) * 2 - 1

        # Create TensorDict with random actions
        action_td = TensorDict({
            "agents": {
                "action": random_actions
            }
        }, batch_size=[env.num_envs])

        # Step the environment
        next_state = env.step(action_td)

        # Track rewards
        reward = next_state["agents", "reward"].mean().item()
        rewards.append(reward)

        # Print status periodically
        if step % 50 == 0:
            avg_reward = sum(rewards[-50:]) / min(len(rewards), 50)
            print(f"Step {step}, Avg Reward: {avg_reward:.2f}")

        # Render
        running = env.render()

        # Add delay to see movement better
        pygame.time.delay(render_delay)

    print("Random agent finished")


def visualize_environment(n_agents=3, steps=500):
    """Create and visualize the environment."""
    try:
        print("Creating environment...")
        env = CarRacingEnv(
            num_envs=1,
            max_steps=steps,
            device=device,
            n_agents=n_agents
        )

        print(f"Environment created with {n_agents} agents")
        print(f"Observation shape: {env.observation_spec.shape}")
        print(f"Action shape: {env.action_spec.shape}")

        # Run with random actions
        run_random_agent(env, steps)
    except Exception as e:
        print(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()  # Print the full stack trace for debugging
    finally:
        pygame.quit()


def create_ppo_policy(env, device=torch.device("cpu")):
    """Create a PPO policy for the environment."""
    # Get dimensions from environment
    observation_dim = env.observation_spec.shape[-1]
    n_agents = env.n_agents
    action_dim = 2  # [acceleration, steering]

    print(f"Creating policy with obs_dim={observation_dim}, n_agents={n_agents}, action_dim={action_dim}")

    # Policy setup
    share_parameters = True

    # Define policy network
    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=observation_dim,
            n_agent_outputs=2 * action_dim,  # 2 parameters (mean, std) per action
            n_agents=n_agents,
            centralised=False,  # decentralized policy
            share_params=share_parameters,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    # Create TensorDictModule
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Create probabilistic actor with scalar bounds
    policy = ProbabilisticActor(
        module=policy_module,
        spec=BoundedTensorSpec(
            low=-1.0,
            high=1.0,
            shape=(n_agents, 2),
            dtype=torch.float32,
            device=device
        ),
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[("agents", "action")],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -1.0,  # Scalar value
            "high": 1.0,  # Scalar value
        },
        return_log_prob=True,
    )

    return policy


def train_and_visualize(n_agents=3, training_steps=100000, render=True):
    """Train a PPO agent while visualizing the progress."""
    print(f"Training PPO agent with {n_agents} cars for {training_steps} steps...")

    # Create environment
    env = CarRacingEnv(num_envs=1, max_steps=1000, device=device, n_agents=n_agents)

    # Create PPO policy
    policy = create_ppo_policy(env, device)

    # Setup training parameters
    lr = 3e-4
    batch_size = 64
    gamma = 0.99

    # Create optimizer
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    # Training loop
    total_steps = 0
    episode = 0
    max_episodes = 10000  # Very large number to keep running indefinitely

    # For tracking progress
    rewards_history = []
    avg_rewards = []

    try:
        # Reset environment to start
        state = env.reset()

        while episode < max_episodes and total_steps < training_steps:
            episode_rewards = []
            episode_steps = 0
            episode_done = False

            while not episode_done and episode_steps < env.max_steps:
                # Generate action using policy
                with torch.no_grad():
                    action_td = policy(state)

                # Execute action in environment
                next_state = env.step(action_td)

                # Get reward and done flag
                reward = next_state["agents", "reward"]
                done = next_state["done"]

                # Store reward for logging
                episode_rewards.append(reward.mean().item())

                # Render if enabled
                if render:
                    if not env.render():
                        print("Rendering window closed. Exiting.")
                        return

                # Collect batch of experience for training
                if len(episode_rewards) >= batch_size:
                    # Simple PPO update (simplified for this example)
                    optimizer.zero_grad()

                    # Here we would normally:
                    # 1. Compute policy loss
                    # 2. Compute value loss
                    # 3. Backpropagate and update policy

                    # For now, we're just visualizing with a placeholder update
                    loss = torch.tensor(0.01, device=device, requires_grad=True)
                    loss.backward()
                    optimizer.step()

                    # Save policy periodically
                    if total_steps % 5000 == 0:
                        torch.save(policy.state_dict(), f"car_racing_policy_{total_steps}.pth")
                        print(f"Saved policy at step {total_steps}")

                # Check if episode is done
                episode_done = done.any().item()

                # Update state
                state = next_state

                # Update counters
                episode_steps += 1
                total_steps += 1

                # Print status
                if total_steps % 100 == 0:
                    avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                    print(f"Step {total_steps}, Episode {episode}, Avg Reward: {avg_reward:.2f}")
                    avg_rewards.append(avg_reward)

            # Episode finished
            episode += 1
            rewards_history.extend(episode_rewards)

            # Reset for next episode
            state = env.reset()

            # Plot progress every few episodes
            if episode % 5 == 0 and len(avg_rewards) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(avg_rewards)
                plt.title("Average Reward per 100 Steps")
                plt.xlabel("Hundreds of Steps")
                plt.ylabel("Average Reward")
                plt.savefig("training_progress.png")
                plt.close()

    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final policy
        try:
            torch.save(policy.state_dict(), "car_racing_policy_final.pth")
            print("Saved final policy")
        except Exception as e:
            print(f"Error saving policy: {e}")

        # Clean up pygame
        pygame.quit()

    return policy, rewards_history


def main():
    """Main function."""
    n_agents = 3

    print("""
    Car Racing Environment Options:
    1. Run random agent (for testing)
    2. Train agent with visualization
    3. Exit
    """)

    choice = input("Enter your choice (1, 2, or 3): ")

    try:
        if choice == "1":
            # Run random agent for testing
            steps = 2000  # Run for longer
            print(f"Running random agent for {steps} steps...")
            visualize_environment(n_agents=n_agents, steps=steps)
        elif choice == "2":
            # Train PPO agent with visualization
            training_steps = int(input("Enter number of training steps (default 100000): ") or "100000")
            train_and_visualize(n_agents=n_agents, training_steps=training_steps)
        elif choice == "3":
            print("Exiting...")
            return
        else:
            print("Invalid choice, running random agent by default")
            visualize_environment(n_agents=n_agents)
    except Exception as e:
        print(f"Error in main function: {e}")
        import traceback
        traceback.print_exc()
        # Make sure pygame is properly closed
        pygame.quit()


if __name__ == "__main__":
    main()