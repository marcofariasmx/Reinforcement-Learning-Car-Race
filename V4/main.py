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
                        movement_reward = max(0, forward_progress * 10)  # Reward for moving toward checkpoint

                # Checkpoint reward
                checkpoint_reward = 0
                if self.checkpoint_progress[env_idx, agent_idx] > 0 and self.step_count[env_idx] == 0:
                    # This is a reset with loaded checkpoint progress
                    pass
                elif self.checkpoint_progress[env_idx, agent_idx] > 0 and self.step_count[env_idx] > 0:
                    # Normal checkpoint progress
                    checkpoint_diff = self.checkpoint_progress[env_idx, agent_idx] - (self.step_count[env_idx] // 10)
                    if checkpoint_diff > 0:
                        checkpoint_reward = checkpoint_diff * 50  # Reward for reaching checkpoints

                # Lap completion reward
                lap_reward = self.laps_completed[env_idx, agent_idx] * 500  # Big reward for completing laps

                # Movement efficiency penalty (small penalty for excessive movement)
                efficiency_penalty = -self.total_movements[env_idx, agent_idx] * 0.01

                # Time penalty (small penalty for each step)
                time_penalty = -0.1

                # Crash penalty (if outside track)
                crash_penalty = 0
                if not inside_track[env_idx, agent_idx]:
                    crash_penalty = -100

                # Combine rewards
                total_reward = movement_reward + checkpoint_reward + lap_reward + efficiency_penalty + time_penalty + crash_penalty
                rewards[env_idx, agent_idx] = total_reward

                # Update cumulative reward
                self.cumulative_rewards[env_idx, agent_idx] += total_reward

        # Update step count
        self.step_count += 1

        # Check termination conditions - an environment is done if any agent is outside the track or max steps reached
        dones = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        terminated = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        for env_idx in range(self.num_envs):
            if (not inside_track[env_idx].all()) or self.step_count[env_idx] >= self.max_steps:
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
            car_pos_x = int(self.car_positions[0, agent_idx, 0].item() * self.screen_width /
                            (2 * self.track_outer_radius) + self.screen_width // 2)
            car_pos_y = int(self.car_positions[0, agent_idx, 1].item() * self.screen_height /
                            (2 * self.track_outer_radius) + self.screen_height // 2)

            # Draw car as a circle for easier visualization
            color = colors[agent_idx % len(colors)]
            pygame.draw.circle(self.screen, color, (car_pos_x, car_pos_y), 10)

            # Draw a line showing orientation
            orientation = self.car_orientations[0, agent_idx].item()
            line_end_x = car_pos_x + int(20 * math.cos(orientation))
            line_end_y = car_pos_y + int(20 * math.sin(orientation))
            pygame.draw.line(self.screen, (0, 0, 0), (car_pos_x, car_pos_y), (line_end_x, line_end_y), 2)

            # Show position and speed info
            font = pygame.font.Font(None, 24)
            pos_text = font.render(
                f"Car {agent_idx}: pos=({self.car_positions[0, agent_idx, 0].item():.1f}, {self.car_positions[0, agent_idx, 1].item():.1f})",
                True, color)
            self.screen.blit(pos_text, (10, 180 + agent_idx * 60))

            speed_text = font.render(f"     speed={self.car_speeds[0, agent_idx].item():.2f}, orient={orientation:.2f}",
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
        env.step(action_td)

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

        # Skip policy creation for now to avoid potential errors
        # Just run with random actions
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


def main():
    """Main function."""
    n_agents = 3

    # Just visualize the environment with random actions
    try:
        visualize_environment(n_agents=n_agents)
    except Exception as e:
        print(f"Error in main function: {e}")
        # Make sure pygame is properly closed
        pygame.quit()


if __name__ == "__main__":
    main()