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

        # Show direction of travel with track arrows
        for i in range(8):
            angle = 2 * math.pi * i / 8
            mid_radius = (self.track_outer_radius + self.track_inner_radius) / 2
            arrow_start_x = int(mid_radius * math.cos(angle) * self.screen_width / (
                        2 * self.track_outer_radius) + self.screen_width // 2)
            arrow_start_y = int(mid_radius * math.sin(angle) * self.screen_height / (
                        2 * self.track_outer_radius) + self.screen_height // 2)

            # Draw direction arrow (CCW around track)
            arrow_angle = angle + math.pi / 2  # Perpendicular to radius, CCW
            arrow_length = 20
            arrow_end_x = arrow_start_x + int(arrow_length * math.cos(arrow_angle))
            arrow_end_y = arrow_start_y + int(arrow_length * math.sin(arrow_angle))

            # Draw arrow
            pygame.draw.line(self.screen, (100, 100, 100), (arrow_start_x, arrow_start_y), (arrow_end_x, arrow_end_y),
                             2)
            # Draw arrowhead
            pygame.draw.polygon(self.screen, (100, 100, 100), [
                (arrow_end_x, arrow_end_y),
                (arrow_end_x - 8 * math.cos(arrow_angle - math.pi / 6),
                 arrow_end_y - 8 * math.sin(arrow_angle - math.pi / 6)),
                (arrow_end_x - 8 * math.cos(arrow_angle + math.pi / 6),
                 arrow_end_y - 8 * math.sin(arrow_angle + math.pi / 6))
            ])

        # Draw checkpoints with numbers
        for i, cp in enumerate(self.checkpoints):
            x = int(cp[0].item() * self.screen_width / (2 * self.track_outer_radius) + self.screen_width // 2)
            y = int(cp[1].item() * self.screen_height / (2 * self.track_outer_radius) + self.screen_height // 2)

            # Draw checkpoint
            pygame.draw.circle(self.screen, (0, 0, 255), (x, y), 5)

            # Draw checkpoint number
            font = pygame.font.Font(None, 20)
            number_text = font.render(str(i), True, (0, 0, 0))
            self.screen.blit(number_text, (x + 10, y))

        # Draw each car - improved visualization
        colors = [(255, 0, 0), (0, 128, 0), (0, 0, 255), (255, 165, 0), (128, 0, 128)]

        for agent_idx in range(self._n_agents):
            # Convert car position to screen coordinates
            car_pos_x = int(self.car_positions[0, agent_idx, 0].item() * self.screen_width /
                            (2 * self.track_outer_radius) + self.screen_width // 2)
            car_pos_y = int(self.car_positions[0, agent_idx, 1].item() * self.screen_height /
                            (2 * self.track_outer_radius) + self.screen_height // 2)

            # Check if car is inside track
            is_inside = self._is_inside_track(self.car_positions[0:1, agent_idx:agent_idx + 1])[0, 0].item()

            # Calculate car shape for a more obvious car representation
            orientation = self.car_orientations[0, agent_idx].item()
            car_radius = 12  # Base car size

            # Main car body as a rounded rectangle
            # Front of car (pointed)
            front_x = car_pos_x + int(car_radius * 1.5 * math.cos(orientation))
            front_y = car_pos_y + int(car_radius * 1.5 * math.sin(orientation))

            # Back of car
            back_x = car_pos_x - int(car_radius * math.cos(orientation))
            back_y = car_pos_y - int(car_radius * math.sin(orientation))

            # Calculate corners for the car body (more rectangular)
            width_factor = 0.7  # How wide the car is relative to length

            # Side points (perpendicular to orientation)
            right_x = int(car_radius * width_factor * math.cos(orientation + math.pi / 2))
            right_y = int(car_radius * width_factor * math.sin(orientation + math.pi / 2))
            left_x = int(car_radius * width_factor * math.cos(orientation - math.pi / 2))
            left_y = int(car_radius * width_factor * math.sin(orientation - math.pi / 2))

            # Four corner points of car body
            front_right = (front_x + right_x, front_y + right_y)
            front_left = (front_x + left_x, front_y + left_y)
            back_right = (back_x + right_x, back_y + right_y)
            back_left = (back_x + left_x, back_y + left_y)

            # Get car color
            color = colors[agent_idx % len(colors)]

            # Draw car body
            pygame.draw.polygon(self.screen, color, [front_right, front_left, back_left, back_right])

            # Draw front windshield (to indicate front clearly)
            windshield_front_x = car_pos_x + int(car_radius * 0.8 * math.cos(orientation))
            windshield_front_y = car_pos_y + int(car_radius * 0.8 * math.sin(orientation))
            wind_right_x = int(car_radius * 0.5 * math.cos(orientation + math.pi / 2))
            wind_right_y = int(car_radius * 0.5 * math.sin(orientation + math.pi / 2))
            wind_left_x = int(car_radius * 0.5 * math.cos(orientation - math.pi / 2))
            wind_left_y = int(car_radius * 0.5 * math.sin(orientation - math.pi / 2))

            windshield = [
                (windshield_front_x + wind_right_x, windshield_front_y + wind_right_y),
                (windshield_front_x + wind_left_x, windshield_front_y + wind_left_y),
                (car_pos_x + wind_left_x, car_pos_y + wind_left_y),
                (car_pos_x + wind_right_x, car_pos_y + wind_right_y)
            ]
            pygame.draw.polygon(self.screen, (100, 100, 255), windshield)  # Blue windshield

            # Draw headlights at front
            headlight_r_pos = (front_right[0] + back_right[0]) // 2, (front_right[1] + back_right[1]) // 2
            headlight_l_pos = (front_left[0] + back_left[0]) // 2, (front_left[1] + back_left[1]) // 2
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (int(front_right[0] * 0.9 + front_x * 0.1),
                                int(front_right[1] * 0.9 + front_y * 0.1)), 3)
            pygame.draw.circle(self.screen, (255, 255, 0),
                               (int(front_left[0] * 0.9 + front_x * 0.1),
                                int(front_left[1] * 0.9 + front_y * 0.1)), 3)

            # Draw direction arrow (outside the car, pointing in the direction of motion)
            # Only draw if speed is significant
            if abs(self.car_speeds[0, agent_idx].item()) > 0.1:
                # Color based on forward/backward
                motion_color = (0, 255, 0) if self.car_speeds[0, agent_idx].item() > 0 else (255, 0, 0)

                # Direction of movement (considering car orientation)
                motion_direction = orientation if self.car_speeds[0, agent_idx].item() > 0 else orientation + math.pi

                # Draw arrow showing movement direction
                arrow_length = min(20, abs(self.car_speeds[0, agent_idx].item() * 10))
                arrow_start_x = front_x if self.car_speeds[0, agent_idx].item() > 0 else back_x
                arrow_start_y = front_y if self.car_speeds[0, agent_idx].item() > 0 else back_y
                arrow_end_x = arrow_start_x + int(arrow_length * math.cos(motion_direction))
                arrow_end_y = arrow_start_y + int(arrow_length * math.sin(motion_direction))

                # Arrow line
                pygame.draw.line(self.screen, motion_color, (arrow_start_x, arrow_start_y),
                                 (arrow_end_x, arrow_end_y), 3)

                # Arrowhead
                pygame.draw.polygon(self.screen, motion_color, [
                    (arrow_end_x, arrow_end_y),
                    (int(arrow_end_x - 8 * math.cos(motion_direction - math.pi / 6)),
                     int(arrow_end_y - 8 * math.sin(motion_direction - math.pi / 6))),
                    (int(arrow_end_x - 8 * math.cos(motion_direction + math.pi / 6)),
                     int(arrow_end_y - 8 * math.sin(motion_direction + math.pi / 6)))
                ])

            # Draw red border if outside track
            if not is_inside:
                pygame.draw.polygon(self.screen, (255, 0, 0), [front_right, front_left, back_left, back_right], 2)

            # Draw lidar sensors
            for ray_idx in range(self.num_lidar_rays):
                ray_angle = orientation + (ray_idx * 2 * math.pi / self.num_lidar_rays)
                ray_dir = torch.tensor([math.cos(ray_angle), math.sin(ray_angle)], device=self.device)

                # Get lidar reading (denormalized)
                lidar_reading = self._calculate_lidar_readings()[0, agent_idx, ray_idx] * self.lidar_max_dist

                # Calculate endpoint based on lidar reading
                ray_end_x = car_pos_x + int(
                    lidar_reading.item() * math.cos(ray_angle) * self.screen_width / (2 * self.track_outer_radius))
                ray_end_y = car_pos_y + int(
                    lidar_reading.item() * math.sin(ray_angle) * self.screen_height / (2 * self.track_outer_radius))

                # Draw lidar ray
                pygame.draw.line(self.screen, (0, 255, 0), (car_pos_x, car_pos_y), (ray_end_x, ray_end_y), 1)

            # Show speed indicator - length and color based on speed
            speed_magnitude = self.car_speeds[0, agent_idx].item()
            speed_line_length = min(int(abs(speed_magnitude) * 20), 50)  # Capped length

            # Color based on speed (green for forward, red for backward)
            speed_color = (0, 255, 0) if speed_magnitude > 0 else (255, 0, 0)
            speed_line_end_x = car_pos_x + int(speed_line_length * math.cos(orientation))
            speed_line_end_y = car_pos_y + int(speed_line_length * math.sin(orientation))

            # Draw speed indicator line
            pygame.draw.line(self.screen, speed_color, (car_pos_x, car_pos_y), (speed_line_end_x, speed_line_end_y), 3)

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

        # Add direction information
        direction_text = font.render("Race Direction: Counter-Clockwise", True, (0, 0, 0))

        self.screen.blit(lap_text, (10, 10))
        self.screen.blit(checkpoint_text, (10, 50))
        self.screen.blit(reward_text, (10, 90))
        self.screen.blit(step_text, (10, 130))
        self.screen.blit(direction_text, (400, 10))

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
        return_log_prob=True,  # This adds action_log_prob to the output
    )

    return policy


def train_and_visualize(n_agents=3, training_steps=100000, render=True):
    """Train a PPO agent while visualizing the progress."""
    print(f"Training PPO agent with {n_agents} cars for {training_steps} steps...")

    # Create environment
    env = CarRacingEnv(num_envs=1, max_steps=1000, device=device, n_agents=n_agents)

    # Hyperparameters
    lr = 3e-4
    gamma = 0.99
    gae_lambda = 0.95
    clip_epsilon = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    ppo_epochs = 4
    batch_size = 64
    max_grad_norm = 0.5  # Maximum gradient norm for clipping

    # Create policy and value networks
    observation_dim = env.observation_spec.shape[-1]
    action_dim = 2  # [acceleration, steering]

    print(f"Creating networks with obs_dim={observation_dim}, n_agents={n_agents}, action_dim={action_dim}")

    # Policy network setup - using parameter sharing across agents
    share_params_policy = True  # Share parameters between agents for faster learning

    policy_net = torch.nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=observation_dim,
            n_agent_outputs=2 * action_dim,  # 2 parameters (mean, std) per action
            n_agents=n_agents,
            centralised=False,  # decentralized policy (each agent decides based on its own observation)
            share_params=share_params_policy,
            device=device,
            depth=2,
            num_cells=256,
            activation_class=torch.nn.Tanh,
        ),
        NormalParamExtractor(),
    )

    # Create policy module
    policy_module = TensorDictModule(
        policy_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )

    # Create probabilistic actor
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
            "low": -1.0,
            "high": 1.0,
        },
        return_log_prob=True,
    )

    # Value network (critic)
    # Using centralized critic (MAPPO approach) with parameter sharing
    share_params_critic = True
    mappo = True  # Using MAPPO (centralized critic) approach

    value_net = MultiAgentMLP(
        n_agent_inputs=observation_dim,
        n_agent_outputs=1,  # 1 value per agent
        n_agents=n_agents,
        centralised=mappo,  # Centralized critic for MAPPO
        share_params=share_params_critic,
        device=device,
        depth=2,
        num_cells=256,
        activation_class=torch.nn.Tanh,
    )

    value_module = TensorDictModule(
        value_net,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "value")]
    )

    # Create optimizers with adjusted learning rates
    policy_optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_net.parameters(), lr=lr)

    # Buffers for experience collection - use lists of tensors instead of TensorDicts
    observations_buffer = []
    actions_buffer = []
    log_probs_buffer = []
    rewards_buffer = []
    dones_buffer = []
    values_buffer = []

    # Training tracking variables
    total_steps = 0
    episode = 0
    max_episodes = 10000  # Very large number to keep running indefinitely

    # For tracking progress
    rewards_history = []
    avg_rewards = []
    episode_rewards = []
    best_avg_reward = float('-inf')

    try:
        # Reset environment to start
        state = env.reset()

        while episode < max_episodes and total_steps < training_steps:
            episode_rewards = []
            episode_steps = 0
            episode_done = False

            # Collect experience for one episode
            while not episode_done and episode_steps < env.max_steps:
                # Get action and value from networks
                with torch.no_grad():
                    # Forward pass through policy
                    action_td = policy(state)
                    action = action_td["agents", "action"]
                    log_prob = action_td["agents", "action_log_prob"]  # Fixed key name

                    # Forward pass through value network
                    value_td = value_module(state)
                    value = value_td["agents", "value"]

                # Execute action in environment
                next_state = env.step(action_td)

                # Get reward and done flag
                reward = next_state["agents", "reward"]
                done = next_state["done"]
                terminated = next_state["terminated"]

                # Store experience - extract just the tensors we need
                observations_buffer.append(state["agents", "observation"].clone())
                actions_buffer.append(action.clone())
                log_probs_buffer.append(log_prob.clone())
                rewards_buffer.append(reward.clone())
                # Store done as a scalar boolean per env (simplify dimensionality)
                dones_buffer.append(done.any().item())  # Just store True/False
                values_buffer.append(value.clone())

                # Track reward for logging
                episode_rewards.append(reward.mean().item())

                # Render if enabled
                if render and total_steps % 5 == 0:  # Render every 5 steps to speed up training
                    if not env.render():
                        print("Rendering window closed. Exiting.")
                        return

                # Check if episode is done
                episode_done = done.any().item()

                # Update state for next step
                state = next_state

                # Update counters
                episode_steps += 1
                total_steps += 1

                # Print status periodically
                if total_steps % 100 == 0:
                    avg_reward = sum(episode_rewards[-100:]) / min(len(episode_rewards), 100)
                    print(f"Step {total_steps}, Episode {episode}, Avg Reward: {avg_reward:.2f}")
                    avg_rewards.append(avg_reward)

                    # Save best policy
                    if avg_reward > best_avg_reward:
                        best_avg_reward = avg_reward
                        torch.save(policy.state_dict(), "car_racing_best_policy.pth")
                        print(f"Saved best policy with avg reward: {best_avg_reward:.2f}")

                # Check if we have enough experience for a PPO update
                if len(observations_buffer) >= batch_size:
                    print(f"Performing PPO update with batch size {len(observations_buffer)}")
                    try:
                        # Perform PPO update with collected batch
                        update_stats = perform_ppo_update(
                            observations_buffer, actions_buffer, log_probs_buffer, rewards_buffer, dones_buffer,
                            values_buffer,
                            policy, policy_optimizer, value_module, value_optimizer,
                            gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef,
                            ppo_epochs, batch_size, device
                        )
                        print("PPO update completed successfully")
                    except Exception as e:
                        print(f"Error during PPO update: {e}")
                        import traceback
                        traceback.print_exc()
                        # Continue training despite errors

                    # Clear experience buffers after update
                    observations_buffer = []
                    actions_buffer = []
                    log_probs_buffer = []
                    rewards_buffer = []
                    dones_buffer = []
                    values_buffer = []

                    # Save policy periodically
                    if total_steps % 5000 == 0:
                        torch.save(policy.state_dict(), f"car_racing_policy_{total_steps}.pth")
                        print(f"Saved policy checkpoint at step {total_steps}")

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

            # Plot final learning curve
            if len(avg_rewards) > 0:
                plt.figure(figsize=(10, 5))
                plt.plot(avg_rewards)
                plt.title("Training Progress")
                plt.xlabel("Updates")
                plt.ylabel("Average Reward")
                plt.savefig("final_training_progress.png")
                plt.close()

        except Exception as e:
            print(f"Error saving policy: {e}")

        # Clean up pygame
        pygame.quit()

    return policy, rewards_history


def perform_ppo_update(observations, actions, log_probs_old, rewards, dones, values,
                       policy, policy_optimizer, value_network, value_optimizer,
                       gamma, gae_lambda, clip_epsilon, entropy_coef, value_coef,
                       ppo_epochs, batch_size, device):
    """Perform PPO update on collected batch of experience."""
    # Print shapes before stacking for debugging
    print(f"Before stacking - Observations: {observations[0].shape}, Actions: {actions[0].shape}")
    print(f"Before stacking - Log probs: {log_probs_old[0].shape}, Rewards: {rewards[0].shape}")
    print(f"Before stacking - Values: {values[0].shape}, Dones: {type(dones[0])}")

    # Stack all tensors
    observations = torch.cat(observations, dim=0)  # [batch_size, n_agents, obs_dim]
    actions = torch.cat(actions, dim=0)  # [batch_size, n_agents, action_dim]
    log_probs_old = torch.cat(log_probs_old, dim=0)  # [batch_size, n_agents]
    rewards = torch.cat(rewards, dim=0)  # [batch_size, n_agents, 1]
    values = torch.cat(values, dim=0)  # [batch_size, n_agents, 1]

    # Print shapes after stacking for debugging
    print(f"After stacking - Observations: {observations.shape}, Actions: {actions.shape}")
    print(f"After stacking - Log probs: {log_probs_old.shape}, Rewards: {rewards.shape}")
    print(f"After stacking - Values: {values.shape}")

    # Handle done flags - convert to tensor and expand to match reward shape
    # Convert list of boolean scalars to tensor
    dones_tensor = torch.tensor(dones, dtype=torch.bool, device=device)  # [batch_size]
    # Get shapes
    batch_size = observations.shape[0]
    n_agents = observations.shape[1]
    # Create a mask matrix [batch_size, n_agents, 1] from scalar done flags
    dones = dones_tensor.unsqueeze(1).unsqueeze(2).expand(batch_size, n_agents, 1)
    print(f"Dones shape after processing: {dones.shape}")

    # Compute returns and advantages
    returns, advantages = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        gamma=gamma,
        gae_lambda=gae_lambda
    )

    # Get dataset size
    dataset_size = observations.shape[0]

    # Track losses for reporting
    total_policy_loss = 0
    total_value_loss = 0
    total_entropy = 0
    update_count = 0

    # PPO update for several epochs
    for epoch in range(ppo_epochs):
        # Create random indices for minibatches
        indices = torch.randperm(dataset_size).to(device)

        # Process in minibatches
        for start_idx in range(0, dataset_size, batch_size):
            # Get minibatch indices
            end_idx = min(start_idx + batch_size, dataset_size)
            minibatch_indices = indices[start_idx:end_idx]

            # Create minibatch
            mb_observations = observations[minibatch_indices]
            mb_actions = actions[minibatch_indices]
            mb_log_probs_old = log_probs_old[minibatch_indices]
            mb_returns = returns[minibatch_indices]
            mb_advantages = advantages[minibatch_indices]

            # Create TensorDict for policy and value network
            mb_states = TensorDict({
                "agents": {
                    "observation": mb_observations,
                    "action": mb_actions
                }
            }, batch_size=[mb_observations.shape[0]])

            # Forward pass through policy - first run with grad to get action distribution
            mb_states_for_policy = mb_states.clone()
            policy_output = policy(mb_states_for_policy)

            # Extract all needed values - ensure log_probs_new has the right shape
            loc = policy_output["agents", "loc"]
            scale = policy_output["agents", "scale"]
            log_probs_new = policy_output["agents", "action_log_prob"]

            # Make sure the shapes match for the PPO ratio calculation
            # The log_probs should be [batch_size, n_agents] for both tensors
            if log_probs_new.dim() > 2:
                log_probs_new = log_probs_new.squeeze(-1)
            if mb_log_probs_old.dim() > 2:
                mb_log_probs_old = mb_log_probs_old.squeeze(-1)

            # Calculate entropy approximation
            entropy = torch.log(scale).sum(-1).mean()

            # Forward pass through value network
            value_output = value_network(mb_states)
            values_new = value_output["agents", "value"]

            # Compute policy loss with clipping
            ratio = torch.exp(log_probs_new - mb_log_probs_old)

            # Ensure advantages have the right shape for elementwise multiplication with ratio
            if mb_advantages.dim() == 3 and ratio.dim() == 2:
                ratio = ratio.unsqueeze(-1)  # Make ratio [batch_size, n_agents, 1]
            elif mb_advantages.dim() == 2 and ratio.dim() == 2:
                # Both are [batch_size, n_agents] - fine as is
                pass
            else:
                # Print shapes for debugging
                print(f"Ratio shape: {ratio.shape}, Advantages shape: {mb_advantages.shape}")
                # Try to make them compatible
                mb_advantages = mb_advantages.reshape(*ratio.shape)

            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * mb_advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # Add entropy bonus for exploration
            policy_loss = policy_loss - entropy_coef * entropy

            # Compute value loss
            value_loss = value_coef * ((values_new - mb_returns) ** 2).mean()

            # Sum losses for combined update
            total_loss = policy_loss + value_loss

            # Update networks
            policy_optimizer.zero_grad()
            value_optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=0.5)
            torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=0.5)

            policy_optimizer.step()
            value_optimizer.step()

            # Track statistics
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.item()
            update_count += 1

    # Report average losses
    if update_count > 0:
        avg_policy_loss = total_policy_loss / update_count
        avg_value_loss = total_value_loss / update_count
        avg_entropy = total_entropy / update_count
        print(
            f"PPO Update - Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Entropy: {avg_entropy:.4f}")

    return {
        "policy_loss": avg_policy_loss,
        "value_loss": avg_value_loss,
        "entropy": avg_entropy
    }


def compute_gae(rewards, values, dones, gamma, gae_lambda):
    """Compute Generalized Advantage Estimation (GAE) and returns."""
    T = rewards.shape[0]
    n_agents = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)

    # Initialize with zeroes for terminal steps
    last_gae = torch.zeros(n_agents, 1, device=rewards.device)
    last_value = torch.zeros(n_agents, 1, device=rewards.device)

    # Compute GAE in reverse order
    for t in reversed(range(T)):
        # If done, reset values to zero (terminal state)
        # Create a mask from done flags (1.0 for not done, 0.0 for done)
        mask = 1.0 - dones[t].float()

        # For non-terminal states, compute next value estimate
        if t < T - 1:
            next_value = values[t + 1]
        else:
            next_value = last_value

        # Compute TD error: r + γV(s') - V(s)
        delta = rewards[t] + gamma * next_value * mask - values[t]

        # Compute GAE: A_t = δ_t + γλA_t+1
        last_gae = delta + gamma * gae_lambda * last_gae * mask
        advantages[t] = last_gae

    # Compute returns: R_t = A_t + V(s_t)
    returns = advantages + values

    # Normalize advantages per batch, but not across agents
    # This improves training stability without mixing advantage scales between agents
    batch_mean = advantages.mean(dim=0, keepdim=True)
    batch_std = advantages.std(dim=0, keepdim=True) + 1e-8
    advantages = (advantages - batch_mean) / batch_std

    return returns, advantages


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
            steps = 2000
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