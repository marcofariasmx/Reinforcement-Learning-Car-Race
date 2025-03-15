import math
import numpy as np
import random
from config import MAX_SPEED, SENSOR_COUNT, MAX_SENSOR_DISTANCE


# Car environment with optimized calculations
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
        self.last_progress = 0
        self.prev_checkpoint_time = 0

        # Precompute for optimizations
        self.half_width = self.width / 2
        self.half_height = self.height / 2
        self.corners_template = [
            (-self.half_width, -self.half_height),  # Front left
            (self.half_width, -self.half_height),  # Front right
            (self.half_width, self.half_height),  # Rear right
            (-self.half_width, self.half_height),  # Rear left
        ]

        # Cache values for optimization
        self.cos_angle = math.cos(self.angle)
        self.sin_angle = math.sin(self.angle)
        self.corners = self.calculate_corners()

        # Precomputed sensor angles for more efficient updates
        self.sensor_angles = [
            0,  # Front
            math.pi / 4,  # Front-right
            math.pi / 2,  # Right
            3 * math.pi / 4,  # Rear-right
            math.pi,  # Rear
            5 * math.pi / 4,  # Rear-left
            3 * math.pi / 2,  # Left
            7 * math.pi / 4  # Front-left
        ]

    def calculate_corners(self):
        """Calculate car corners based on current position and angle (cached)"""
        corners = []
        for corner_x, corner_y in self.corners_template:
            # Rotate
            x_rotated = corner_x * self.cos_angle - corner_y * self.sin_angle
            y_rotated = corner_x * self.sin_angle + corner_y * self.cos_angle
            # Translate
            x_final = self.x + x_rotated
            y_final = self.y + y_rotated
            corners.append((x_final, y_final))
        return corners

    def get_corners(self):
        """Get cached corners if angle hasn't changed, otherwise recalculate"""
        return self.corners

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
        self.last_progress = 0
        self.prev_checkpoint_time = 0

        # Reset cached values
        self.cos_angle = math.cos(self.angle)
        self.sin_angle = math.sin(self.angle)
        self.corners = self.calculate_corners()

    def update(self, walls):
        # Update angle based on steering
        old_angle = self.angle
        self.angle += self.steering * (self.speed / self.max_speed) * 0.1
        self.angle %= 2 * math.pi  # Keep angle between 0 and 2π

        # Only recalculate trig functions if angle changed significantly
        if abs(self.angle - old_angle) > 0.001:
            self.cos_angle = math.cos(self.angle)
            self.sin_angle = math.sin(self.angle)

        # Update speed based on acceleration
        self.speed += self.acceleration * 0.1

        # Apply friction
        self.speed *= 0.95

        # Clamp speed
        self.speed = max(-self.max_speed / 2, min(self.speed, self.max_speed))  # Allow reverse but at lower max speed

        # Store previous position for movement calculation
        self.prev_position = (self.x, self.y)

        # Update position
        self.x += self.cos_angle * self.speed
        self.y += self.sin_angle * self.speed

        # Calculate traveled distance
        dx = self.x - self.prev_position[0]
        dy = self.y - self.prev_position[1]
        distance_moved = math.sqrt(dx * dx + dy * dy)

        # Only add to total distance if moving forward
        if self.speed > 0:
            self.total_distance += distance_moved

        # Increment time alive
        self.time_alive += 1

        # Recalculate corners after position/angle update
        self.corners = self.calculate_corners()

        # Check for collision
        self.crashed = self.check_collision(walls)

        # Update sensors
        self.update_sensors(walls)

        return not self.crashed

    def check_collision(self, walls):
        """Optimized collision detection using cached corners"""
        # Get precomputed corners
        car_corners = self.corners

        # Fast check - see if any point is near the car before detailed collision test
        car_center = (self.x, self.y)
        car_radius = max(self.width, self.height)  # Simple bounding circle

        # Check each car edge against each wall
        for i in range(4):
            car_p1 = car_corners[i]
            car_p2 = car_corners[(i + 1) % 4]

            for wall in walls:
                wall_p1 = (wall[0], wall[1])
                wall_p2 = (wall[2], wall[3])

                # Fast bounding box check
                if self.line_intersection(car_p1, car_p2, wall_p1, wall_p2):
                    return True

        return False

    def update_sensors(self, walls):
        """Update sensor readings with optimized calculations"""
        # Since sensor angles are fixed relative to car direction,
        # we can use the precomputed angles
        for i, rel_angle in enumerate(self.sensor_angles):
            # Calculate absolute angle
            angle = (self.angle + rel_angle) % (2 * math.pi)

            # Precompute sin and cos for this angle
            angle_cos = math.cos(angle)
            angle_sin = math.sin(angle)

            # Calculate sensor end point (at maximum distance)
            end_x = self.x + angle_cos * MAX_SENSOR_DISTANCE
            end_y = self.y + angle_sin * MAX_SENSOR_DISTANCE

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
        """Check if car has reached the next checkpoint with distance squared for performance"""
        # Check if car has reached the next checkpoint
        next_checkpoint = checkpoints[self.checkpoint_idx]
        dx = self.x - next_checkpoint[0]
        dy = self.y - next_checkpoint[1]

        # Use distance squared to avoid square root calculation
        distance_squared = dx * dx + dy * dy
        checkpoint_radius_squared = 50 * 50  # Checkpoint radius squared

        if distance_squared < checkpoint_radius_squared:
            self.checkpoint_idx = (self.checkpoint_idx + 1) % len(checkpoints)

            # If we've completed a full lap
            if self.checkpoint_idx == 0:
                self.laps_completed += 1
                return True

            return True  # Reached a checkpoint

        return False  # No checkpoint reached

    def get_state(self):
        """Get the current state vector with optimized normalization"""
        # Normalize sensor readings (using numpy for vectorized operations)
        normalized_sensors = np.array(self.sensor_readings) / MAX_SENSOR_DISTANCE

        # Normalized speed
        normalized_speed = self.speed / self.max_speed

        # Use cached sin and cos angle values
        # Append additional state information
        state = np.append(normalized_sensors, [normalized_speed, self.sin_angle, self.cos_angle])

        return state.astype(np.float32)

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


# Precomputed track data for better performance
class TrackCache:
    def __init__(self):
        self.walls = None
        self.checkpoints = None
        self.outer_track = None
        self.inner_track = None
        self.center_x = None
        self.center_y = None

    def create_map(self):
        """Cache and return the track data"""
        if self.walls is not None:
            return self.walls, self.checkpoints

        # Create a simple oval track - much more consistent and clean
        # Track parameters
        self.center_x, self.center_y = 600, 400  # Center of the screen
        outer_width, outer_height = 700, 500  # Oval dimensions for outer track
        inner_width, inner_height = 400, 250  # Oval dimensions for inner track
        segments = 16  # Number of segments to approximate the oval (higher = smoother)

        self.outer_track = []
        self.inner_track = []
        self.checkpoints = []

        # Precompute sin and cos for each angle
        angle_cos = []
        angle_sin = []
        for i in range(segments):
            angle = 2 * math.pi * i / segments
            angle_cos.append(math.cos(angle))
            angle_sin.append(math.sin(angle))

        # Use precomputed values for efficiency
        for i in range(segments):
            angle1 = 2 * math.pi * i / segments
            angle2 = 2 * math.pi * (i + 1) / segments

            # Use precomputed trig functions when possible
            cos1 = angle_cos[i]
            sin1 = angle_sin[i]
            cos2 = angle_cos[(i + 1) % segments]
            sin2 = angle_sin[(i + 1) % segments]

            # Outer track points
            x1_outer = self.center_x + outer_width / 2 * cos1
            y1_outer = self.center_y + outer_height / 2 * sin1
            x2_outer = self.center_x + outer_width / 2 * cos2
            y2_outer = self.center_y + outer_height / 2 * sin2

            # Inner track points
            x1_inner = self.center_x + inner_width / 2 * cos1
            y1_inner = self.center_y + inner_height / 2 * sin1
            x2_inner = self.center_x + inner_width / 2 * cos2
            y2_inner = self.center_y + inner_height / 2 * sin2

            # Add line segments
            self.outer_track.append((x1_outer, y1_outer, x2_outer, y2_outer))
            self.inner_track.append((x1_inner, y1_inner, x2_inner, y2_inner))

            # Add checkpoint at every segment (more checkpoints for visibility)
            checkpoint_x = self.center_x + (outer_width / 2 + inner_width / 2) / 2 * cos1
            checkpoint_y = self.center_y + (outer_height / 2 + inner_height / 2) / 2 * sin1
            self.checkpoints.append((checkpoint_x, checkpoint_y))

        # All walls
        self.walls = self.outer_track + self.inner_track
        return self.walls, self.checkpoints


# Singleton track cache
track_cache = TrackCache()


# Map creation - walls are defined as line segments (start_x, start_y, end_x, end_y)
def create_map():
    """Create the track map using cached data for better performance"""
    return track_cache.create_map()


# Environment class
class CarEnv:
    def __init__(self):
        # Get cached track data for better performance
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

        # Precompute safe starting position for reset
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = math.pi  # Set car pointing tangent to the track (to the left)

    def reset(self):
        """Reset environment to initial state"""
        # Make sure car starts at the same safe position using precomputed values
        self.car.reset(self.start_x, self.start_y, self.start_angle)

        self.done = False
        self.steps = 0
        self.reward = 0
        self.total_reward = 0
        self.prev_checkpoint_time = 0
        self.last_progress = 0
        return self.car.get_state()

    def step(self, action):
        """Take a step in the environment with optimized reward calculation"""
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

        # Calculate car's forward direction vector - use cached values
        forward_x = self.car.cos_angle
        forward_y = self.car.sin_angle

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


# New class for multi-car environment to be added to environment.py
class MultiCarEnv:
    """Environment that manages multiple cars training in parallel"""

    def __init__(self, num_cars=4, min_starting_distance=100):
        # Create a shared track for all cars
        self.walls, self.checkpoints = create_map()

        # Determine a safe starting position (same as in the original CarEnv)
        center_x, center_y = 600, 400
        start_x = center_x + 550 / 2  # A bit more than halfway from center to outer wall on the right
        start_y = center_y  # At the horizontal middle
        start_angle = math.pi  # Set car pointing tangent to the track (to the left)

        # Save starting position for resets
        self.start_x = start_x
        self.start_y = start_y
        self.start_angle = start_angle

        # Initialize cars at exactly the same position
        self.cars = []
        self.done_cars = []  # Track which cars have finished their episodes
        self.rewards = []
        self.total_rewards = []
        self.steps = 0
        self.max_steps = 2000
        self.num_cars = num_cars

        # Create all cars at the exact same position
        for i in range(num_cars):
            # Create car at starting position (all in exactly the same position)
            car = Car(start_x, start_y)
            car.angle = start_angle  # Set car heading tangent to the track
            self.cars.append(car)

            # Initialize tracking data for this car
            self.rewards.append(0)
            self.total_rewards.append(0)
            self.done_cars.append(False)

        # Designate the first car as the "main" car for detailed tracking
        self.main_car_idx = 0

        # For compatibility with original environment
        self.state_dim = 8 + 3  # Sensors + speed + sin(angle) + cos(angle)
        self.action_dim = 2  # Acceleration and steering

    def reset(self):
        """Reset all cars or just the ones that have finished their episodes"""
        # Reset global environment state
        self.steps = 0

        # If all cars are done, reset them all to exactly the same position
        if all(self.done_cars):
            for i in range(self.num_cars):
                # Reset car at the exact same starting position
                self.cars[i].reset(self.start_x, self.start_y, self.start_angle)

                # Reset tracking data
                self.rewards[i] = 0
                self.total_rewards[i] = 0
                self.done_cars[i] = False
        else:
            # Only reset cars that are done
            for i in range(self.num_cars):
                if self.done_cars[i]:
                    # Reset the car to the exact same starting position
                    self.cars[i].reset(self.start_x, self.start_y, self.start_angle)
                    self.rewards[i] = 0
                    self.total_rewards[i] = 0
                    self.done_cars[i] = False

        # Return states for all cars
        return [car.get_state() for car in self.cars]

    def step(self, actions):
        """Take a step for all cars based on their individual actions"""
        self.steps += 1

        next_states = []
        rewards = []
        dones = []
        infos = []

        # Update each car based on its action
        for i, car in enumerate(self.cars):
            # Skip cars that are already done
            if self.done_cars[i]:
                # Return placeholder values for done cars
                next_states.append(car.get_state())
                rewards.append(0)
                dones.append(True)
                infos.append({'laps': car.laps_completed, 'checkpoints': car.checkpoint_idx,
                              'distance': car.total_distance, 'speed': car.speed, 'steps': self.steps})
                continue

            # Apply action to this car
            car.acceleration = actions[i][0] * 2  # Scale action
            car.steering = actions[i][1]

            # Update car
            alive = car.update(self.walls)

            # Check for checkpoint progress
            checkpoint_reached = car.check_checkpoint(self.checkpoints)

            # Calculate reward similar to the original environment
            # Get the next checkpoint and calculate direction to it
            next_checkpoint_idx = car.checkpoint_idx
            next_checkpoint = self.checkpoints[next_checkpoint_idx]

            # Calculate vector from car to next checkpoint
            dx = next_checkpoint[0] - car.x
            dy = next_checkpoint[1] - car.y
            distance_to_checkpoint = math.sqrt(dx * dx + dy * dy)

            # Calculate car's forward direction vector - use cached values
            forward_x = car.cos_angle
            forward_y = car.sin_angle

            # Calculate dot product to see if car is pointing toward checkpoint
            if distance_to_checkpoint > 0:
                checkpoint_dx = dx / distance_to_checkpoint
                checkpoint_dy = dy / distance_to_checkpoint
                direction_alignment = forward_x * checkpoint_dx + forward_y * checkpoint_dy
            else:
                direction_alignment = 1  # Prevent division by zero

            # Calculate reward
            reward = 0.0
            done = False

            if not alive:
                # Penalize crashing, scale based on progress
                progress_factor = min(1.0, car.total_distance / 500)
                crash_penalty = -5.0 * (1.0 - progress_factor)
                reward += crash_penalty
                done = True
            elif self.steps >= self.max_steps:
                # Timeout
                reward += -2.0
                done = True
            elif checkpoint_reached:
                # Reward checkpoint progress
                checkpoint_reward = 20.0
                reward += checkpoint_reward

                # Add time bonus
                time_to_checkpoint = self.steps - car.prev_checkpoint_time if hasattr(car,
                                                                                      'prev_checkpoint_time') else self.steps
                car.prev_checkpoint_time = self.steps

                speed_bonus = 10.0 / max(10, time_to_checkpoint) * 10
                reward += speed_bonus

                # Extra reward for lap completion
                if car.checkpoint_idx == 0:
                    reward += 100.0
            else:
                # Reward for progress toward checkpoint
                progress = car.total_distance
                progress_diff = progress - (car.last_progress if hasattr(car, 'last_progress') else 0)
                car.last_progress = progress

                if progress_diff > 0:
                    reward += progress_diff * 0.5 * max(0, direction_alignment)

                if progress_diff < 0.01:
                    reward -= 0.05

                reward += direction_alignment * 0.05

                # Reward smooth driving
                smoothness = 1.0 - min(1.0, abs(car.steering) * 2)
                reward += smoothness * 0.05

                # Reward for approaching checkpoint
                reward += 0.5 / max(10, distance_to_checkpoint)

            # Store reward
            self.rewards[i] = reward
            self.total_rewards[i] = self.total_rewards[i] + reward

            # Mark car as done if crashed or timeout
            self.done_cars[i] = done

            # Get next state
            next_state = car.get_state()

            # Return step information
            next_states.append(next_state)
            rewards.append(reward)
            dones.append(done)

            info = {
                'laps': car.laps_completed,
                'checkpoints': car.checkpoint_idx,
                'distance': car.total_distance,
                'speed': car.speed,
                'steps': self.steps,
                'direction_alignment': direction_alignment
            }
            infos.append(info)

        # Check if all cars are done
        all_done = all(self.done_cars)

        return next_states, rewards, dones, infos, all_done

    def get_main_car(self):
        """Return the main car for visualization focus"""
        return self.cars[self.main_car_idx]

    def get_main_car_data(self):
        """Get data for the main car for visualization"""
        return {
            'car': self.cars[self.main_car_idx],
            'reward': self.rewards[self.main_car_idx],
            'total_reward': self.total_rewards[self.main_car_idx]
        }

    def get_best_car_index(self):
        """Return the index of the car with the highest total reward"""
        if all(reward == 0 for reward in self.total_rewards):
            return self.main_car_idx
        return self.total_rewards.index(max(self.total_rewards))

    def set_main_car_to_best(self):
        """Set the main car to the one with the highest reward"""
        self.main_car_idx = self.get_best_car_index()
        return self.main_car_idx