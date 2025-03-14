import math
import numpy as np
from config import MAX_SPEED, SENSOR_COUNT, MAX_SENSOR_DISTANCE


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
