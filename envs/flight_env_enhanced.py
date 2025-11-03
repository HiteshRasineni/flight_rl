"""
Enhanced Flight Landing Environment with wind effects and improved physics
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class EnhancedFlightEnv(gym.Env):
    """
    Enhanced Flight Landing Environment for RL with wind effects
    
    State:
      - altitude (0-5000 m)
      - speed (0-300 knots)
      - distance to runway (0-10000 m)
      - pitch angle (-30 to +30 degrees)
      - runway condition (0=dry,0.5=wet,1=icy)
      - wind speed (0-50 knots)
      - wind direction (-180 to +180 degrees, 0=headwind)
      - vertical velocity (m/s)
    """

    def __init__(self, enable_wind=True, wind_severity=1.0):
        super(EnhancedFlightEnv, self).__init__()

        self.enable_wind = enable_wind
        self.wind_severity = wind_severity

        # Observation space (extended)
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -30, 0, 0, -180, -50], dtype=np.float32),
            high=np.array([5000, 300, 10000, 30, 1, 50, 180, 50], dtype=np.float32),
            dtype=np.float32
        )

        # Action space
        self.action_space = spaces.Discrete(5)

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.altitude = 3000.0     # meters
        self.speed = 200.0         # knots
        self.distance = 8000.0     # meters to runway
        self.angle = 0.0           # pitch angle in degrees
        self.vertical_velocity = 0.0  # m/s (negative = descending)
        self.runway_condition = np.random.choice([0.0, 0.5, 1.0])  # dry/wet/icy
        
        # Wind initialization
        if self.enable_wind:
            self.wind_speed = np.random.uniform(0, 30 * self.wind_severity)  # knots
            self.wind_direction = np.random.uniform(-180, 180)  # degrees
            self.wind_gust = 0.0  # temporary gust component
        else:
            self.wind_speed = 0.0
            self.wind_direction = 0.0
            self.wind_gust = 0.0
        
        self.steps = 0
        self.wind_change_counter = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # --- Wind effects ---
        if self.enable_wind:
            # Periodic wind gusts
            if self.steps % 50 == 0 and np.random.random() < 0.3:
                self.wind_gust = np.random.uniform(-10, 10) * self.wind_severity
            else:
                # Decay gust over time
                self.wind_gust *= 0.95
            
            # Wind can change over time
            if self.steps % 100 == 0 and np.random.random() < 0.2:
                self.wind_direction += np.random.uniform(-30, 30)
                self.wind_direction = np.clip(self.wind_direction, -180, 180)
            
            # Effective wind speed (wind_speed + gust)
            effective_wind = self.wind_speed + self.wind_gust
            # Headwind/tailwind component (affects speed)
            wind_angle_rad = np.radians(self.wind_direction)
            headwind_component = effective_wind * np.cos(wind_angle_rad)
            crosswind_component = effective_wind * np.sin(wind_angle_rad)
            
            # Headwind increases effective speed, tailwind decreases
            wind_speed_effect = headwind_component * 0.3  # reduced effect for stability
        else:
            wind_speed_effect = 0.0
            crosswind_component = 0.0

        # --- Action effects ---
        if action == 0:   # throttle+
            self.speed += 5 - wind_speed_effect * 0.1
        elif action == 1: # throttle-
            self.speed -= 5 + wind_speed_effect * 0.1
        elif action == 2: # pitch+
            self.altitude += 50
            self.angle += 2
            self.vertical_velocity += 1.0  # climbing
        elif action == 3: # pitch-
            self.altitude -= 50
            self.angle -= 2
            self.vertical_velocity -= 1.0  # descending

        # --- Enhanced physics ---
        # Distance change depends on horizontal speed component
        horizontal_speed = self.speed * np.cos(np.radians(self.angle))
        self.distance -= max(horizontal_speed * 0.51, 1)  # approach runway
        
        # Gravity effect (more realistic)
        gravity_effect = 9.8 * 0.5  # m/s^2 converted to step effect
        self.vertical_velocity -= gravity_effect
        self.altitude += self.vertical_velocity * 0.1  # integrate vertical velocity
        
        # Drag (increases with speed and angle)
        drag_coefficient = 0.5 + abs(self.angle) * 0.01
        self.speed -= max(self.speed * drag_coefficient * 0.01, 0.5)
        
        # Crosswind effect on angle (sideways drift)
        if self.enable_wind and abs(crosswind_component) > 5:
            angle_drift = crosswind_component * 0.05
            self.angle += angle_drift

        # Clamp values
        self.altitude = max(self.altitude, 0)
        self.speed = max(self.speed, 0)
        self.speed = min(self.speed, 300)  # max speed
        self.angle = np.clip(self.angle, -30, 30)
        self.vertical_velocity = np.clip(self.vertical_velocity, -50, 50)

        # --- Enhanced reward shaping ---
        reward = -0.5  # small step penalty
        reward += -0.02 * abs(self.angle)  # penalty for high pitch
        reward += 0.01 * (8000 - self.distance) / 100  # reward for approaching runway
        reward += -0.001 * max(self.altitude - 500, 0)  # penalty for too high altitude
        
        # Smooth descent reward
        if 0 < self.altitude < 1000 and self.vertical_velocity < -5:
            reward += 0.1  # reward smooth descent
        
        # Speed control reward (optimal approach speed)
        optimal_speed = 150
        speed_diff = abs(self.speed - optimal_speed)
        reward += -0.01 * speed_diff / 10  # reward maintaining optimal speed

        done = False
        success = False

        # --- Landing success (enhanced criteria) ---
        if self.distance <= 0:
            # Stricter landing criteria
            altitude_ok = 0 <= self.altitude <= 50
            speed_ok = 100 <= self.speed <= 200
            angle_ok = abs(self.angle) < 10
            vertical_velocity_ok = abs(self.vertical_velocity) < 5  # smooth landing
            
            if altitude_ok and speed_ok and angle_ok and vertical_velocity_ok:
                reward += 100.0
                success = True
            else:
                reward -= 100.0
            done = True

        # --- Crash conditions ---
        if self.altitude <= 0 and self.distance > 0:
            reward -= 100.0
            done = True

        if self.speed <= 30 and self.altitude > 100:
            reward -= 100.0
            done = True

        # Vertical velocity crash
        if abs(self.vertical_velocity) > 30 and self.altitude < 100:
            reward -= 100.0
            done = True

        if self.steps >= 500:  # max steps
            done = True

        obs = self._get_obs()
        info = {
            "success": success,
            "wind_speed": self.wind_speed + self.wind_gust if self.enable_wind else 0.0,
            "wind_direction": self.wind_direction if self.enable_wind else 0.0,
            "vertical_velocity": self.vertical_velocity
        }
        return obs, reward, done, False, info

    def _get_obs(self):
        # Normalize observations into [0,1] or appropriate ranges
        obs = np.array([
            self.altitude / 5000,
            self.speed / 300,
            self.distance / 10000,
            (self.angle + 30) / 60,
            self.runway_condition,
            (self.wind_speed + self.wind_gust) / 50 if self.enable_wind else 0.0,
            (self.wind_direction + 180) / 360 if self.enable_wind else 0.0,
            (self.vertical_velocity + 50) / 100
        ], dtype=np.float32)
        return obs

