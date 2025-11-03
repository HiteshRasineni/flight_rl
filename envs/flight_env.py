import numpy as np
import gymnasium as gym
from gymnasium import spaces

class FlightEnv(gym.Env):
    """
    Simplified Flight Landing Environment for RL

    State:
      - altitude (0-5000 m)
      - speed (0-300 knots)
      - distance to runway (0-10000 m)
      - pitch angle (-30 to +30 degrees)
      - runway condition (0=dry,0.5=wet,1=icy)

    Actions:
      0 = throttle up
      1 = throttle down
      2 = pitch up
      3 = pitch down
      4 = do nothing
    """

    def __init__(self):
        super(FlightEnv, self).__init__()

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([0, 0, 0, -30, 0], dtype=np.float32),
            high=np.array([5000, 300, 10000, 30, 1], dtype=np.float32),
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
        self.runway_condition = np.random.choice([0.0, 0.5, 1.0])  # dry/wet/icy
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # --- Action effects ---
        if action == 0:   # throttle+
            self.speed += 5
        elif action == 1: # throttle-
            self.speed -= 5
        elif action == 2: # pitch+
            self.altitude += 50
            self.angle += 2
        elif action == 3: # pitch-
            self.altitude -= 50
            self.angle -= 2

        # --- Natural drift ---
        self.distance -= max(self.speed * 0.5, 1)  # approach runway
        self.altitude -= 20                        # gravity
        self.speed -= 1                            # drag

        # Clamp values
        self.altitude = max(self.altitude, 0)
        self.speed = max(self.speed, 0)
        self.angle = np.clip(self.angle, -30, 30)

        # --- Reward shaping ---
        reward = -0.5  # small step penalty
        reward += -0.02 * abs(self.angle)  # penalty for high pitch
        reward += 0.01 * (8000 - self.distance)  # reward for approaching runway
        reward += -0.001 * max(self.altitude - 500, 0)  # penalty for too high altitude

        done = False
        success = False

        # --- Landing success ---
        if self.distance <= 0:
            if 0 <= self.altitude <= 50 and 100 <= self.speed <= 200 and abs(self.angle) < 10:
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

        if self.steps >= 500:  # max steps
            done = True

        obs = self._get_obs()
        return obs, reward, done, False, {"success": success}

    def _get_obs(self):
        # Normalize observations into [0,1]
        obs = np.array([
            self.altitude / 5000,
            self.speed / 300,
            self.distance / 10000,
            (self.angle + 30) / 60,
            self.runway_condition
        ], dtype=np.float32)
        return obs
