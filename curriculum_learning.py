"""
Curriculum Learning for FlightRL
Progressive difficulty training
"""
import numpy as np
from envs.flight_env import FlightEnv
from envs.flight_env_enhanced import EnhancedFlightEnv


class CurriculumSchedule:
    """Manages curriculum progression during training"""
    
    def __init__(self, schedule_type='linear', total_steps=100000):
        self.schedule_type = schedule_type
        self.total_steps = total_steps
        self.current_step = 0
        
    def get_difficulty(self):
        """Returns current difficulty level [0, 1]"""
        progress = min(self.current_step / self.total_steps, 1.0)
        
        if self.schedule_type == 'linear':
            return progress
        elif self.schedule_type == 'exponential':
            return 1 - np.exp(-3 * progress)
        elif self.schedule_type == 'step':
            # Step-wise progression
            if progress < 0.33:
                return 0.33
            elif progress < 0.66:
                return 0.66
            else:
                return 1.0
        else:
            return progress
    
    def update(self, steps=1):
        """Update curriculum step counter"""
        self.current_step += steps


class CurriculumFlightEnv:
    """
    Wraps FlightEnv with curriculum learning
    Adjusts initial conditions and constraints based on difficulty
    """
    
    def __init__(self, base_env_class=FlightEnv, use_enhanced=False):
        if use_enhanced:
            self.base_env = EnhancedFlightEnv(enable_wind=False)
        else:
            self.base_env = base_env_class()
        
        self.curriculum = CurriculumSchedule()
        self.min_difficulty = 0.2
        self.max_difficulty = 1.0
        
    def reset(self, seed=None, options=None):
        """Reset environment with curriculum-adjusted difficulty"""
        difficulty = self.min_difficulty + (self.max_difficulty - self.min_difficulty) * self.curriculum.get_difficulty()
        
        # Adjust initial conditions based on difficulty
        # Easy: closer to runway, better initial speed/altitude
        # Hard: farther, worse conditions
        
        obs, info = self.base_env.reset(seed=seed, options=options)
        
        # Modify initial conditions
        easy_factor = 1.0 - difficulty  # 0 (hard) to 1 (easy)
        
        # Closer initial distance (easier)
        self.base_env.distance = 5000 + (8000 - 5000) * difficulty
        
        # Better initial altitude (easier)
        self.base_env.altitude = 2000 + (3000 - 2000) * difficulty
        
        # Better initial speed (easier)
        self.base_env.speed = 150 + (200 - 150) * difficulty
        
        # Runway condition: easy mode uses only dry/wet, hard uses all
        if difficulty < 0.5:
            # Easy: mostly dry runways
            if np.random.random() < 0.8:
                self.base_env.runway_condition = 0.0  # dry
            else:
                self.base_env.runway_condition = 0.5  # wet
        else:
            # Hard: random across all conditions
            self.base_env.runway_condition = np.random.choice([0.0, 0.5, 1.0])
        
        obs = self.base_env._get_obs()
        return obs, info
    
    def step(self, action):
        """Standard step, but adjust landing criteria based on difficulty"""
        difficulty = self.curriculum.get_difficulty()
        
        obs, reward, terminated, truncated, info = self.base_env.step(action)
        
        # Adjust landing success criteria based on difficulty
        if self.base_env.distance <= 0:
            # Easier landing criteria for low difficulty
            if difficulty < 0.5:
                # More lenient: larger altitude/speed/angle bounds
                altitude_ok = 0 <= self.base_env.altitude <= 100  # was 50
                speed_ok = 80 <= self.base_env.speed <= 220  # was 100-200
                angle_ok = abs(self.base_env.angle) < 15  # was 10
            else:
                # Standard criteria
                altitude_ok = 0 <= self.base_env.altitude <= 50
                speed_ok = 100 <= self.base_env.speed <= 200
                angle_ok = abs(self.base_env.angle) < 10
            
            # Update success in info
            if altitude_ok and speed_ok and angle_ok:
                info['success'] = True
                if 'reward' in info:
                    reward += 100.0
            else:
                info['success'] = False
                if 'reward' in info:
                    reward -= 100.0
        
        return obs, reward, terminated, truncated, info
    
    def update_curriculum(self, steps=1):
        """Update curriculum progression"""
        self.curriculum.update(steps)
    
    def get_difficulty(self):
        """Get current difficulty level"""
        return self.curriculum.get_difficulty()
    
    # Delegate other attributes to base_env
    def __getattr__(self, name):
        return getattr(self.base_env, name)


def create_curriculum_env(use_enhanced=False):
    """Factory function to create curriculum learning environment"""
    return CurriculumFlightEnv(use_enhanced=use_enhanced)

