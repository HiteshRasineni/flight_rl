import pygame
import torch
import time
import math
import random
from envs.flight_env import FlightEnv
from agents.dqn_agent import DQNAgent

CHECKPOINT = "./checkpoints/dqn_final.pt"  # Change to your model
EPISODES = 3
FPS = 60  # Increased FPS for smoother animations

# Particle class for explosion effects
class Particle:
    def __init__(self, x, y, color):
        self.x = x
        self.y = y
        self.color = color
        self.vel_x = random.uniform(-5, 5)
        self.vel_y = random.uniform(-8, -2)
        self.lifetime = random.uniform(20, 40)
        self.size = random.uniform(2, 6)
        self.gravity = 0.3
        
    def update(self):
        self.x += self.vel_x
        self.y += self.vel_y
        self.vel_y += self.gravity
        self.lifetime -= 1
        self.size = max(0, self.size - 0.1)
        
    def draw(self, screen):
        if self.lifetime > 0 and self.size > 0:
            pygame.draw.circle(screen, self.color, (int(self.x), int(self.y)), int(self.size))

# Explosion effect
class Explosion:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.particles = []
        self.duration = 30
        
        # Create particles with different colors
        colors = [(255, 100, 0), (255, 200, 0), (255, 50, 0), (200, 200, 200), (100, 100, 100)]
        for _ in range(50):
            color = random.choice(colors)
            self.particles.append(Particle(x, y, color))
    
    def update(self):
        for particle in self.particles:
            particle.update()
        self.particles = [p for p in self.particles if p.lifetime > 0]
        self.duration -= 1
        
    def draw(self, screen):
        for particle in self.particles:
            particle.draw(screen)
    
    def is_done(self):
        return self.duration <= 0 or len(self.particles) == 0

# Smoke trail for the plane
class SmokeTrail:
    def __init__(self):
        self.particles = []
        
    def add_particle(self, x, y):
        self.particles.append({
            'x': x,
            'y': y,
            'life': 30,
            'size': random.uniform(3, 6),
            'alpha': 200
        })
    
    def update(self):
        for p in self.particles:
            p['life'] -= 1
            p['size'] = max(1, p['size'] - 0.15)
            p['alpha'] = max(0, p['alpha'] - 7)
        self.particles = [p for p in self.particles if p['life'] > 0]
    
    def draw(self, screen):
        for p in self.particles:
            if p['alpha'] > 0:
                color = (150, 150, 150, int(p['alpha']))
                surface = pygame.Surface((int(p['size'] * 2), int(p['size'] * 2)), pygame.SRCALPHA)
                pygame.draw.circle(surface, color, (int(p['size']), int(p['size'])), int(p['size']))
                screen.blit(surface, (int(p['x'] - p['size']), int(p['y'] - p['size'])))

def draw_gradient_background(screen, width, height):
    """Draw a gradient sky background"""
    for y in range(height - 50):  # Exclude runway area
        # Gradient from light blue to lighter blue
        ratio = y / (height - 50)
        r = int(135 + (100 - 135) * ratio)
        g = int(206 + (150 - 206) * ratio)
        b = int(235 + (200 - 235) * ratio)
        pygame.draw.line(screen, (r, g, b), (0, y), (width, y))

def draw_cloud(screen, x, y, size):
    """Draw a fluffy cloud"""
    cloud_color = (255, 255, 255)
    pygame.draw.circle(screen, cloud_color, (x, y), size)
    pygame.draw.circle(screen, cloud_color, (x + size, y), size)
    pygame.draw.circle(screen, cloud_color, (x - size, y), size)
    pygame.draw.circle(screen, cloud_color, (x, y - size), size)

def draw_bar(screen, x, y, w, h, pct, color=(0, 255, 0), bg_color=(50, 50, 50), outline_color=(200, 200, 200)):
    """Draw a vertical bar with better styling"""
    # Background
    pygame.draw.rect(screen, bg_color, (x, y, w, h))
    # Fill
    fill_h = int(h * max(0, min(1, pct)))
    if fill_h > 0:
        pygame.draw.rect(screen, color, (x, y + h - fill_h, w, fill_h))
    # Outline
    pygame.draw.rect(screen, outline_color, (x, y, w, h), 2)
    # Border highlight
    pygame.draw.line(screen, (255, 255, 255, 100), (x, y), (x + w, y), 1)

def draw_plane(screen, x, y, angle, plane_size=40):
    """Draw a better-looking plane"""
    # Create plane surface
    plane_surf = pygame.Surface((plane_size, plane_size), pygame.SRCALPHA)
    
    # Body (main rectangle)
    body_rect = pygame.Rect(plane_size//4, plane_size//2 - 5, plane_size//2, 10)
    pygame.draw.ellipse(plane_surf, (200, 200, 200), body_rect)
    pygame.draw.ellipse(plane_surf, (150, 150, 150), body_rect, 2)
    
    # Nose (triangle)
    nose_points = [
        (plane_size//2, plane_size//2 - 8),
        (plane_size//2 + 15, plane_size//2),
        (plane_size//2, plane_size//2 + 8)
    ]
    pygame.draw.polygon(plane_surf, (255, 100, 100), nose_points)
    
    # Wings
    wing_left = pygame.Rect(plane_size//4 + 5, plane_size//2 - 2, 8, 4)
    wing_right = pygame.Rect(plane_size//4 + 5, plane_size//2 + 2, 8, 4)
    pygame.draw.rect(plane_surf, (180, 180, 180), wing_left)
    pygame.draw.rect(plane_surf, (180, 180, 180), wing_right)
    
    # Tail
    tail_points = [
        (plane_size//4, plane_size//2 - 6),
        (plane_size//4 - 5, plane_size//2 - 8),
        (plane_surf.get_width()//4, plane_size//2)
    ]
    pygame.draw.polygon(plane_surf, (160, 160, 160), tail_points)
    
    # Rotate and blit
    rotated = pygame.transform.rotate(plane_surf, -angle)
    rect = rotated.get_rect(center=(x, y))
    screen.blit(rotated, rect)

def draw_runway(screen, width, height):
    """Draw a detailed runway with markings"""
    runway_y = height - 50
    runway_height = 50
    
    # Runway base
    pygame.draw.rect(screen, (60, 60, 60), (0, runway_y, width, runway_height))
    
    # Center line
    for i in range(0, width, 40):
        pygame.draw.rect(screen, (255, 255, 255), (i, runway_y + runway_height//2 - 2, 20, 4))
    
    # Side lines
    pygame.draw.line(screen, (255, 255, 0), (0, runway_y), (width, runway_y), 3)
    pygame.draw.line(screen, (255, 255, 0), (0, runway_y + runway_height), 
                    (width, runway_y + runway_height), 3)
    
    # Threshold markings (at edges)
    for x in [50, width - 50]:
        for i in range(5):
            pygame.draw.line(screen, (255, 255, 255), 
                           (x + i * 5, runway_y + 10),
                           (x + i * 5, runway_y + runway_height - 10), 2)

def draw_hud(screen, width, height, font, text_lines):
    """Draw HUD with semi-transparent background"""
    # HUD background panel
    hud_surface = pygame.Surface((250, len(text_lines) * 25 + 20), pygame.SRCALPHA)
    hud_surface.fill((0, 0, 0, 180))  # Semi-transparent black
    screen.blit(hud_surface, (10, 10))
    
    # Border
    pygame.draw.rect(screen, (0, 255, 0), (10, 10, 250, len(text_lines) * 25 + 20), 2)
    
    # Text with better styling
    for i, line in enumerate(text_lines):
        # Determine color based on content
        if "Success: True" in line:
            color = (0, 255, 0)
        elif "Success: False" in line or "Crash" in line:
            color = (255, 0, 0)
        elif "Runway" in line:
            if "0.0" in line:
                color = (255, 255, 255)  # Dry - white
            elif "0.5" in line:
                color = (0, 150, 255)  # Wet - blue
            else:
                color = (150, 200, 255)  # Icy - light blue
        else:
            color = (255, 255, 255)
            
        text = font.render(line, True, color)
        screen.blit(text, (20, 15 + i * 25))

def run_visualization():
    # Device and agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    env = FlightEnv()
    obs_dim = env.observation_space.shape[0]
    n_actions = env.action_space.n

    agent = DQNAgent(obs_dim, n_actions, device=device)
    agent.load(CHECKPOINT)

    # Pygame setup
    pygame.init()
    WIDTH, HEIGHT = 1200, 700
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Flight RL Visualization - Enhanced")
    clock = pygame.time.Clock()

    font = pygame.font.SysFont('Arial', 20, bold=True)
    title_font = pygame.font.SysFont('Arial', 16)

    # Cloud positions (static for now)
    clouds = [(200, 100, 30), (600, 150, 40), (1000, 80, 35), (400, 200, 25)]
    
    # Smoke trail
    smoke = SmokeTrail()
    
    # Explosion effect (will be created on crash)
    explosion = None

    for ep in range(1, EPISODES + 1):
        obs, _ = env.reset()
        done = False
        explosion = None  # Reset explosion
        
        # Track previous position for smoke trail
        prev_x, prev_y = None, None

        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    return

            # Agent action
            action = agent.act(obs, epsilon=0.0)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Map env distance/altitude to screen coordinates
            dist_norm = obs[2]  # normalized distance [0,1]
            alt_norm = obs[0]   # normalized altitude [0,1]
            angle_deg = (obs[3] * 60) - 30  # actual angle in degrees

            x = int((1 - dist_norm) * (WIDTH - 150))  # left to right
            y = int(HEIGHT - 50 - (alt_norm * (HEIGHT - 100)))  # ground at bottom

            # Draw background
            draw_gradient_background(screen, WIDTH, HEIGHT)
            
            # Draw clouds
            for cloud_x, cloud_y, cloud_size in clouds:
                draw_cloud(screen, cloud_x, cloud_y, cloud_size)
            
            # Draw runway
            draw_runway(screen, WIDTH, HEIGHT)

            # Update and draw smoke trail
            if prev_x is not None and prev_y is not None and alt_norm > 0.1:
                # Only show smoke when plane is moving
                smoke.add_particle(prev_x, prev_y)
            smoke.update()
            smoke.draw(screen)
            prev_x, prev_y = x, y

            # Draw plane
            if alt_norm > 0:  # Only draw if not crashed yet
                draw_plane(screen, x, y, angle_deg)
            
            # Handle crash/imperfect landing
            if done and not info.get('success', False):
                if explosion is None:
                    # Create explosion at crash location
                    explosion = Explosion(x, y)
                explosion.update()
                explosion.draw(screen)
            
            # Success indicator
            if done and info.get('success', False):
                # Draw success message
                success_text = title_font.render("✓ SAFE LANDING!", True, (0, 255, 0))
                text_rect = success_text.get_rect(center=(WIDTH//2, HEIGHT//2 - 50))
                # Background for text
                bg_rect = text_rect.inflate(20, 10)
                pygame.draw.rect(screen, (0, 0, 0, 200), bg_rect)
                screen.blit(success_text, text_rect)

            # Calculate throttle and pitch percentages
            throttle_pct = obs[1]  # Speed normalized
            angle_normalized = (obs[3] + 30) / 60  # angle normalized to [0,1]

            # Draw throttle and pitch bars (improved)
            bar_x = WIDTH - 120
            bar_y = 80
            bar_width = 25
            bar_height = 200
            
            # Throttle bar
            draw_bar(screen, bar_x, bar_y, bar_width, bar_height, throttle_pct, 
                    color=(0, 255, 0), bg_color=(40, 40, 40))
            throttle_label = title_font.render("THR", True, (255, 255, 255))
            screen.blit(throttle_label, (bar_x - 5, bar_y - 20))
            
            # Pitch bar
            draw_bar(screen, bar_x + 40, bar_y, bar_width, bar_height, angle_normalized, 
                    color=(0, 150, 255), bg_color=(40, 40, 40))
            pitch_label = title_font.render("PCH", True, (255, 255, 255))
            screen.blit(pitch_label, (bar_x + 35, bar_y - 20))

            # Determine runway condition text
            runway_cond = env.runway_condition
            if runway_cond == 0.0:
                runway_text = "Dry"
            elif runway_cond == 0.5:
                runway_text = "Wet"
            else:
                runway_text = "Icy"

            # Draw HUD
            text_lines = [
                f"Episode: {ep}/{EPISODES}",
                f"Success: {info.get('success', False)}",
                f"Runway: {runway_text}",
                f"Altitude: {int(obs[0]*5000)} m",
                f"Speed: {int(obs[1]*300)} kn",
                f"Distance: {int(obs[2]*10000)} m",
                f"Angle: {int(angle_deg):.1f}°",
                f"Reward: {reward:.2f}"
            ]
            draw_hud(screen, WIDTH, HEIGHT, font, text_lines)

            # Draw episode info at top
            episode_text = title_font.render(f"Episode {ep} of {EPISODES}", True, (255, 255, 255))
            screen.blit(episode_text, (WIDTH//2 - 80, 10))

            pygame.display.flip()
            clock.tick(FPS)
            
            # Keep explosion going for a bit after crash
            if done and explosion is not None:
                if not explosion.is_done():
                    explosion.update()
                    explosion.draw(screen)
                    pygame.display.flip()
                    clock.tick(FPS)
                else:
                    time.sleep(0.5)  # Brief pause after explosion
                    break
            elif done:
                time.sleep(1)  # Brief pause after successful landing

    pygame.quit()

if __name__ == "__main__":
    run_visualization()
