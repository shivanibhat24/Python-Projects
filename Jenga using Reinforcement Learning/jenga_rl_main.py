import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
import gymnasium as gym
from gymnasium import spaces

# Camera-based State Estimation (simplified for Pygame)
class CameraStateEstimator:
    def __init__(self, image_size=(224, 224)):
        self.image_size = image_size
        self.feature_extractor = self._create_feature_extractor()
    
    def _create_feature_extractor(self):
        model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        return model
    
    def capture_and_process_image(self, screen):
        # Convert Pygame screen to tensor
        screen_array = pygame.surfarray.array3d(screen)
        image_tensor = torch.from_numpy(screen_array).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)
        
        resized_image = torch.nn.functional.interpolate(
            image_tensor, 
            size=self.image_size, 
            mode='bilinear', 
            align_corners=False
        )
        
        with torch.no_grad():
            image_features = self.feature_extractor(resized_image)
        
        return image_features

# Advanced Reward Shaping
class AdvancedRewardShaper:
    def __init__(self):
        self.reward_components = {
            'stability': 1.0,
            'precision': 1.5,
            'complexity_bonus': 0.5,
            'risk_penalty': 1.0
        }
    
    def calculate_reward(self, env):
        stability_score = env._calculate_tower_stability()
        stability_reward = stability_score * self.reward_components['stability']
        
        precision_reward = self._calculate_block_precision(env)
        complexity_bonus = self._calculate_complexity_bonus(env)
        risk_penalty = self._calculate_risk_penalty(env)
        
        total_reward = (
            stability_reward + 
            precision_reward * self.reward_components['precision'] +
            complexity_bonus * self.reward_components['complexity_bonus'] -
            risk_penalty * self.reward_components['risk_penalty']
        )
        
        return total_reward
    
    def _calculate_block_precision(self, env):
        block_positions = [block.body.position for block in env.blocks]
        positional_variance = np.var(block_positions, axis=0)
        return 1.0 / (1 + np.mean(positional_variance))
    
    def _calculate_complexity_bonus(self, env):
        orientations = [block.body.angle for block in env.blocks]
        orientation_complexity = np.std(orientations)
        return np.exp(-orientation_complexity)
    
    def _calculate_risk_penalty(self, env):
        velocities = [block.body.velocity for block in env.blocks]
        max_velocity = np.max(np.abs(velocities))
        return max_velocity

# Advanced Jenga Environment
class AdvancedJengaEnvironment(gym.Env):
    def __init__(self, render_mode=True, max_steps=100, screen_width=800, screen_height=600):
        super().__init__()
        
        # Pygame and Pymunk setup
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        
        # Setup screen for rendering
        self.screen = pygame.display.set_mode((screen_width, screen_height))
        pygame.display.set_caption("Jenga AI Simulation")
        
        # Pymunk space setup
        self.space = pymunk.Space()
        self.space.gravity = (0, 980)  # Pymunk uses pixels per second squared
        
        # Draw options for Pymunk
        self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
        
        # Camera state estimator
        self.camera_estimator = CameraStateEstimator()
        
        # Advanced reward shaper
        self.reward_shaper = AdvancedRewardShaper()
        
        # Environment parameters
        self.max_steps = max_steps
        self.current_step = 0
        
        # Block and tower parameters
        self.block_width = 75
        self.block_height = 25
        self.blocks = []
        
        # Action and observation spaces
        self.action_space = spaces.Dict({
            'block_index': spaces.Discrete(54),
            'extraction_x': spaces.Box(low=0, high=self.screen_width, shape=(1,)),
            'extraction_y': spaces.Box(low=0, high=self.screen_height, shape=(1,))
        })
        
        self.observation_space = spaces.Dict({
            'block_states': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(54, 4),  # pos_x, pos_y, angle, velocity
                dtype=np.float32
            ),
            'tower_stability': spaces.Box(
                low=0, high=1, 
                shape=(1,), 
                dtype=np.float32
            ),
            'camera_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(256,),
                dtype=np.float32
            )
        })
        
        # Create initial tower
        self._create_tower()
        self._create_ground()
    
    def _create_ground(self):
        # Create ground body
        ground_body = self.space.static_body
        ground_shape = pymunk.Segment(ground_body, (0, self.screen_height-50), 
                                      (self.screen_width, self.screen_height-50), 5)
        ground_shape.friction = 0.5
        self.space.add(ground_shape)
    
    def _create_tower(self):
        # Create Jenga tower with alternate layer orientations
        for layer in range(9):  # 9 layers
            y_offset = self.screen_height - 100 - layer * self.block_height * 1.1
            horizontal_direction = layer % 2 == 0
            
            for i in range(3):  # 3 blocks per layer
                mass = 1
                moment = pymunk.moment_for_box(mass, (self.block_width, self.block_height))
                body = pymunk.Body(mass, moment)
                
                if horizontal_direction:
                    body.position = (
                        self.screen_width/2 + (i-1) * self.block_width, 
                        y_offset
                    )
                    shape = pymunk.Poly.create_box(body, (self.block_width, self.block_height))
                else:
                    body.position = (
                        self.screen_width/2 + (i-1) * self.block_height, 
                        y_offset
                    )
                    shape = pymunk.Poly.create_box(body, (self.block_height, self.block_width))
                
                shape.friction = 0.5
                shape.elasticity = 0.1
                self.space.add(body, shape)
                self.blocks.append(shape)
    
    def _calculate_tower_stability(self):
        # Calculate tower stability based on block positions and velocities
        velocities = [np.linalg.norm(block.body.velocity) for block in self.blocks]
        height_variation = np.std([block.body.position.y for block in self.blocks])
        
        # Stability is high when velocities are low and height variation is minimal
        stability = max(0, 1 - np.mean(velocities)/100 - height_variation/100)
        return stability
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        
        # Clear existing bodies
        for block in self.blocks:
            self.space.remove(block.body, block)
        self.blocks.clear()
        
        # Recreate tower
        self._create_tower()
        
        # Capture initial state
        camera_features = self._capture_camera_features()
        observation = self._get_comprehensive_observation(camera_features)
        
        return observation, {}
    
    def _capture_camera_features(self):
        # Render current state to screen
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        
        # Get camera features
        return self.camera_estimator.capture_and_process_image(self.screen)
    
    def _get_comprehensive_observation(self, camera_features):
        block_states = []
        
        for block in self.blocks:
            # Extract key block state information
            block_state = [
                block.body.position.x, 
                block.body.position.y, 
                block.body.angle, 
                np.linalg.norm(block.body.velocity)
            ]
            block_states.append(block_state)
        
        stability_score = self._calculate_tower_stability()
        
        return {
            'block_states': np.array(block_states),
            'tower_stability': np.array([stability_score]),
            'camera_features': camera_features.numpy().flatten()
        }
    
    def step(self, action):
        self.current_step += 1
        
        # Extract action parameters
        block_index = action['block_index']
        extraction_x = action['extraction_x'][0]
        extraction_y = action['extraction_y'][0]
        
        # Apply force to selected block
        selected_block = self.blocks[block_index]
        force_magnitude = 500
        force_direction = pymunk.Vec2d(
            extraction_x - selected_block.body.position.x,
            extraction_y - selected_block.body.position.y
        ).normalized()
        
        selected_block.body.apply_force_at_local_point(
            force_direction * force_magnitude, 
            (0, 0)
        )
        
        # Step physics simulation
        for _ in range(10):  # Substeps for more stable physics
            self.space.step(0.01)
        
        # Capture camera features
        camera_features = self._capture_camera_features()
        
        # Get observation
        observation = self._get_comprehensive_observation(camera_features)
        
        # Calculate reward
        reward = self.reward_shaper.calculate_reward(self)
        
        # Check termination conditions
        terminated = (
            observation['tower_stability'][0] < 0.1 or
            self.current_step >= self.max_steps
        )
        
        truncated = False
        
        return observation, reward, terminated, truncated, {}
    
    def render(self):
        # Render current physics state
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
    
    def close(self):
        pygame.quit()

# Training function
def train_advanced_jenga_agent():
    # Create environment
    env = AdvancedJengaEnvironment(render_mode=True)
    env = DummyVecEnv([lambda: env])
    
    # Custom neural network architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(shared=[256, 128], pi=[128], vf=[128])]
    )
    
    # Train PPO with advanced configuration
    model = PPO(
        "MultiInputPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=1e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        verbose=1,
        tensorboard_log="./jenga_tensorboard/"
    )
    
    # Train with enhanced learning
    model.learn(
        total_timesteps=100000
    )
    
    return model

def main():
    # Train Jenga agent
    trained_model = train_advanced_jenga_agent()
    
    # Optional: Demonstration of model
    env = AdvancedJengaEnvironment(render_mode=True)
    obs, _ = env.reset()
    
    for _ in range(100):
        action, _ = trained_model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        env.render()
        
        if done:
            break
    
    env.close()

if __name__ == "__main__":
    main()
