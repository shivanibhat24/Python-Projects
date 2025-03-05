import pygame
import pymunk
import pymunk.pygame_util
import pybullet as p
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
import cv2

class AdvancedJengaEnvironment(gym.Env):
    def __init__(self, render_mode=True, max_steps=100):
        super().__init__()
        
        # Enhanced physics setup
        self.client = p.connect(p.GUI if render_mode else p.DIRECT)
        p.setGravity(0, -9.81, 0)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1/240.0,
            numSolverIterations=200
        )
        
        # Advanced action and observation spaces
        self.max_steps = max_steps
        self.current_step = 0
        
        # More complex action space: block selection, extraction direction, force
        self.action_space = spaces.Dict({
            'block_index': spaces.Discrete(54),  # Total Jenga blocks
            'extraction_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'extraction_force': spaces.Box(low=0, high=10, shape=(1,))
        })
        
        # Detailed observation space
        self.observation_space = spaces.Dict({
            'block_states': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(54, 7),  # 54 blocks, 7 features per block
                dtype=np.float32
            ),
            'tower_stability': spaces.Box(
                low=0, high=1, 
                shape=(1,), 
                dtype=np.float32
            )
        })
        
        self.blocks = []
        self.tower_height = 0
        
    def reset(self, seed=None):
        super().reset(seed=seed)
        p.resetSimulation()
        
        # Enhanced tower creation with more realistic physics
        self.blocks = self._create_advanced_jenga_tower()
        self.current_step = 0
        
        observation = self._get_comprehensive_observation()
        return observation, {}
    
    def _create_advanced_jenga_tower(self):
        blocks = []
        block_dimensions = [0.025, 0.075, 0.015]
        layers = 18  # More layers for complexity
        
        for layer in range(layers):
            orientation = 0 if layer % 2 == 0 else np.pi/2
            
            for block in range(3):
                x_offset = (block - 1) * 0.026
                z_offset = layer * 0.016
                
                block_pos = [
                    x_offset, 
                    z_offset, 
                    layer * 0.075
                ]
                
                block_orientation = p.getQuaternionFromEuler([0, orientation, 0])
                
                block_id = p.loadURDF(
                    "block.urdf",
                    basePosition=block_pos,
                    baseOrientation=block_orientation,
                    useFixedBase=False
                )
                
                blocks.append({
                    'id': block_id,
                    'position': block_pos,
                    'orientation': block_orientation
                })
        
        self.tower_height = layers * 0.075
        return blocks
    
    def _get_comprehensive_observation(self):
        block_states = []
        
        for block in self.blocks:
            pos, ori = p.getBasePositionAndOrientation(block['id'])
            linear_vel, angular_vel = p.getBaseVelocity(block['id'])
            
            # Comprehensive block state: pos_x, pos_y, pos_z, ori_x, ori_y, ori_z, linear_velocity
            block_state = list(pos) + list(ori) + list(linear_vel)
            block_states.append(block_state)
        
        # Tower stability calculation
        stability_score = self._calculate_tower_stability()
        
        return {
            'block_states': np.array(block_states),
            'tower_stability': np.array([stability_score])
        }
    
    def _calculate_tower_stability(self):
        # Advanced stability calculation
        center_of_mass = np.mean([p.getBasePositionAndOrientation(block['id'])[0] for block in self.blocks], axis=0)
        
        # Calculate variance and tilting
        positions = np.array([p.getBasePositionAndOrientation(block['id'])[0] for block in self.blocks])
        position_variance = np.var(positions, axis=0)
        
        # Check for excessive tilting
        orientations = [p.getBasePositionAndOrientation(block['id'])[1] for block in self.blocks]
        euler_angles = [p.getEulerFromQuaternion(ori) for ori in orientations]
        max_tilt = np.max(np.abs(euler_angles))
        
        # Combine metrics
        stability = 1.0 - (np.mean(position_variance) + max_tilt/np.pi)
        return max(0, min(1, stability))
    
    def step(self, action):
        self.current_step += 1
        
        # Extract block and apply extraction
        block_index = action['block_index']
        angle = action['extraction_angle'][0]
        force = action['extraction_force'][0]
        
        # Apply extraction force to specific block
        block = self.blocks[block_index]
        p.applyExternalForce(
            block['id'], 
            -1,  # Link index (-1 for base)
            [np.cos(angle) * force, 0, np.sin(angle) * force],
            p.getBasePositionAndOrientation(block['id'])[0],
            p.BASE_LINK
        )
        
        # Physics simulation
        p.stepSimulation()
        
        # Get new observation
        observation = self._get_comprehensive_observation()
        
        # Reward calculation
        stability_score = observation['tower_stability'][0]
        
        # Penalty for tower collapse, reward for maintaining stability
        reward = stability_score - (1 if stability_score < 0.3 else 0)
        
        # Termination conditions
        terminated = (
            stability_score < 0.1 or  # Tower collapsed
            self.current_step >= self.max_steps  # Max steps reached
        )
        
        truncated = False
        
        return observation, reward, terminated, truncated, {}
    
    def render(self, mode='human'):
        # Optional advanced rendering
        pass

class CustomCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.best_mean_reward = -np.inf
    
    def _on_step(self) -> bool:
        # Log training progress
        if self.num_timesteps % 1000 == 0:
            mean_reward = np.mean(self.training_env.get_attr('rewards'))
            print(f"Step {self.num_timesteps}: Mean Reward = {mean_reward}")
            
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.model.save("best_jenga_model")
        
        return True

def train_advanced_jenga_agent():
    # Create environment
    env = AdvancedJengaEnvironment(render_mode=True)
    env = DummyVecEnv([lambda: env])
    
    # Custom neural network architecture
    policy_kwargs = dict(
        activation_fn=torch.nn.ReLU,
        net_arch=[dict(pi=[256, 128], vf=[256, 128])]
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
    
    # Train with custom callback
    model.learn(
        total_timesteps=100000, 
        callback=CustomCallback()
    )
    
    return model

def visualize_trained_agent(model, env):
    # Visualization of trained agent performance
    obs, _ = env.reset()
    for _ in range(200):
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = env.step(action)
        
        if done:
            break

def main():
    # Train advanced Jenga agent
    trained_model = train_advanced_jenga_agent()
    
    # Create environment for visualization
    env = AdvancedJengaEnvironment(render_mode=True)
    
    # Visualize trained performance
    visualize_trained_agent(trained_model, env)

if __name__ == "__main__":
    main()
