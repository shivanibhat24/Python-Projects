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
import ray

# Camera-based State Estimation
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
    
    def capture_and_process_image(self, pybullet_client):
        width, height = 640, 480
        img_arr = pybullet_client.getCameraImage(
            width, height, 
            renderer=pybullet_client.ER_BULLET_HARDWARE_OPENGL
        )
        rgb_image = img_arr[2]
        
        image_tensor = torch.from_numpy(rgb_image).float() / 255.0
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
        block_positions = [
            p.getBasePositionAndOrientation(block['id'])[0] 
            for block in env.blocks
        ]
        positional_variance = np.var(block_positions, axis=0)
        return 1.0 / (1 + np.mean(positional_variance))
    
    def _calculate_complexity_bonus(self, env):
        orientations = [
            p.getEulerFromQuaternion(
                p.getBasePositionAndOrientation(block['id'])[1]
            ) 
            for block in env.blocks
        ]
        
        orientation_complexity = np.std(orientations)
        return np.exp(-orientation_complexity)
    
    def _calculate_risk_penalty(self, env):
        velocities = [
            p.getBaseVelocity(block['id'])[0] 
            for block in env.blocks
        ]
        
        max_velocity = np.max(np.abs(velocities))
        return max_velocity

# Transfer Learning Support
class JengaTransferLearner:
    def __init__(self, base_model_path=None):
        self.base_model = self._load_or_create_base_model(base_model_path)
        
    def _load_or_create_base_model(self, model_path):
        if model_path:
            try:
                return PPO.load(model_path)
            except Exception as e:
                print(f"Could not load base model: {e}")
        
        env = DummyVecEnv([lambda: AdvancedJengaEnvironment()])
        return PPO("MultiInputPolicy", env)
    
    def fine_tune(self, new_environment, training_steps=50000):
        transfer_policy_kwargs = dict(
            activation_fn=torch.nn.ReLU,
            net_arch=[
                dict(shared=[256, 128], 
                     pi=[64], 
                     vf=[64])
            ]
        )
        
        fine_tuned_model = PPO(
            "MultiInputPolicy",
            new_environment,
            policy_kwargs=transfer_policy_kwargs,
            learning_rate=5e-5,
            **self.base_model.get_parameters()
        )
        
        fine_tuned_model.learn(total_timesteps=training_steps)
        return fine_tuned_model

# Multi-Agent Training
class MultiAgentJengaTrainer:
    def __init__(self, num_agents=4):
        ray.init(num_cpus=num_agents)
        self.num_agents = num_agents
        self.environments = [AdvancedJengaEnvironment() for _ in range(num_agents)]
    
    @ray.remote
    def train_agent(self, agent_id):
        env = self.environments[agent_id]
        
        model = PPO(
            "MultiInputPolicy", 
            env,
            verbose=0,
            learning_rate=1e-4
        )
        
        model.learn(
            total_timesteps=50000,
            callback=self._create_collaborative_callback(agent_id)
        )
        
        return {
            'agent_id': agent_id,
            'final_performance': self._evaluate_agent(model, env)
        }
    
    def _create_collaborative_callback(self, agent_id):
        class CollaborativeCallback(BaseCallback):
            def __init__(self, agent_id, verbose=0):
                super().__init__(verbose)
                self.agent_id = agent_id
            
            def _on_step(self):
                if self.num_timesteps % 1000 == 0:
                    print(f"Agent {self.agent_id} progress: {self.num_timesteps}")
                return True
        
        return CollaborativeCallback(agent_id)
    
    def _evaluate_agent(self, model, env):
        obs, _ = env.reset()
        total_reward = 0
        
        for _ in range(100):
            action, _ = model.predict(obs)
            obs, reward, done, _, _ = env.step(action)
            total_reward += reward
            
            if done:
                break
        
        return total_reward
    
    def parallel_training(self):
        agent_futures = [self.train_agent.remote(i) for i in range(self.num_agents)]
        return ray.get(agent_futures)

# Modified AdvancedJengaEnvironment with enhanced capabilities
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
        
        # Camera state estimator
        self.camera_estimator = CameraStateEstimator()
        
        # Advanced reward shaper
        self.reward_shaper = AdvancedRewardShaper()
        
        # Rest of the original initialization...
        self.max_steps = max_steps
        self.current_step = 0
        
        # More complex action space
        self.action_space = spaces.Dict({
            'block_index': spaces.Discrete(54),
            'extraction_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,)),
            'extraction_force': spaces.Box(low=0, high=10, shape=(1,))
        })
        
        # Enhanced observation space
        self.observation_space = spaces.Dict({
            'block_states': spaces.Box(
                low=-np.inf, 
                high=np.inf, 
                shape=(54, 7),
                dtype=np.float32
            ),
            'tower_stability': spaces.Box(
                low=0, high=1, 
                shape=(1,), 
                dtype=np.float32
            ),
            'camera_features': spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(256,),  # Matches feature extractor output
                dtype=np.float32
            )
        })
        
        self.blocks = []
        self.tower_height = 0
    
    def reset(self, seed=None):
        super().reset(seed=seed)
        p.resetSimulation()
        
        self.blocks = self._create_advanced_jenga_tower()
        self.current_step = 0
        
        # Capture initial camera features
        camera_features = self.camera_estimator.capture_and_process_image(p)
        
        observation = self._get_comprehensive_observation(camera_features)
        return observation, {}
    
    def _get_comprehensive_observation(self, camera_features):
        block_states = []
        
        for block in self.blocks:
            pos, ori = p.getBasePositionAndOrientation(block['id'])
            linear_vel, angular_vel = p.getBaseVelocity(block['id'])
            
            block_state = list(pos) + list(ori) + list(linear_vel)
            block_states.append(block_state)
        
        stability_score = self._calculate_tower_stability()
        
        return {
            'block_states': np.array(block_states),
            'tower_stability': np.array([stability_score]),
            'camera_features': camera_features.numpy().flatten()
        }
    
    def step(self, action):
        self.current_step += 1
        
        # Original step logic with enhanced reward calculation
        block_index = action['block_index']
        angle = action['extraction_angle'][0]
        force = action['extraction_force'][0]
        
        block = self.blocks[block_index]
        p.applyExternalForce(
            block['id'], 
            -1,
            [np.cos(angle) * force, 0, np.sin(angle) * force],
            p.getBasePositionAndOrientation(block['id'])[0],
            p.BASE_LINK
        )
        
        p.stepSimulation()
        
        # Capture camera features
        camera_features = self.camera_estimator.capture_and_process_image(p)
        
        observation = self._get_comprehensive_observation(camera_features)
        
        # Advanced reward calculation
        reward = self.reward_shaper.calculate_reward(self)
        
        terminated = (
            observation['tower_stability'][0] < 0.1 or
            self.current_step >= self.max_steps
        )
        
        truncated = False
        
        return observation, reward, terminated, truncated, {}

# Main training function
def train_advanced_jenga_agent():
    # Create environment with enhanced capabilities
    env = AdvancedJengaEnvironment(render_mode=True)
    env = DummyVecEnv([lambda: env])
    
    # Transfer learning support
    transfer_learner = JengaTransferLearner()
    
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
        total_timesteps=100000, 
        callback=CustomCallback()
    )
    
    # Optional: Fine-tune on a slightly modified environment
    modified_env = AdvancedJengaEnvironment(render_mode=False)
    fine_tuned_model = transfer_learner.fine_tune(modified_env)
    
    return model, fine_tuned_model

def main():
    # Train advanced Jenga agent
    trained_model, fine_tuned_model = train_advanced_jenga_agent()
    
    # Multi-agent training demonstration
    multi_agent_trainer = MultiAgentJengaTrainer(num_agents=4)
    multi_agent_results = multi_agent_trainer.parallel_training()
    
    print("Multi-Agent Training Results:", multi_agent_results)

if __name__ == "__main__":
    main()
