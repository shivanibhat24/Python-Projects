import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from collections import deque
from tensordict import TensorDict
import gymnasium as gym
from gymnasium import spaces

class RubiksCubeEnv:
    """
    Rubik's Cube environment that simulates a 3x3 Rubik's Cube.
    This is a simplified version for demonstration purposes.
    """
    
    def __init__(self):
        # Define the action space (12 possible moves for a 3x3 cube)
        # Each face can be rotated clockwise or counterclockwise
        # F, F', B, B', U, U', D, D', L, L', R, R'
        self.action_space = spaces.Discrete(12)
        
        # Initialize the cube state
        self.reset()
    
    def reset(self):
        """Reset the cube to a solved state."""
        # Represents a 3x3 cube with 6 faces
        # Each face has 9 stickers, and there are 6 colors
        # 0: white, 1: yellow, 2: red, 3: orange, 4: blue, 5: green
        self.state = np.array([
            # Face 0 (Up)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],
            # Face 1 (Down)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],
            # Face 2 (Front)
            [2, 2, 2, 2, 2, 2, 2, 2, 2],
            # Face 3 (Back)
            [3, 3, 3, 3, 3, 3, 3, 3, 3],
            # Face 4 (Left)
            [4, 4, 4, 4, 4, 4, 4, 4, 4],
            # Face 5 (Right)
            [5, 5, 5, 5, 5, 5, 5, 5, 5]
        ])
        return self.state.flatten()
    
    def _get_state(self):
        """Return the current state of the cube as a flattened array."""
        return self.state.flatten()
    
    def is_solved(self):
        """Check if the cube is solved."""
        for face in range(6):
            color = self.state[face, 0]
            if not np.all(self.state[face] == color):
                return False
        return True
    
    def _rotate_face(self, face, clockwise=True):
        """Rotate a face of the cube."""
        if clockwise:
            # Rotate the face stickers
            self.state[face] = self.state[face].reshape(3, 3).transpose(1, 0)[:, ::-1].flatten()
        else:
            # Rotate counter-clockwise
            self.state[face] = self.state[face].reshape(3, 3).transpose(1, 0)[::-1, :].flatten()
            
        # Update adjacent faces (simplified for demonstration)
        # In a real implementation, this would update the stickers on adjacent faces
    
    def step(self, action):
        """
        Execute one step in the environment.
        
        Args:
            action: An integer in the range [0, 11] representing one of the 12 possible moves.
            
        Returns:
            next_state: The new state after taking the action.
            reward: The reward for taking the action.
            done: Whether the episode is done (cube is solved).
            info: Additional information.
        """
        # Determine which face to rotate and in which direction
        face = action // 2
        clockwise = action % 2 == 0
        
        # Apply the rotation
        self._rotate_face(face, clockwise)
        
        # Check if the cube is solved
        done = self.is_solved()
        
        # Reward scheme: positive reward for solving, small negative for each move
        reward = 10.0 if done else -0.1
        
        return self._get_state(), reward, done, {}
    
    def scramble(self, num_moves):
        """Scramble the cube with random moves."""
        for _ in range(num_moves):
            action = random.randint(0, 11)
            self.step(action)
    
    def render(self, canvas):
        """Render the cube on a tkinter canvas."""
        canvas.delete("all")
        
        # This is a simplified 2D rendering
        # In a real implementation, this would draw a proper 3D cube
        
        # Define the size and position of the cube faces
        size = 40
        padding = 10
        
        # Define the positions of the 6 faces in a net layout
        positions = [
            (1, 0),  # Up
            (1, 2),  # Down
            (1, 1),  # Front
            (3, 1),  # Back
            (0, 1),  # Left
            (2, 1)   # Right
        ]
        
        # Define colors for each face value
        colors = ["white", "yellow", "red", "orange", "blue", "green"]
        
        for face_idx, (x_offset, y_offset) in enumerate(positions):
            face_state = self.state[face_idx].reshape(3, 3)
            
            for i in range(3):
                for j in range(3):
                    color_idx = face_state[i, j]
                    x = padding + (x_offset * 3 + j) * size
                    y = padding + (y_offset * 3 + i) * size
                    
                    canvas.create_rectangle(
                        x, y, x + size, y + size,
                        fill=colors[color_idx],
                        outline="black",
                        width=2
                    )
                    
                    # Add face label in the center
                    if i == 1 and j == 1:
                        face_labels = ["U", "D", "F", "B", "L", "R"]
                        canvas.create_text(
                            x + size // 2, y + size // 2,
                            text=face_labels[face_idx],
                            fill="black",
                            font=("Arial", 12, "bold")
                        )


class EnvWrapper:
    """Wrapper for the RubiksCubeEnv to make it compatible with TensorDict."""
    
    def __init__(self, env):
        self.env = env
    
    def reset(self):
        state = self.env.reset()
        return TensorDict(
            {"observation": torch.tensor(state, dtype=torch.float32).unsqueeze(0)},
            batch_size=[1]
        )
    
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return TensorDict(
            {
                "observation": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                "reward": torch.tensor([reward], dtype=torch.float32),
                "done": torch.tensor([done], dtype=torch.bool)
            },
            batch_size=[1]
        )


class QNetwork(nn.Module):
    """Neural network for Q-value estimation."""
    
    def __init__(self, state_dim, action_dim):
        super(QNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQNAgent:
    """Deep Q-Network agent for solving the Rubik's Cube."""
    
    def __init__(self, state_dim, action_dim, gamma=0.99, epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        
        # Q-Networks
        self.q_network = QNetwork(state_dim, action_dim)
        self.target_network = QNetwork(state_dim, action_dim)
        self.update_target_network()
        
        # Optimizer
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        
        # Memory
        self.memory = deque(maxlen=10000)
        
        # Metrics
        self.rewards = []
        self.losses = []
        
        # Training step counter
        self.train_step_counter = 0
        
        # Target update frequency
        self.target_update_freq = 100
    
    def update_target_network(self):
        """Update the target network with the current Q-network weights."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def select_action(self, state, evaluate=False):
        """Select an action using epsilon-greedy policy."""
        if not evaluate and random.random() < self.epsilon:
            return torch.tensor([[random.randrange(self.action_dim)]])
        
        with torch.no_grad():
            q_values = self.q_network(state)
            return torch.argmax(q_values, dim=1, keepdim=True)
    
    def add_experience(self, state, action, next_state, reward, done):
        """Add experience to memory."""
        self.memory.append((
            state["observation"], 
            action, 
            next_state["observation"], 
            reward, 
            done
        ))
    
    def update(self, batch_size):
        """Update the Q-network using a batch of experiences."""
        if len(self.memory) < batch_size:
            return None
        
        # Sample a batch of experiences
        batch = random.sample(self.memory, batch_size)
        
        # Prepare batch data
        states = torch.cat([batch[i][0] for i in range(batch_size)], dim=0)
        actions = torch.tensor([batch[i][1] for i in range(batch_size)], dtype=torch.long).unsqueeze(1)
        next_states = torch.cat([batch[i][2] for i in range(batch_size)], dim=0)
        rewards = torch.tensor([batch[i][3] for i in range(batch_size)], dtype=torch.float).unsqueeze(1)
        dones = torch.tensor([batch[i][4] for i in range(batch_size)], dtype=torch.float).unsqueeze(1)
        
        # Compute current Q-values
        current_q = self.q_network(states).gather(1, actions)
        
        # Compute target Q-values
        with torch.no_grad():
            max_next_q = self.target_network(next_states).max(1, keepdim=True)[0]
            target_q = rewards + (1 - dones) * self.gamma * max_next_q
        
        # Compute loss
        loss = F.mse_loss(current_q, target_q)
        
        # Update Q-network
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Store loss
        loss_value = loss.item()
        self.losses.append(loss_value)
        
        # Update target network periodically
        self.train_step_counter += 1
        if self.train_step_counter % self.target_update_freq == 0:
            self.update_target_network()
        
        return loss_value
    
    def save_model(self, path):
        """Save the Q-network model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'rewards': self.rewards,
            'losses': self.losses
        }, path)
    
    def load_model(self, path):
        """Load the Q-network model."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.rewards = checkpoint['rewards']
        self.losses = checkpoint['losses']
