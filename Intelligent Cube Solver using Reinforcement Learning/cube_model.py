import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from collections import deque
import random
import copy
import os

class CubeEnvironment:
    """
    Rubik's Cube environment that simulates a 3x3 cube with standard moves.
    """
    def __init__(self):
        # Define the colors
        # 0: White, 1: Yellow, 2: Red, 3: Orange, 4: Blue, 5: Green
        self.colors = ["white", "yellow", "red", "orange", "blue", "green"]
        
        # Define the cube as 6 faces (up, down, front, back, left, right)
        # Each face is a 3x3 grid where each cell has a color value
        self.reset()
        
        # Define possible actions (18 total: 6 faces * 3 rotations)
        # For each face: clockwise, counterclockwise, and 180 degrees
        self.actions = 12  # U, U', F, F', R, R', B, B', L, L', D, D'
        self.action_space = list(range(self.actions))
        
        # Current state for tracking if the cube is solved
        self.done = False
    
    def reset(self):
        """
        Reset the cube to its solved state.
        """
        # Initialize each face with its color
        self.cube = {
            'up': np.full((3, 3), 0),      # White
            'down': np.full((3, 3), 1),    # Yellow
            'front': np.full((3, 3), 2),   # Red
            'back': np.full((3, 3), 3),    # Orange
            'left': np.full((3, 3), 4),    # Blue
            'right': np.full((3, 3), 5)    # Green
        }
        
        self.done = self._check_solved()
        return self._get_observation()
    
    def _check_solved(self):
        """
        Check if the cube is solved.
        """
        for face, grid in self.cube.items():
            # If any face has more than one color, it's not solved
            if not np.all(grid == grid[0, 0]):
                return False
        return True
    
    def get_state_dict(self):
        """
        Return the cube state as a dictionary with face arrays converted to lists.
        """
        state_dict = {}
        for face, grid in self.cube.items():
            state_dict[face] = grid.tolist()
        return state_dict
    
    def set_state(self, state_dict):
        """
        Set the cube state from a dictionary representation.
        """
        for face, grid in state_dict.items():
            self.cube[face] = np.array(grid)
        
        self.done = self._check_solved()
    
    def _get_observation(self):
        """
        Convert the cube state to a flattened one-hot encoding observation.
        """
        # Flatten the cube faces
        observation = []
        
        for face in ['up', 'down', 'front', 'back', 'left', 'right']:
            face_grid = self.cube[face]
            # Flatten the 3x3 grid to a 9-element array
            face_flat = face_grid.flatten()
            
            # Convert to one-hot encoding
            for i in range(9):
                color = face_flat[i]
                one_hot = [0] * 6  # 6 colors
                one_hot[color] = 1
                observation.extend(one_hot)
        
        return np.array(observation)
    
    def _rotate_face(self, face, direction):
        """
        Rotate a face of the cube.
        
        Args:
            face: The face to rotate ('up', 'down', 'front', 'back', 'left', 'right')
            direction: 1 for clockwise, -1 for counterclockwise, 2 for 180 degrees
        """
        # Make a copy of the face
        face_grid = self.cube[face].copy()
        
        if direction == 1:  # Clockwise
            self.cube[face] = np.rot90(face_grid, 3)
        elif direction == -1:  # Counterclockwise
            self.cube[face] = np.rot90(face_grid, 1)
        elif direction == 2:  # 180 degrees
            self.cube[face] = np.rot90(face_grid, 2)
    
    def _update_adjacent_faces(self, face, direction):
        """
        Update the adjacent faces after rotating a face.
        
        Args:
            face: The face that was rotated
            direction: Direction of rotation (1, -1, or 2)
        """
        # Define the adjacent faces and edges for each face
        adjacent_faces = {
            'up': {
                'faces': ['back', 'right', 'front', 'left'],
                'edges': [(0, slice(None, None, -1)), (0, slice(None)), (0, slice(None)), (0, slice(None, None, -1))],
                'clockwise': True
            },
            'down': {
                'faces': ['front', 'right', 'back', 'left'],
                'edges': [(2, slice(None)), (2, slice(None)), (2, slice(None, None, -1)), (2, slice(None, None, -1))],
                'clockwise': True
            },
            'front': {
                'faces': ['up', 'right', 'down', 'left'],
                'edges': [(2, slice(None)), (slice(None), 0), (0, slice(None, None, -1)), (slice(None, None, -1), 2)],
                'clockwise': True
            },
            'back': {
                'faces': ['up', 'left', 'down', 'right'],
                'edges': [(0, slice(None, None, -1)), (slice(None), 0), (2, slice(None)), (slice(None, None, -1), 2)],
                'clockwise': True
            },
            'left': {
                'faces': ['up', 'front', 'down', 'back'],
                'edges': [(slice(None), 0), (slice(None), 0), (slice(None), 0), (slice(None, None, -1), 2)],
                'clockwise': True
            },
            'right': {
                'faces': ['up', 'back', 'down', 'front'],
                'edges': [(slice(None), 2), (slice(None, None, -1), 0), (slice(None), 2), (slice(None), 2)],
                'clockwise': True
            }
        }
        
        adj = adjacent_faces[face]
        faces = adj['faces']
        edges = adj['edges']
        
        # Get the current values from all adjacent faces
        current_values = []
        for i in range(4):
            f = faces[i]
            edge = edges[i]
            current_values.append(self.cube[f][edge].copy())
        
        # Determine how to shift values based on rotation direction
        if direction == 1:  # Clockwise
            shifts = 1
        elif direction == -1:  # Counterclockwise
            shifts = 3
        else:  # 180 degrees
            shifts = 2
        
        # Update the adjacent faces
        for i in range(4):
            f = faces[i]
            edge = edges[i]
            prev_idx = (i - shifts) % 4
            self.cube[f][edge] = current_values[prev_idx]
    
    def action_to_string(self, action):
        """
        Convert action index to a string representation.
        """
        actions = [
            "U", "U'", "F", "F'", "R", "R'", "B", "B'", "L", "L'", "D", "D'"
        ]
        return actions[action]
    
    def step(self, action):
        """
        Perform a move on the cube.
        
        Args:
            action: Integer representing the action to take (0-11)
        
        Returns:
            observation: The new state observation
            reward: The reward received
            done: Whether the cube is solved
            info: Additional information
        """
        # Map action to face and direction
        face_map = {
            0: ('up', 1),       # U
            1: ('up', -1),      # U'
            2: ('front', 1),    # F
            3: ('front', -1),   # F'
            4: ('right', 1),    # R
            5: ('right', -1),   # R'
            6: ('back', 1),     # B
            7: ('back', -1),    # B'
            8: ('left', 1),     # L
            9: ('left', -1),    # L'
            10: ('down', 1),    # D
            11: ('down', -1)    # D'
        }
        
        if action not in range(self.actions):
            raise ValueError(f"Invalid action: {action}")
        
        # Get the face and direction to rotate
        face, direction = face_map[action]
        
        # Rotate the face
        self._rotate_face(face, direction)
        
        # Update adjacent faces
        self._update_adjacent_faces(face, direction)
        
        # Check if the cube is solved
        self.done = self._check_solved()
        
        # Calculate reward
        if self.done:
            reward = 100.0  # Big reward for solving the cube
        else:
            # Negative reward to encourage finding shortest solution
            reward = -1.0
        
        return self._get_observation(), reward, self.done, {}
    
    def scramble(self, moves):
        """
        Scramble the cube with random moves.
        
        Args:
            moves: Number of random moves to make
        
        Returns:
            List of actions taken
        """
        actions = []
        for _ in range(moves):
            action = random.choice(self.action_space)
            self.step(action)
            actions.append(action)
        
        return actions

class DQNAgent:
    """
    Deep Q-Network agent for solving Rubik's cube.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
    
    def _build_model(self):
        """
        Build a neural network model for DQN.
        """
        model = Sequential()
        model.add(Dense(512, input_dim=self.state_size, activation='relu'))
        model.add(Dense(256, activation='relu'))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(learning_rate=self.learning_rate))
        return model
    
    def update_target_model(self):
        """
        Copy weights from model to target_model.
        """
        self.target_model.set_weights(self.model.get_weights())
    
    def remember(self, state, action, reward, next_state, done):
        """
        Store experience in memory.
        """
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state, test=False):
        """
        Choose an action based on the current state.
        
        Args:
            state: Current state observation
            test: If True, act greedily (no exploration)
        """
        if not test and np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        
        state = np.reshape(state, [1, self.state_size])
        act_values = self.model.predict(state, verbose=0)
        return np.argmax(act_values[0])
    
    def replay(self, batch_size):
        """
        Train the model using experience replay.
        """
        if len(self.memory) < batch_size:
            return 0
        
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([i[0] for i in minibatch])
        actions = np.array([i[1] for i in minibatch])
        rewards = np.array([i[2] for i in minibatch])
        next_states = np.array([i[3] for i in minibatch])
        dones = np.array([i[4] for i in minibatch])
        
        states = np.reshape(states, [batch_size, self.state_size])
        next_states = np.reshape(next_states, [batch_size, self.state_size])
        
        # Get target Q values from target model
        target = rewards + self.gamma * np.amax(self.target_model.predict(next_states, verbose=0), axis=1) * (1 - dones)
        
        # Get current Q values from model
        target_f = self.model.predict(states, verbose=0)
        
        # Update the Q values for actions taken
        for i, action in enumerate(actions):
            target_f[i][action] = target[i]
        
        # Train the model
        history = self.model.fit(states, target_f, epochs=1, verbose=0)
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return history.history['loss'][0] if 'loss' in history.history else 0
    
    def load(self, name):
        """
        Load model weights from file.
        """
        self.model.load_weights(name)
        self.update_target_model()
    
    def save(self, name):
        """
        Save model weights to file.
        """
        self.model.save_weights(name)

def train_model(episodes=1000, learning_rate=0.001, gamma=0.99, scramble_moves=5):
    """
    Train a DQN model to solve the Rubik's cube.
    
    Args:
        episodes: Number of training episodes
        learning_rate: Learning rate for the model
        gamma: Discount factor for future rewards
        scramble_moves: Number of random moves to scramble the cube
    
    Returns:
        Trained agent and training metrics
    """
    env = CubeEnvironment()
    state_size = 324  # 54 stickers * 6 colors (one-hot)
    action_size = env.actions
    agent = DQNAgent(state_size, action_size)
    
    # Set hyperparameters
    agent.learning_rate = learning_rate
    agent.gamma = gamma
    
    batch_size = 32
    rewards = []
    losses = []
    
    for e in range(episodes):
        # Reset the environment
        state = env.reset()
        
        # Scramble the cube
        env.scramble(scramble_moves)
        state = env._get_observation()
        
        # Reset accumulated reward for this episode
        total_reward = 0
        avg_loss = 0
        steps = 0
        
        while not env.done and steps < 50:
            # Choose an action
            action = agent.act(state)
            
            # Take the action
            next_state, reward, done, _ = env.step(action)
            
            # Remember the experience
            agent.remember(state, action, reward, next_state, done)
            
            # Update state and rewards
            state = next_state
            total_reward += reward
            steps += 1
            
            # Train the model
            loss = agent.replay(batch_size)
            if loss > 0:
                avg_loss += loss
        
        # Update target model periodically
        if e % 10 == 0:
            agent.update_target_model()
        
        # Calculate average loss
        avg_loss = avg_loss / steps if steps > 0 else 0
        
        # Store metrics
        rewards.append(total_reward)
        losses.append(avg_loss)
        
        # Print progress
        if e % 10 == 0:
            print(f"Episode: {e}/{episodes}, Epsilon: {agent.epsilon:.2f}, "
                  f"Reward: {total_reward:.2f}, Avg Loss: {avg_loss:.6f}")
        
        # Save model periodically
        if e % 100 == 0:
            agent.save(f"models/cube_model_{e}.h5")
    
    # Save final model
    agent.save("models/cube_model_final.h5")
    
    # Return metrics
    metrics = {
        'rewards': rewards,
        'losses': losses,
        'episodes': episodes,
        'epsilon': agent.epsilon
    }
    
    return agent, metrics

def solve_cube(cube_state, model_path, max_moves=50):
    """
    Attempt to solve a Rubik's cube from a given state using a trained model.
    
    Args:
        cube_state: Dictionary with the cube state
        model_path: Path to the trained model
        max_moves: Maximum number of moves to try
    
    Returns:
        Dictionary with solution and final state
    """
    # Create environment and set state
    env = CubeEnvironment()
    env.set_state(cube_state)
    
    # Create agent and load model
    state_size = 324
    action_size = env.actions
    agent = DQNAgent(state_size, action_size)
    agent.load(model_path)
    
    # Set epsilon to 0 for greedy actions
    agent.epsilon = 0
    
    # Try to solve the cube
    state = env._get_observation()
    solved = False
    solution = []
    
    for _ in range(max_moves):
        # Choose best action
        action = agent.act(state, test=True)
        
        # Take action
        next_state, _, done, _ = env.step(action)
        
        # Add to solution
        solution.append(env.action_to_string(action))
        
        # Update state
        state = next_state
        
        # Check if solved
        if done:
            solved = True
            break
    
    return {
        'solution': solution,
        'solved': solved,
        'final_state': env.get_state_dict()
    }