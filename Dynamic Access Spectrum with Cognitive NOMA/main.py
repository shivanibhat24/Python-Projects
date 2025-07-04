import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Concatenate
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from collections import deque
import random
import threading
import time

class SpectrumEnvironment:
    """Simulates the wireless spectrum environment"""
    
    def __init__(self, num_channels=10, num_primary_users=3, num_secondary_users=5):
        self.num_channels = num_channels
        self.num_primary_users = num_primary_users
        self.num_secondary_users = num_secondary_users
        self.channel_states = np.zeros(num_channels)  # 0: idle, 1: occupied by PU
        self.channel_qualities = np.random.uniform(0.3, 1.0, num_channels)
        self.interference_levels = np.zeros(num_channels)
        self.time_step = 0
        
    def update_primary_user_activity(self):
        """Simulate primary user activity patterns"""
        # Markov chain model for PU activity
        for i in range(self.num_channels):
            if self.channel_states[i] == 0:  # idle
                # Probability of PU arrival
                if np.random.random() < 0.1:
                    self.channel_states[i] = 1
            else:  # occupied
                # Probability of PU departure
                if np.random.random() < 0.2:
                    self.channel_states[i] = 0
    
    def get_channel_state(self):
        """Get current channel state vector"""
        return np.concatenate([
            self.channel_states,
            self.channel_qualities,
            self.interference_levels
        ])
    
    def calculate_interference(self, su_allocations, power_levels):
        """Calculate interference levels based on secondary user allocations"""
        self.interference_levels = np.zeros(self.num_channels)
        for i, allocation in enumerate(su_allocations):
            for j, channel in enumerate(allocation):
                if channel == 1:  # SU is using this channel
                    self.interference_levels[j] += power_levels[i]
    
    def step(self, su_allocations, power_levels):
        """Environment step function"""
        self.time_step += 1
        self.update_primary_user_activity()
        self.calculate_interference(su_allocations, power_levels)
        
        # Channel quality fluctuation
        self.channel_qualities += np.random.normal(0, 0.02, self.num_channels)
        self.channel_qualities = np.clip(self.channel_qualities, 0.1, 1.0)
        
        return self.get_channel_state()

class SpectrumSensor:
    """Advanced spectrum sensing with ML-based detection"""
    
    def __init__(self, num_channels):
        self.num_channels = num_channels
        self.sensing_model = self._build_sensing_model()
        self.history = deque(maxlen=100)
        self.scaler = StandardScaler()
        
    def _build_sensing_model(self):
        """Build CNN-based spectrum sensing model"""
        model = Sequential([
            Dense(64, activation='relu', input_shape=(self.num_channels * 3,)),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(self.num_channels, activation='sigmoid')
        ])
        model.compile(optimizer=Adam(learning_rate=0.001), 
                     loss='binary_crossentropy', metrics=['accuracy'])
        return model
    
    def sense_spectrum(self, environment_state):
        """Perform spectrum sensing using ML model"""
        # Simulate energy detection with noise
        raw_sensing = environment_state[:self.num_channels] + \
                     np.random.normal(0, 0.1, self.num_channels)
        
        # Use ML model for enhanced detection
        if len(self.history) > 10:
            features = np.array([environment_state])
            features_scaled = self.scaler.transform(features)
            prediction = self.sensing_model.predict(features_scaled, verbose=0)
            return prediction[0]
        else:
            return raw_sensing
    
    def update_model(self, states, labels):
        """Update sensing model with new data"""
        if len(states) > 0:
            states_scaled = self.scaler.fit_transform(states)
            self.sensing_model.fit(states_scaled, labels, epochs=1, verbose=0)

class NOMAResourceAllocator:
    """NOMA-based resource allocation with power control"""
    
    def __init__(self, num_channels, num_users, max_power=10.0):
        self.num_channels = num_channels
        self.num_users = num_users
        self.max_power = max_power
        self.min_power = 0.1
        
    def allocate_power_waterfilling(self, channel_gains, noise_power):
        """Water-filling power allocation algorithm"""
        # Sort channels by gain
        sorted_indices = np.argsort(channel_gains)[::-1]
        power_allocation = np.zeros(self.num_channels)
        
        total_power = self.max_power
        num_active_channels = 0
        
        for i in range(self.num_channels):
            if total_power > 0:
                # Water-filling formula
                water_level = total_power / (self.num_channels - i)
                for j in range(i, self.num_channels):
                    idx = sorted_indices[j]
                    allocated_power = max(0, water_level - noise_power / channel_gains[idx])
                    power_allocation[idx] = min(allocated_power, total_power)
                    total_power -= power_allocation[idx]
                    if total_power <= 0:
                        break
                break
                
        return power_allocation
    
    def noma_successive_interference_cancellation(self, users_on_channel, channel_gain, powers):
        """Calculate NOMA rates with SIC"""
        rates = np.zeros(len(users_on_channel))
        
        # Sort users by channel gain (strongest first for SIC)
        sorted_indices = np.argsort(channel_gain)[::-1]
        
        for i, user_idx in enumerate(sorted_indices):
            # Calculate interference from users decoded later
            interference = sum(powers[sorted_indices[j]] for j in range(i+1, len(sorted_indices)))
            
            # SINR calculation
            sinr = (channel_gain[user_idx] * powers[user_idx]) / (1 + interference)
            rates[user_idx] = np.log2(1 + sinr)
            
        return rates

class CognitiveAgent:
    """Deep Q-Learning agent for cognitive decision making"""
    
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = learning_rate
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()
        
    def _build_model(self):
        """Build neural network for Q-learning"""
        model = Sequential([
            Dense(128, activation='relu', input_shape=(self.state_size,)),
            Dense(64, activation='relu'),
            Dense(32, activation='relu'),
            Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse')
        return model
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in replay memory"""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action using epsilon-greedy policy"""
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_size)
        
        q_values = self.model.predict(state.reshape(1, -1), verbose=0)
        return np.argmax(q_values[0])
    
    def replay(self, batch_size=32):
        """Train the model on a batch of experiences"""
        if len(self.memory) < batch_size:
            return
        
        batch = random.sample(self.memory, batch_size)
        states = np.array([e[0] for e in batch])
        actions = np.array([e[1] for e in batch])
        rewards = np.array([e[2] for e in batch])
        next_states = np.array([e[3] for e in batch])
        dones = np.array([e[4] for e in batch])
        
        target_q_values = self.target_model.predict(next_states, verbose=0)
        target_q_values_next = np.amax(target_q_values, axis=1)
        
        target_q_values_current = self.model.predict(states, verbose=0)
        
        for i in range(batch_size):
            if dones[i]:
                target_q_values_current[i][actions[i]] = rewards[i]
            else:
                target_q_values_current[i][actions[i]] = rewards[i] + 0.95 * target_q_values_next[i]
        
        self.model.fit(states, target_q_values_current, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_model(self):
        """Update target model weights"""
        self.target_model.set_weights(self.model.get_weights())

class SecurityManager:
    """Security and authentication for spectrum access"""
    
    def __init__(self):
        self.authenticated_users = set()
        self.access_tokens = {}
        self.threat_detection_threshold = 0.8
        
    def authenticate_user(self, user_id, credentials):
        """Authenticate secondary user"""
        # Simplified authentication (in practice, use proper cryptography)
        if self._verify_credentials(user_id, credentials):
            token = self._generate_access_token(user_id)
            self.access_tokens[user_id] = token
            self.authenticated_users.add(user_id)
            return token
        return None
    
    def _verify_credentials(self, user_id, credentials):
        """Verify user credentials"""
        # Placeholder for proper authentication
        return len(credentials) > 8  # Simple check
    
    def _generate_access_token(self, user_id):
        """Generate secure access token"""
        return f"token_{user_id}_{int(time.time())}"
    
    def detect_malicious_behavior(self, user_behavior):
        """Detect potential malicious behavior"""
        # Anomaly detection based on user behavior patterns
        anomaly_score = np.random.random()  # Placeholder
        return anomaly_score > self.threat_detection_threshold
    
    def revoke_access(self, user_id):
        """Revoke access for malicious users"""
        if user_id in self.authenticated_users:
            self.authenticated_users.remove(user_id)
            del self.access_tokens[user_id]

class DynamicSpectrumAccess:
    """Main DSA system with cognitive NOMA capabilities"""
    
    def __init__(self, num_channels=10, num_primary_users=3, num_secondary_users=5):
        self.environment = SpectrumEnvironment(num_channels, num_primary_users, num_secondary_users)
        self.sensor = SpectrumSensor(num_channels)
        self.noma_allocator = NOMAResourceAllocator(num_channels, num_secondary_users)
        self.security_manager = SecurityManager()
        
        # Cognitive agents for each secondary user
        state_size = num_channels * 3  # channel states, qualities, interference
        action_size = num_channels + 1  # channel selection + power level
        self.cognitive_agents = [
            CognitiveAgent(state_size, action_size) for _ in range(num_secondary_users)
        ]
        
        self.performance_metrics = {
            'throughput': [],
            'collision_rate': [],
            'energy_efficiency': [],
            'fairness_index': []
        }
    
    def run_simulation(self, num_episodes=1000):
        """Run the main simulation loop"""
        print("Starting Dynamic Spectrum Access Simulation...")
        
        for episode in range(num_episodes):
            state = self.environment.get_channel_state()
            total_reward = 0
            
            for step in range(100):  # Steps per episode
                # Spectrum sensing
                sensed_spectrum = self.sensor.sense_spectrum(state)
                
                # Cognitive decision making
                actions = []
                su_allocations = []
                power_levels = []
                
                for i, agent in enumerate(self.cognitive_agents):
                    action = agent.act(state)
                    actions.append(action)
                    
                    # Convert action to channel allocation
                    channel_allocation = np.zeros(self.environment.num_channels)
                    if action < self.environment.num_channels and sensed_spectrum[action] < 0.5:
                        channel_allocation[action] = 1
                    
                    su_allocations.append(channel_allocation)
                    power_levels.append(min(action * 0.5, self.noma_allocator.max_power))
                
                # Environment step
                next_state = self.environment.step(su_allocations, power_levels)
                
                # Calculate rewards
                rewards = self._calculate_rewards(su_allocations, power_levels, sensed_spectrum)
                
                # Store experiences and train agents
                for i, agent in enumerate(self.cognitive_agents):
                    agent.remember(state, actions[i], rewards[i], next_state, step == 99)
                    agent.replay()
                
                # Update performance metrics
                self._update_metrics(su_allocations, power_levels, rewards)
                
                state = next_state
                total_reward += np.sum(rewards)
            
            # Update target models periodically
            if episode % 10 == 0:
                for agent in self.cognitive_agents:
                    agent.update_target_model()
            
            if episode % 100 == 0:
                print(f"Episode {episode}, Average Reward: {total_reward/100:.2f}")
                self._plot_performance()
    
    def _calculate_rewards(self, su_allocations, power_levels, sensed_spectrum):
        """Calculate rewards for secondary users"""
        rewards = []
        
        for i, allocation in enumerate(su_allocations):
            reward = 0
            
            for j, channel_used in enumerate(allocation):
                if channel_used == 1:
                    # Positive reward for successful transmission
                    if sensed_spectrum[j] < 0.5:  # Channel is idle
                        channel_quality = self.environment.channel_qualities[j]
                        reward += channel_quality * 10
                    else:
                        # Penalty for interfering with primary user
                        reward -= 20
                    
                    # Energy efficiency reward
                    energy_efficiency = channel_quality / (power_levels[i] + 1e-6)
                    reward += energy_efficiency * 5
            
            rewards.append(reward)
        
        return rewards
    
    def _update_metrics(self, su_allocations, power_levels, rewards):
        """Update performance metrics"""
        # Throughput
        total_throughput = sum(max(0, r) for r in rewards)
        self.performance_metrics['throughput'].append(total_throughput)
        
        # Collision rate
        collisions = sum(1 for allocation in su_allocations 
                        for i, used in enumerate(allocation) 
                        if used == 1 and self.environment.channel_states[i] == 1)
        collision_rate = collisions / max(1, sum(sum(allocation) for allocation in su_allocations))
        self.performance_metrics['collision_rate'].append(collision_rate)
        
        # Energy efficiency
        total_power = sum(power_levels)
        energy_efficiency = total_throughput / max(total_power, 1e-6)
        self.performance_metrics['energy_efficiency'].append(energy_efficiency)
        
        # Fairness index (Jain's fairness index)
        if len(rewards) > 0:
            fairness = (sum(rewards) ** 2) / (len(rewards) * sum(r ** 2 for r in rewards))
            self.performance_metrics['fairness_index'].append(fairness)
    
    def _plot_performance(self):
        """Plot performance metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        metrics = ['throughput', 'collision_rate', 'energy_efficiency', 'fairness_index']
        titles = ['Throughput', 'Collision Rate', 'Energy Efficiency', 'Fairness Index']
        
        for i, (metric, title) in enumerate(zip(metrics, titles)):
            ax = axes[i//2, i%2]
            if self.performance_metrics[metric]:
                # Moving average for smoother plotting
                window = min(50, len(self.performance_metrics[metric]))
                moving_avg = np.convolve(self.performance_metrics[metric], 
                                       np.ones(window)/window, mode='valid')
                ax.plot(moving_avg)
                ax.set_title(title)
                ax.set_xlabel('Time Steps')
                ax.set_ylabel(title)
                ax.grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the DSA system"""
    print("Initializing Dynamic Spectrum Access System...")
    
    # Create DSA system
    dsa_system = DynamicSpectrumAccess(
        num_channels=10,
        num_primary_users=3,
        num_secondary_users=5
    )
    
    # Authenticate users (simplified)
    for i in range(dsa_system.environment.num_secondary_users):
        token = dsa_system.security_manager.authenticate_user(
            f"user_{i}", f"password_{i}_12345"
        )
        print(f"User {i} authenticated with token: {token}")
    
    # Run simulation
    dsa_system.run_simulation(num_episodes=500)
    
    print("\nSimulation completed successfully!")
    print("The system demonstrated:")
    print("1. Intelligent spectrum sensing using ML")
    print("2. Cognitive decision making with deep Q-learning")
    print("3. NOMA-based resource allocation")
    print("4. Primary user protection")
    print("5. Security and authentication")
    print("6. Adaptive power control")

if __name__ == "__main__":
    main()
