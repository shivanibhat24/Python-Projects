import numpy as np
import tensorflow as tf
import gym
from gym import spaces
from collections import deque
import random
import matplotlib.pyplot as plt

# Define the Power Converter Environment
class PowerConverterEnv(gym.Env):
    """
    A power converter environment for reinforcement learning.
    
    This environment simulates a DC-DC buck converter where the agent
    controls the duty cycle to maintain a desired output voltage
    while maximizing efficiency.
    """
    def __init__(self):
        super(PowerConverterEnv, self).__init__()
        
        # Constants for buck converter
        self.Vin = 24.0        # Input voltage (V)
        self.Vref = 12.0       # Reference output voltage (V)
        self.R = 10.0          # Load resistance (Ohm)
        self.L = 100e-6        # Inductor (H)
        self.C = 100e-6        # Capacitor (F)
        self.fs = 100e-3       # Switching frequency (kHz)
        self.dt = 1.0/self.fs  # Time step
        
        # Circuit state
        self.Vout = 0.0        # Output voltage
        self.IL = 0.0          # Inductor current
        
        # Parasitic components (for efficiency calculation)
        self.Ron = 0.1         # Switch on-resistance (Ohm)
        self.Rdiode = 0.05     # Diode forward resistance (Ohm)
        self.Vdiode = 0.7      # Diode forward voltage drop (V)
        self.Rind = 0.1        # Inductor winding resistance (Ohm)
        
        # Action: duty cycle control (between 0.0 and 0.95)
        self.action_space = spaces.Box(
            low=np.array([0.0]), high=np.array([0.95]), dtype=np.float32
        )
        
        # Observation: [Vout, IL, Vin, R_load]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0]), 
            high=np.array([50.0, 20.0, 50.0, 100.0]), 
            dtype=np.float32
        )
        
        # Simulation parameters
        self.time_steps = 0
        self.max_steps = 200
        self.load_change_interval = 50
        
    def reset(self):
        """Reset the environment to initial state."""
        self.Vout = 0.0
        self.IL = 0.0
        self.time_steps = 0
        
        # Randomize input voltage with some variance
        self.Vin = np.random.uniform(22.0, 26.0)
        
        # Start with random load in range
        self.R = np.random.uniform(5.0, 15.0)
        
        return np.array([self.Vout, self.IL, self.Vin, self.R])
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action: Duty cycle [0, 0.95]
            
        Returns:
            observation, reward, done, info
        """
        # Extract duty cycle from action
        duty_cycle = float(action[0])
        duty_cycle = np.clip(duty_cycle, 0.0, 0.95)
        
        # Possibly change load resistance to simulate dynamic conditions
        if self.time_steps % self.load_change_interval == 0 and self.time_steps > 0:
            self.R = np.random.uniform(5.0, 15.0)
        
        # Update converter state using averaged model
        # Inductor current dynamics: L * dIL/dt = Vin*duty - Vout
        dIL = (self.Vin * duty_cycle - self.Vout) / self.L
        self.IL = self.IL + dIL * self.dt
        
        # Capacitor voltage dynamics: C * dVout/dt = IL - Vout/R
        dVout = (self.IL - self.Vout / self.R) / self.C
        self.Vout = self.Vout + dVout * self.dt
        
        # Ensure physical limits
        self.IL = max(0.0, self.IL)
        self.Vout = max(0.0, self.Vout)
        
        # Calculate power losses
        p_switch_loss = duty_cycle * self.IL**2 * self.Ron
        p_diode_loss = (1-duty_cycle) * (self.IL**2 * self.Rdiode + self.IL * self.Vdiode)
        p_inductor_loss = self.IL**2 * self.Rind
        p_total_loss = p_switch_loss + p_diode_loss + p_inductor_loss
        
        # Output power
        p_out = self.Vout**2 / self.R
        
        # Input power
        p_in = self.Vin * duty_cycle * self.IL
        
        # Efficiency
        efficiency = p_out / (p_out + p_total_loss) if (p_out + p_total_loss) > 0 else 0
        
        # Calculate voltage regulation error
        v_error = abs(self.Vout - self.Vref) / self.Vref
        
        # Define reward function: balance between voltage regulation and efficiency
        reward = 1.0 - v_error - 0.1 * (1.0 - efficiency)
        
        # Penalize if voltage is too far from reference
        if v_error > 0.2:  # More than 20% error
            reward -= 5
        
        # Check if episode is done
        self.time_steps += 1
        done = (self.time_steps >= self.max_steps) or (v_error > 0.5)
        
        # Return observation, reward, done flag, and info
        observation = np.array([self.Vout, self.IL, self.Vin, self.R])
        info = {
            'voltage_error': v_error,
            'efficiency': efficiency,
            'duty_cycle': duty_cycle
        }
        
        return observation, reward, done, info

# Deep Q-Network Agent
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    def _build_model(self):
        """Build a neural network model for DQN."""
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        """Update target model with weights from main model."""
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Choose action based on epsilon-greedy policy."""
        if np.random.rand() <= self.epsilon:
            # We'll choose a discrete action from a set of 20 possible duty cycles
            return np.array([np.random.uniform(0, 0.95)])
        
        act_values = self.model.predict(state.reshape(1, -1))
        # Return continuous action value (duty cycle)
        return np.array([np.clip(act_values[0][0], 0, 0.95)])

    def replay(self, batch_size):
        """Train the network using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(
                    self.target_model.predict(next_state.reshape(1, -1))[0]
                )
            
            target_f = self.model.predict(state.reshape(1, -1))
            target_f[0][0] = target
            
            self.model.fit(state.reshape(1, -1), target_f, epochs=1, verbose=0)
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Continuous Action DDPG Agent (more suitable for continuous control)
class DDPGAgent:
    def __init__(self, state_size, action_size, action_high):
        self.state_size = state_size
        self.action_size = action_size
        self.action_high = action_high  # Maximum action value
        self.memory = deque(maxlen=5000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0   # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.tau = 0.001     # target network update rate
        
        # Build actor and critic networks
        self.actor = self._build_actor()
        self.critic = self._build_critic()
        self.target_actor = self._build_actor()
        self.target_critic = self._build_critic()
        
        # Initialize target network weights with main network weights
        self.target_actor.set_weights(self.actor.get_weights())
        self.target_critic.set_weights(self.critic.get_weights())
        
        # Optimizer for both networks
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    
    def _build_actor(self):
        """Build actor (policy) network."""
        actor = tf.keras.Sequential([
            tf.keras.layers.Dense(24, input_dim=self.state_size, activation='relu'),
            tf.keras.layers.Dense(24, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='sigmoid')
        ])
        return actor
    
    def _build_critic(self):
        """Build critic (value) network."""
        state_input = tf.keras.layers.Input(shape=(self.state_size,))
        action_input = tf.keras.layers.Input(shape=(self.action_size,))
        
        # Process state path
        state_h1 = tf.keras.layers.Dense(24, activation='relu')(state_input)
        
        # Merge state and action paths
        merged = tf.keras.layers.Concatenate()([state_h1, action_input])
        merged_h1 = tf.keras.layers.Dense(24, activation='relu')(merged)
        
        # Output single Q-value
        output = tf.keras.layers.Dense(1, activation='linear')(merged_h1)
        
        critic = tf.keras.Model(inputs=[state_input, action_input], outputs=output)
        return critic
    
    def remember(self, state, action, reward, next_state, done):
        """Store experience in memory."""
        self.memory.append((state, action, reward, next_state, done))
    
    def act(self, state):
        """Choose action based on current policy with noise for exploration."""
        action = self.actor(state.reshape(1, -1)).numpy()[0]
        if np.random.rand() <= self.epsilon:
            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=self.action_size)
            action = action + noise
        
        # Scale action to the appropriate range and clip
        action = action * self.action_high
        return np.clip(action, 0, self.action_high)
    
    def update_target_models(self):
        """Soft update target networks."""
        # Update target actor
        actor_weights = self.actor.get_weights()
        target_actor_weights = self.target_actor.get_weights()
        for i in range(len(actor_weights)):
            target_actor_weights[i] = self.tau * actor_weights[i] + (1 - self.tau) * target_actor_weights[i]
        self.target_actor.set_weights(target_actor_weights)
        
        # Update target critic
        critic_weights = self.critic.get_weights()
        target_critic_weights = self.target_critic.get_weights()
        for i in range(len(critic_weights)):
            target_critic_weights[i] = self.tau * critic_weights[i] + (1 - self.tau) * target_critic_weights[i]
        self.target_critic.set_weights(target_critic_weights)
    
    def train(self, batch_size=64):
        """Train actor and critic networks using experience replay."""
        if len(self.memory) < batch_size:
            return
        
        # Sample random minibatch from memory
        minibatch = random.sample(self.memory, batch_size)
        states = np.array([experience[0] for experience in minibatch])
        actions = np.array([experience[1] for experience in minibatch])
        rewards = np.array([experience[2] for experience in minibatch])
        next_states = np.array([experience[3] for experience in minibatch])
        dones = np.array([experience[4] for experience in minibatch])
        
        # Training critic network
        with tf.GradientTape() as tape:
            target_actions = self.target_actor(next_states)
            target_q_values = self.target_critic([next_states, target_actions])
            target_q = rewards + (1 - dones) * self.gamma * target_q_values
            current_q = self.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(target_q - current_q))
        
        critic_gradients = tape.gradient(critic_loss, self.critic.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_gradients, self.critic.trainable_variables))
        
        # Training actor network
        with tf.GradientTape() as tape:
            actions_pred = self.actor(states)
            actor_loss = -tf.reduce_mean(self.critic([states, actions_pred]))
        
        actor_gradients = tape.gradient(actor_loss, self.actor.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_gradients, self.actor.trainable_variables))
        
        # Update target networks
        self.update_target_models()
        
        # Decay exploration rate
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

def train_ddpg():
    env = PowerConverterEnv()
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_high = env.action_space.high[0]
    
    # Create DDPG agent
    agent = DDPGAgent(state_size, action_size, action_high)
    
    # Training parameters
    batch_size = 64
    episodes = 1000
    
    # Lists to store results
    episode_rewards = []
    average_rewards = []
    voltage_errors = []
    efficiencies = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        episode_voltage_error = 0
        episode_efficiency = 0
        
        for time_step in range(env.max_steps):
            # Select action
            action = agent.act(state)
            
            # Take action and observe next state and reward
            next_state, reward, done, info = env.step(action)
            
            # Store experience in replay buffer
            agent.remember(state, action, reward, next_state, done)
            
            # Train agent
            agent.train(batch_size)
            
            # Update state and metrics
            state = next_state
            episode_reward += reward
            episode_voltage_error += info['voltage_error']
            episode_efficiency += info['efficiency']
            
            if done:
                break
        
        # Track episode metrics
        episode_rewards.append(episode_reward)
        voltage_errors.append(episode_voltage_error / (time_step + 1))
        efficiencies.append(episode_efficiency / (time_step + 1))
        
        # Calculate average reward over last 100 episodes
        if len(episode_rewards) > 100:
            avg_reward = np.mean(episode_rewards[-100:])
            average_rewards.append(avg_reward)
        else:
            average_rewards.append(np.mean(episode_rewards))
        
        # Print progress
        if episode % 10 == 0:
            print(f"Episode: {episode}, Reward: {episode_reward:.2f}, "
                  f"Avg Reward: {average_rewards[-1]:.2f}, "
                  f"Voltage Error: {voltage_errors[-1]:.4f}, "
                  f"Efficiency: {efficiencies[-1]:.4f}")
    
    return agent, episode_rewards, average_rewards, voltage_errors, efficiencies

# Testing function
def test_agent(agent, num_episodes=5):
    env = PowerConverterEnv()
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        
        # Lists to store episode data for plotting
        times = []
        voltages = []
        currents = []
        duty_cycles = []
        
        for t in range(env.max_steps):
            times.append(t * env.dt)
            
            # Select action without exploration
            agent.epsilon = 0
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store metrics
            voltages.append(state[0])  # Vout
            currents.append(state[1])  # IL
            duty_cycles.append(info['duty_cycle'])
            
            state = next_state
            total_reward += reward
            
            if done:
                break
        
        # Plot results
        plt.figure(figsize=(15, 10))
        
        plt.subplot(4, 1, 1)
        plt.plot(times, voltages)
        plt.axhline(y=env.Vref, color='r', linestyle='--')
        plt.title(f'Episode {episode} - Reward: {total_reward:.2f}')
        plt.ylabel('Output Voltage (V)')
        
        plt.subplot(4, 1, 2)
        plt.plot(times, currents)
        plt.ylabel('Inductor Current (A)')
        
        plt.subplot(4, 1, 3)
        plt.plot(times, duty_cycles)
        plt.ylabel('Duty Cycle')
        
        plt.subplot(4, 1, 4)
        plt.plot(times[:-1], np.diff(voltages))
        plt.ylabel('Voltage Ripple (V)')
        plt.xlabel('Time (s)')
        
        plt.tight_layout()
        plt.show()
        
        print(f"Episode {episode} - Total Reward: {total_reward:.2f}")
        print(f"Final Voltage: {voltages[-1]:.2f}V (Reference: {env.Vref:.2f}V)")
        print(f"Voltage Error: {abs(voltages[-1] - env.Vref)/env.Vref * 100:.2f}%")

# Visualization function
def plot_training_results(rewards, avg_rewards, voltage_errors, efficiencies):
    plt.figure(figsize=(15, 10))
    
    plt.subplot(3, 1, 1)
    plt.plot(rewards, label='Episode Reward')
    plt.plot(avg_rewards, label='Average Reward (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.legend()
    plt.title('Training Rewards')
    
    plt.subplot(3, 1, 2)
    plt.plot(voltage_errors)
    plt.xlabel('Episode')
    plt.ylabel('Average Voltage Error')
    plt.title('Voltage Regulation Performance')
    
    plt.subplot(3, 1, 3)
    plt.plot(efficiencies)
    plt.xlabel('Episode')
    plt.ylabel('Average Efficiency')
    plt.title('Converter Efficiency')
    
    plt.tight_layout()
    plt.show()

# Main execution
if __name__ == "__main__":
    # Train the agent
    print("Starting DDPG training...")
    agent, rewards, avg_rewards, v_errors, effs = train_ddpg()
    
    # Plot training results
    plot_training_results(rewards, avg_rewards, v_errors, effs)
    
    # Test the trained agent
    print("\nTesting the trained agent...")
    test_agent(agent)
    
    # Save the trained model
    agent.actor.save("dc_dc_converter_actor.h5")
    agent.critic.save("dc_dc_converter_critic.h5")
    print("Models saved.")
