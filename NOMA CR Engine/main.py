import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, differential_evolution
from scipy.stats import expon, rayleigh
import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

@dataclass
class SystemParameters:
    """System configuration parameters"""
    num_primary_users: int = 2
    num_secondary_users: int = 6
    num_channels: int = 4
    max_power_budget: float = 1.0  # Watts
    noise_power: float = 1e-9  # Watts
    primary_interference_threshold: float = 1e-8  # Watts
    channel_bandwidth: float = 1e6  # Hz
    path_loss_exponent: float = 3.0
    
    # NOMA parameters
    min_power_allocation: float = 0.1
    max_power_allocation: float = 0.9
    sic_error_threshold: float = 0.01
    
    # Cooperative parameters
    relay_selection_threshold: float = 0.5
    cooperation_probability: float = 0.8
    
    # Optimization parameters
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

class ChannelModel:
    """Channel modeling for wireless communications"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.reset_channels()
    
    def reset_channels(self):
        """Initialize channel coefficients"""
        # Primary user channels
        self.h_primary = np.random.rayleigh(1.0, (self.params.num_primary_users, self.params.num_channels))
        
        # Secondary user channels (direct links)
        self.h_secondary = np.random.rayleigh(1.0, (self.params.num_secondary_users, self.params.num_channels))
        
        # Cooperative relay channels
        self.h_relay = np.random.rayleigh(1.0, (self.params.num_secondary_users, self.params.num_secondary_users))
        
        # Interference channels (secondary to primary)
        self.h_interference = np.random.rayleigh(0.5, (self.params.num_secondary_users, self.params.num_primary_users))
    
    def update_channels(self, coherence_blocks: int = 1):
        """Update channel coefficients for time-varying channels"""
        correlation = 0.9 ** coherence_blocks
        
        # Update with temporal correlation
        self.h_primary = correlation * self.h_primary + np.sqrt(1 - correlation**2) * np.random.rayleigh(1.0, self.h_primary.shape)
        self.h_secondary = correlation * self.h_secondary + np.sqrt(1 - correlation**2) * np.random.rayleigh(1.0, self.h_secondary.shape)
        self.h_relay = correlation * self.h_relay + np.sqrt(1 - correlation**2) * np.random.rayleigh(1.0, self.h_relay.shape)
        self.h_interference = correlation * self.h_interference + np.sqrt(1 - correlation**2) * np.random.rayleigh(0.5, self.h_interference.shape)

class PrimaryUserModel:
    """Primary user activity modeling"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.activity_prob = 0.3  # Probability of primary user being active
        self.state = np.random.binomial(1, self.activity_prob, params.num_primary_users)
    
    def update_activity(self):
        """Update primary user activity based on Markov model"""
        transition_prob = 0.1  # Probability of state change
        
        for i in range(self.params.num_primary_users):
            if np.random.random() < transition_prob:
                self.state[i] = 1 - self.state[i]
    
    def get_active_users(self) -> np.ndarray:
        """Return indices of active primary users"""
        return np.where(self.state == 1)[0]

class NOMAModel:
    """NOMA power allocation and SIC modeling"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
    
    def calculate_power_coefficients(self, channel_gains: np.ndarray, total_power: float) -> np.ndarray:
        """Calculate NOMA power allocation coefficients based on channel gains"""
        # Sort users by channel gain (weakest first for NOMA)
        sorted_indices = np.argsort(channel_gains)
        num_users = len(channel_gains)
        
        # Allocate more power to weaker users
        power_coeffs = np.zeros(num_users)
        remaining_power = total_power
        
        for i, idx in enumerate(sorted_indices):
            if i == num_users - 1:  # Last user gets remaining power
                power_coeffs[idx] = remaining_power
            else:
                # Allocate power inversely proportional to channel gain
                power_fraction = (num_users - i) / sum(range(1, num_users + 1))
                power_coeffs[idx] = power_fraction * total_power * 0.8  # Leave some for last user
                remaining_power -= power_coeffs[idx]
        
        # Ensure minimum power allocation
        power_coeffs = np.maximum(power_coeffs, self.params.min_power_allocation * total_power / num_users)
        power_coeffs = power_coeffs / np.sum(power_coeffs) * total_power  # Normalize
        
        return power_coeffs
    
    def calculate_sinr(self, user_idx: int, power_coeffs: np.ndarray, channel_gains: np.ndarray) -> float:
        """Calculate SINR for a NOMA user considering SIC"""
        signal_power = power_coeffs[user_idx] * channel_gains[user_idx]**2
        
        # Interference from users with higher power allocation (not yet cancelled)
        interference = 0
        for j in range(len(power_coeffs)):
            if j != user_idx and power_coeffs[j] > power_coeffs[user_idx]:
                interference += power_coeffs[j] * channel_gains[user_idx]**2
        
        # Add noise
        sinr = signal_power / (interference + self.params.noise_power)
        return sinr
    
    def sic_success_probability(self, sinr: float) -> float:
        """Calculate probability of successful SIC based on SINR"""
        return 1 - np.exp(-sinr / self.params.sic_error_threshold)

class CooperativeModel:
    """Cooperative communication modeling"""
    
    def __init__(self, params: SystemParameters, channel_model: ChannelModel):
        self.params = params
        self.channel_model = channel_model
    
    def select_relays(self, source_user: int, channel_idx: int) -> List[int]:
        """Select best relays for cooperative transmission"""
        potential_relays = []
        
        for relay in range(self.params.num_secondary_users):
            if relay != source_user:
                # Check if relay has good channel to source and destination
                relay_to_source = self.channel_model.h_relay[relay, source_user]
                relay_channel_gain = self.channel_model.h_secondary[relay, channel_idx]
                
                if (relay_to_source > self.params.relay_selection_threshold and 
                    relay_channel_gain > self.params.relay_selection_threshold):
                    potential_relays.append(relay)
        
        # Select best relay based on combined channel gain
        if potential_relays:
            best_relay = max(potential_relays, 
                           key=lambda r: self.channel_model.h_relay[r, source_user] * 
                                       self.channel_model.h_secondary[r, channel_idx])
            return [best_relay]
        
        return []
    
    def calculate_cooperative_rate(self, source_user: int, relay_user: int, 
                                 channel_idx: int, power_allocation: float) -> float:
        """Calculate achievable rate with cooperative transmission"""
        # Direct link rate
        direct_sinr = (power_allocation * self.channel_model.h_secondary[source_user, channel_idx]**2 / 
                      self.params.noise_power)
        direct_rate = np.log2(1 + direct_sinr)
        
        # Relay link rate (decode-and-forward)
        relay_sinr = (power_allocation * self.channel_model.h_relay[relay_user, source_user]**2 / 
                     self.params.noise_power)
        relay_rate = np.log2(1 + relay_sinr)
        
        # Cooperative rate is limited by bottleneck
        cooperative_rate = min(direct_rate, relay_rate)
        
        return max(direct_rate, cooperative_rate)

class PerformanceMetrics:
    """Performance evaluation metrics"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
    
    def calculate_spectral_efficiency(self, rates: np.ndarray) -> float:
        """Calculate spectral efficiency in bits/s/Hz"""
        return np.sum(rates) / self.params.channel_bandwidth
    
    def calculate_energy_efficiency(self, rates: np.ndarray, power_consumption: float) -> float:
        """Calculate energy efficiency in bits/J"""
        total_rate = np.sum(rates)
        return total_rate / power_consumption if power_consumption > 0 else 0
    
    def calculate_outage_probability(self, achieved_rates: np.ndarray, 
                                   required_rates: np.ndarray) -> float:
        """Calculate outage probability"""
        outages = achieved_rates < required_rates
        return np.mean(outages)
    
    def calculate_throughput(self, rates: np.ndarray, success_probabilities: np.ndarray) -> float:
        """Calculate effective throughput considering transmission success"""
        return np.sum(rates * success_probabilities)
    
    def calculate_latency(self, num_hops: int, processing_delay: float = 1e-3) -> float:
        """Calculate end-to-end latency"""
        transmission_delay = num_hops * processing_delay
        return transmission_delay

class OptimizationEngine:
    """Multi-objective optimization for resource allocation"""
    
    def __init__(self, params: SystemParameters, channel_model: ChannelModel, 
                 primary_model: PrimaryUserModel, noma_model: NOMAModel, 
                 cooperative_model: CooperativeModel, metrics: PerformanceMetrics):
        self.params = params
        self.channel_model = channel_model
        self.primary_model = primary_model
        self.noma_model = noma_model
        self.cooperative_model = cooperative_model
        self.metrics = metrics
    
    def objective_function(self, x: np.ndarray) -> float:
        """Multi-objective optimization function"""
        # Decode optimization variables
        power_allocation = x[:self.params.num_secondary_users]
        user_channel_assignment = x[self.params.num_secondary_users:2*self.params.num_secondary_users]
        cooperation_flags = x[2*self.params.num_secondary_users:]
        
        # Normalize power allocation
        power_allocation = power_allocation / np.sum(power_allocation) * self.params.max_power_budget
        
        # Calculate performance metrics
        rates = []
        interference_to_primary = 0
        
        active_primaries = self.primary_model.get_active_users()
        
        for user in range(self.params.num_secondary_users):
            channel_idx = int(user_channel_assignment[user] * self.params.num_channels) % self.params.num_channels
            
            # Check if channel is occupied by primary user
            if channel_idx < len(active_primaries) and active_primaries[channel_idx] if len(active_primaries) > channel_idx else False:
                rates.append(0)  # Cannot use occupied channel
                continue
            
            # Calculate interference to primary users
            for primary_idx in active_primaries:
                if primary_idx < self.params.num_primary_users:
                    interference_to_primary += (power_allocation[user] * 
                                               self.channel_model.h_interference[user, primary_idx]**2)
            
            # Calculate achievable rate
            if cooperation_flags[user] > 0.5:  # Use cooperation
                relays = self.cooperative_model.select_relays(user, channel_idx)
                if relays:
                    rate = self.cooperative_model.calculate_cooperative_rate(
                        user, relays[0], channel_idx, power_allocation[user])
                else:
                    # Fallback to direct transmission
                    sinr = (power_allocation[user] * 
                           self.channel_model.h_secondary[user, channel_idx]**2 / 
                           self.params.noise_power)
                    rate = np.log2(1 + sinr)
            else:
                # Direct transmission
                sinr = (power_allocation[user] * 
                       self.channel_model.h_secondary[user, channel_idx]**2 / 
                       self.params.noise_power)
                rate = np.log2(1 + sinr)
            
            rates.append(rate)
        
        rates = np.array(rates)
        
        # Calculate objectives
        spectral_efficiency = self.metrics.calculate_spectral_efficiency(rates)
        energy_efficiency = self.metrics.calculate_energy_efficiency(rates, np.sum(power_allocation))
        throughput = self.metrics.calculate_throughput(rates, np.ones(len(rates)))
        
        # Penalty for violating primary user interference constraint
        interference_penalty = max(0, interference_to_primary - self.params.primary_interference_threshold) * 1e6
        
        # Multi-objective function (maximize spectral efficiency and throughput, minimize interference)
        objective = -(0.5 * spectral_efficiency + 0.3 * throughput + 0.2 * energy_efficiency) + interference_penalty
        
        return objective
    
    def optimize_resources(self) -> Dict:
        """Perform resource optimization"""
        # Optimization variables: [power_allocation, user_channel_assignment, cooperation_flags]
        num_vars = 3 * self.params.num_secondary_users
        
        # Bounds for optimization variables
        bounds = []
        
        # Power allocation bounds
        for _ in range(self.params.num_secondary_users):
            bounds.append((self.params.min_power_allocation, self.params.max_power_allocation))
        
        # User-channel assignment bounds (normalized)
        for _ in range(self.params.num_secondary_users):
            bounds.append((0, 1))
        
        # Cooperation flags bounds
        for _ in range(self.params.num_secondary_users):
            bounds.append((0, 1))
        
        # Use differential evolution for global optimization
        result = differential_evolution(
            self.objective_function,
            bounds,
            maxiter=self.params.max_iterations,
            popsize=15,
            seed=42
        )
        
        if result.success:
            # Decode optimal solution
            x_opt = result.x
            power_opt = x_opt[:self.params.num_secondary_users]
            power_opt = power_opt / np.sum(power_opt) * self.params.max_power_budget
            
            channel_assignment = x_opt[self.params.num_secondary_users:2*self.params.num_secondary_users]
            channel_assignment = (channel_assignment * self.params.num_channels).astype(int) % self.params.num_channels
            
            cooperation_flags = x_opt[2*self.params.num_secondary_users:] > 0.5
            
            return {
                'power_allocation': power_opt,
                'channel_assignment': channel_assignment,
                'cooperation_flags': cooperation_flags,
                'objective_value': -result.fun,
                'optimization_success': True
            }
        else:
            return {
                'optimization_success': False,
                'message': 'Optimization failed'
            }

class SimulationEngine:
    """Main simulation engine"""
    
    def __init__(self, params: SystemParameters):
        self.params = params
        self.channel_model = ChannelModel(params)
        self.primary_model = PrimaryUserModel(params)
        self.noma_model = NOMAModel(params)
        self.cooperative_model = CooperativeModel(params, self.channel_model)
        self.metrics = PerformanceMetrics(params)
        self.optimizer = OptimizationEngine(
            params, self.channel_model, self.primary_model, 
            self.noma_model, self.cooperative_model, self.metrics
        )
        
        # Results storage
        self.results = {
            'spectral_efficiency': [],
            'energy_efficiency': [],
            'throughput': [],
            'outage_probability': [],
            'latency': [],
            'power_allocation': [],
            'channel_assignment': [],
            'cooperation_usage': []
        }
    
    def run_simulation(self, num_iterations: int = 1000) -> Dict:
        """Run complete simulation"""
        print(f"Starting simulation with {num_iterations} iterations...")
        
        for iteration in range(num_iterations):
            if iteration % 100 == 0:
                print(f"Iteration {iteration}/{num_iterations}")
            
            # Update channel conditions
            self.channel_model.update_channels()
            
            # Update primary user activity
            self.primary_model.update_activity()
            
            # Perform optimization
            opt_result = self.optimizer.optimize_resources()
            
            if opt_result['optimization_success']:
                # Calculate performance metrics
                power_allocation = opt_result['power_allocation']
                channel_assignment = opt_result['channel_assignment']
                cooperation_flags = opt_result['cooperation_flags']
                
                # Calculate rates for each user
                rates = []
                for user in range(self.params.num_secondary_users):
                    channel_idx = channel_assignment[user]
                    
                    if cooperation_flags[user]:
                        relays = self.cooperative_model.select_relays(user, channel_idx)
                        if relays:
                            rate = self.cooperative_model.calculate_cooperative_rate(
                                user, relays[0], channel_idx, power_allocation[user])
                        else:
                            sinr = (power_allocation[user] * 
                                   self.channel_model.h_secondary[user, channel_idx]**2 / 
                                   self.params.noise_power)
                            rate = np.log2(1 + sinr)
                    else:
                        sinr = (power_allocation[user] * 
                               self.channel_model.h_secondary[user, channel_idx]**2 / 
                               self.params.noise_power)
                        rate = np.log2(1 + sinr)
                    
                    rates.append(rate)
                
                rates = np.array(rates)
                
                # Store results
                self.results['spectral_efficiency'].append(
                    self.metrics.calculate_spectral_efficiency(rates)
                )
                self.results['energy_efficiency'].append(
                    self.metrics.calculate_energy_efficiency(rates, np.sum(power_allocation))
                )
                self.results['throughput'].append(
                    self.metrics.calculate_throughput(rates, np.ones(len(rates)))
                )
                self.results['outage_probability'].append(
                    self.metrics.calculate_outage_probability(rates, np.ones(len(rates)) * 1.0)
                )
                self.results['latency'].append(
                    self.metrics.calculate_latency(2 if np.any(cooperation_flags) else 1)
                )
                self.results['power_allocation'].append(power_allocation.copy())
                self.results['channel_assignment'].append(channel_assignment.copy())
                self.results['cooperation_usage'].append(np.mean(cooperation_flags))
        
        # Calculate statistics
        statistics = {}
        for key in ['spectral_efficiency', 'energy_efficiency', 'throughput', 
                   'outage_probability', 'latency', 'cooperation_usage']:
            if self.results[key]:
                statistics[key] = {
                    'mean': np.mean(self.results[key]),
                    'std': np.std(self.results[key]),
                    'min': np.min(self.results[key]),
                    'max': np.max(self.results[key])
                }
        
        return statistics
    
    def plot_results(self):
        """Plot simulation results"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Spectral Efficiency
        axes[0, 0].plot(self.results['spectral_efficiency'])
        axes[0, 0].set_title('Spectral Efficiency')
        axes[0, 0].set_ylabel('bits/s/Hz')
        axes[0, 0].grid(True)
        
        # Energy Efficiency
        axes[0, 1].plot(self.results['energy_efficiency'])
        axes[0, 1].set_title('Energy Efficiency')
        axes[0, 1].set_ylabel('bits/J')
        axes[0, 1].grid(True)
        
        # Throughput
        axes[0, 2].plot(self.results['throughput'])
        axes[0, 2].set_title('Throughput')
        axes[0, 2].set_ylabel('bits/s')
        axes[0, 2].grid(True)
        
        # Outage Probability
        axes[1, 0].plot(self.results['outage_probability'])
        axes[1, 0].set_title('Outage Probability')
        axes[1, 0].set_ylabel('Probability')
        axes[1, 0].grid(True)
        
        # Latency
        axes[1, 1].plot(self.results['latency'])
        axes[1, 1].set_title('Latency')
        axes[1, 1].set_ylabel('seconds')
        axes[1, 1].grid(True)
        
        # Cooperation Usage
        axes[1, 2].plot(self.results['cooperation_usage'])
        axes[1, 2].set_title('Cooperation Usage')
        axes[1, 2].set_ylabel('Fraction')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plt.show()

# Example usage and testing
if __name__ == "__main__":
    # Create system parameters
    params = SystemParameters(
        num_primary_users=2,
        num_secondary_users=4,
        num_channels=3,
        max_power_budget=2.0,
        max_iterations=50  # Reduced for faster testing
    )
    
    # Create simulation engine
    sim_engine = SimulationEngine(params)
    
    # Run simulation
    print("Running Cooperative NOMA-CR Simulation...")
    stats = sim_engine.run_simulation(num_iterations=200)
    
    # Display results
    print("\nSimulation Results:")
    print("==================")
    for metric, values in stats.items():
        print(f"{metric.replace('_', ' ').title()}:")
        print(f"  Mean: {values['mean']:.4f}")
        print(f"  Std:  {values['std']:.4f}")
        print(f"  Min:  {values['min']:.4f}")
        print(f"  Max:  {values['max']:.4f}")
        print()
    
    # Plot results
    sim_engine.plot_results()
    
    # Example of single optimization run
    print("\nSingle Optimization Example:")
    print("============================")
    opt_result = sim_engine.optimizer.optimize_resources()
    if opt_result['optimization_success']:
        print("Optimization successful!")
        print(f"Power allocation: {opt_result['power_allocation']}")
        print(f"Channel assignment: {opt_result['channel_assignment']}")
        print(f"Cooperation flags: {opt_result['cooperation_flags']}")
        print(f"Objective value: {opt_result['objective_value']:.4f}")
    else:
        print("Optimization failed!")
