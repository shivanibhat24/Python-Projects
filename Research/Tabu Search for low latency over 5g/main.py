import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import deque
import random
import time
import math

class TabuSearchTCP5GOptimizer:
    """
    Tabu Search-based optimization framework for TCP over 5G networks.
    Optimizes congestion control parameters and buffer management to reduce latency.
    """
    
    def __init__(self, 
                 tabu_tenure=10, 
                 max_iterations=100, 
                 neighborhood_size=20,
                 aspiration_criteria=True,
                 diversification=True):
        # Tabu search parameters
        self.tabu_tenure = tabu_tenure
        self.max_iterations = max_iterations
        self.neighborhood_size = neighborhood_size
        self.aspiration_criteria = aspiration_criteria
        self.diversification = diversification
        
        # Initialize tabu list as FIFO queue
        self.tabu_list = deque(maxlen=tabu_tenure)
        
        # Parameter bounds for optimization
        self.param_bounds = {
            'cwnd_init': (10, 100),           # Initial congestion window (segments)
            'cwnd_max': (100, 1000),          # Maximum congestion window (segments)
            'rtt_alpha': (0.1, 0.9),          # RTT estimation smoothing factor
            'rto_beta': (1.1, 4.0),           # RTO backoff factor
            'rto_min': (1, 10),               # Minimum RTO (ms)
            'buffer_size': (32, 512),         # Buffer size (KB)
            'pacing_rate': (0.5, 2.0),        # Pacing rate factor
            'ack_frequency': (1, 10),         # ACK frequency
            'delayed_ack_timeout': (1, 50)    # Delayed ACK timeout (ms)
        }
        
        # Best solution found so far
        self.best_solution = None
        self.best_objective = float('inf')
        
        # History for analytics
        self.history = []
    
    def generate_initial_solution(self):
        """Generate a random initial solution within the parameter bounds."""
        solution = {}
        for param, (lower, upper) in self.param_bounds.items():
            if param in ['rtt_alpha', 'rto_beta', 'pacing_rate']:
                # Continuous parameters
                solution[param] = random.uniform(lower, upper)
            else:
                # Integer parameters
                solution[param] = random.randint(lower, upper)
        return solution
    
    def generate_neighbor(self, current_solution):
        """Generate a neighbor by modifying one or two parameters."""
        neighbor = current_solution.copy()
        
        # Randomly choose number of parameters to change (1 or 2)
        num_params_to_change = random.randint(1, min(2, len(self.param_bounds)))
        
        # Select parameters to change
        params_to_change = random.sample(list(self.param_bounds.keys()), num_params_to_change)
        
        for param in params_to_change:
            lower, upper = self.param_bounds[param]
            
            # Determine perturbation magnitude (adaptive to iteration progress)
            if param in ['rtt_alpha', 'rto_beta', 'pacing_rate']:
                # Continuous parameters
                perturbation = random.uniform(-0.1, 0.1) * (upper - lower)
                new_value = current_solution[param] + perturbation
                new_value = max(lower, min(upper, new_value))
            else:
                # Integer parameters
                perturbation_range = max(1, int((upper - lower) * 0.2))
                perturbation = random.randint(-perturbation_range, perturbation_range)
                new_value = current_solution[param] + perturbation
                new_value = max(lower, min(upper, new_value))
                new_value = int(new_value)
            
            neighbor[param] = new_value
            
        return neighbor
    
    def generate_neighborhood(self, current_solution):
        """Generate a set of neighboring solutions."""
        neighborhood = []
        for _ in range(self.neighborhood_size):
            neighborhood.append(self.generate_neighbor(current_solution))
        return neighborhood
    
    def is_tabu(self, solution):
        """Check if a solution is in the tabu list."""
        # Create a solution signature for comparison
        solution_sig = self._solution_signature(solution)
        
        for tabu_sig in self.tabu_list:
            if solution_sig == tabu_sig:
                return True
        return False
    
    def _solution_signature(self, solution):
        """Create a signature for a solution for tabu list comparison."""
        # Create a tuple of rounded parameter values
        return tuple(round(solution[param], 2) if isinstance(solution[param], float) 
                    else solution[param] for param in sorted(solution.keys()))
    
    def evaluate_solution(self, solution, network_simulator):
        """
        Evaluate a solution using the network simulator.
        Returns a weighted objective function value combining latency, throughput, and packet loss.
        """
        # Run the network simulation with the given parameters
        results = network_simulator.run_simulation(solution)
        
        # Extract performance metrics
        latency = results['avg_latency_ms']
        throughput = results['throughput_mbps']
        packet_loss = results['packet_loss_percent']
        jitter = results['jitter_ms']
        
        # Calculate objective function (weighted combination with priority on latency)
        # Lower value is better
        objective = (
            0.6 * latency +                    # Higher weight for latency
            0.2 * (1000 / max(1, throughput)) + # Inverse of throughput (we want to maximize throughput)
            0.15 * (packet_loss * 10) +         # Penalize packet loss
            0.05 * jitter                       # Small penalty for jitter
        )
        
        return objective, results
    
    def run_optimization(self, network_simulator):
        """Run the tabu search optimization process."""
        # Generate initial solution
        current_solution = self.generate_initial_solution()
        
        # Evaluate initial solution
        current_objective, current_results = self.evaluate_solution(current_solution, network_simulator)
        
        # Initialize best solution
        self.best_solution = current_solution.copy()
        self.best_objective = current_objective
        
        # Iteration counter
        iteration = 0
        
        # Diversification counter
        iterations_without_improvement = 0
        
        # Main tabu search loop
        while iteration < self.max_iterations:
            # Generate neighborhood
            neighborhood = self.generate_neighborhood(current_solution)
            
            # Evaluate all neighbors
            best_neighbor = None
            best_neighbor_objective = float('inf')
            
            for neighbor in neighborhood:
                # Skip if neighbor is tabu, unless aspiration criteria is met
                if self.is_tabu(neighbor) and not self.aspiration_criteria:
                    continue
                
                # Evaluate neighbor
                neighbor_objective, neighbor_results = self.evaluate_solution(neighbor, network_simulator)
                
                # Check if this is the best neighbor or if it satisfies aspiration criteria
                if (not self.is_tabu(neighbor) and neighbor_objective < best_neighbor_objective) or \
                   (self.is_tabu(neighbor) and self.aspiration_criteria and neighbor_objective < self.best_objective):
                    best_neighbor = neighbor
                    best_neighbor_objective = neighbor_objective
            
            # If no non-tabu neighbor was found (unlikely but possible)
            if best_neighbor is None:
                # Pick the least recently tabu neighbor
                for neighbor in neighborhood:
                    neighbor_objective, neighbor_results = self.evaluate_solution(neighbor, network_simulator)
                    if neighbor_objective < best_neighbor_objective:
                        best_neighbor = neighbor
                        best_neighbor_objective = neighbor_objective
            
            # Update current solution
            current_solution = best_neighbor
            current_objective = best_neighbor_objective
            
            # Add to tabu list
            self.tabu_list.append(self._solution_signature(current_solution))
            
            # Update best solution if improved
            if current_objective < self.best_objective:
                self.best_solution = current_solution.copy()
                self.best_objective = current_objective
                iterations_without_improvement = 0
            else:
                iterations_without_improvement += 1
            
            # Record history
            self.history.append({
                'iteration': iteration,
                'current_objective': current_objective,
                'best_objective': self.best_objective,
                'current_solution': current_solution.copy()
            })
            
            # Apply diversification if needed
            if self.diversification and iterations_without_improvement > self.max_iterations // 5:
                current_solution = self._diversify(current_solution)
                current_objective, current_results = self.evaluate_solution(current_solution, network_simulator)
                iterations_without_improvement = 0
            
            # Increment iteration counter
            iteration += 1
            
            # Print progress
            if iteration % 10 == 0:
                print(f"Iteration {iteration}, Best Objective: {self.best_objective:.2f}")
        
        # Return the best solution found
        return self.best_solution, self.best_objective
    
    def _diversify(self, solution):
        """Apply diversification to escape local optima."""
        diversified = solution.copy()
        
        # Select a larger number of parameters to change
        num_params = random.randint(3, len(self.param_bounds))
        params_to_change = random.sample(list(self.param_bounds.keys()), num_params)
        
        for param in params_to_change:
            lower, upper = self.param_bounds[param]
            
            # Apply larger changes to escape local optima
            if param in ['rtt_alpha', 'rto_beta', 'pacing_rate']:
                # For continuous parameters
                diversified[param] = random.uniform(lower, upper)
            else:
                # For integer parameters
                diversified[param] = random.randint(lower, upper)
        
        return diversified
    
    def plot_convergence(self):
        """Plot the convergence of the optimization process."""
        iterations = [h['iteration'] for h in self.history]
        current_objectives = [h['current_objective'] for h in self.history]
        best_objectives = [h['best_objective'] for h in self.history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, current_objectives, 'b-', alpha=0.5, label='Current Solution')
        plt.plot(iterations, best_objectives, 'r-', label='Best Solution')
        plt.xlabel('Iteration')
        plt.ylabel('Objective Function Value')
        plt.title('Tabu Search Convergence')
        plt.legend()
        plt.grid(True)
        return plt
    
    def analyze_parameter_sensitivity(self):
        """Analyze the sensitivity of parameters on the objective function."""
        # Extract parameter values from history
        param_values = {param: [] for param in self.param_bounds.keys()}
        objectives = []
        
        for entry in self.history:
            for param in self.param_bounds.keys():
                param_values[param].append(entry['current_solution'][param])
            objectives.append(entry['current_objective'])
        
        # Create a dataframe for analysis
        df = pd.DataFrame(param_values)
        df['objective'] = objectives
        
        # Calculate correlation between parameters and objective function
        correlations = df.corr()['objective'].drop('objective')
        
        # Sort correlations by absolute value
        sorted_correlations = correlations.abs().sort_values(ascending=False)
        
        return sorted_correlations, df
    
    def get_recommended_configuration(self):
        """Get a comprehensive report of the recommended configuration."""
        if self.best_solution is None:
            return "Optimization not yet performed."
        
        report = "Recommended TCP Configuration for 5G Networks:\n"
        report += "=" * 50 + "\n\n"
        
        report += "Parameter Settings:\n"
        report += "-" * 30 + "\n"
        for param, value in self.best_solution.items():
            param_desc = self._get_parameter_description(param)
            report += f"{param} = {value:.2f if isinstance(value, float) else value} \t({param_desc})\n"
        
        return report
    
    def _get_parameter_description(self, param):
        """Get a description for a parameter."""
        descriptions = {
            'cwnd_init': "Initial congestion window size (segments)",
            'cwnd_max': "Maximum congestion window size (segments)",
            'rtt_alpha': "RTT estimation smoothing factor",
            'rto_beta': "RTO backoff factor",
            'rto_min': "Minimum retransmission timeout (ms)",
            'buffer_size': "Buffer size (KB)",
            'pacing_rate': "Pacing rate factor",
            'ack_frequency': "ACK frequency (packets)",
            'delayed_ack_timeout': "Delayed ACK timeout (ms)"
        }
        return descriptions.get(param, "")


class Network5GSimulator:
    """
    Simulates 5G network conditions for TCP parameter optimization.
    Models realistic 5G network characteristics to evaluate TCP performance.
    """
    
    def __init__(self, 
                 simulation_duration=30,  # seconds
                 network_profile="urban",
                 mobility_pattern="static",
                 background_traffic="moderate",
                 random_seed=42):
        
        self.simulation_duration = simulation_duration
        self.network_profile = network_profile
        self.mobility_pattern = mobility_pattern
        self.background_traffic = background_traffic
        
        # Set random seed for reproducibility
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Initialize network profile
        self._init_network_profile()
        
    def _init_network_profile(self):
        """Initialize network characteristics based on the selected profile."""
        profiles = {
            "urban": {
                "base_bandwidth": 100,    # Mbps
                "base_latency": 10,       # ms
                "latency_variation": 5,   # ms
                "packet_loss_rate": 0.01, # %
                "handover_probability": 0.05 if self.mobility_pattern != "static" else 0,
                "fading_severity": "moderate"
            },
            "suburban": {
                "base_bandwidth": 150,    # Mbps
                "base_latency": 15,       # ms
                "latency_variation": 7,   # ms
                "packet_loss_rate": 0.02, # %
                "handover_probability": 0.03 if self.mobility_pattern != "static" else 0,
                "fading_severity": "mild"
            },
            "rural": {
                "base_bandwidth": 50,     # Mbps
                "base_latency": 25,       # ms
                "latency_variation": 10,  # ms
                "packet_loss_rate": 0.05, # %
                "handover_probability": 0.01 if self.mobility_pattern != "static" else 0,
                "fading_severity": "severe"
            },
            "mmWave": {
                "base_bandwidth": 500,    # Mbps
                "base_latency": 5,        # ms
                "latency_variation": 3,   # ms
                "packet_loss_rate": 0.1,  # %
                "handover_probability": 0.1 if self.mobility_pattern != "static" else 0,
                "fading_severity": "very severe"
            }
        }
        
        # Set network parameters
        self.network_params = profiles.get(self.network_profile, profiles["urban"])
        
        # Adjust for background traffic
        self._adjust_for_background_traffic()
        
    def _adjust_for_background_traffic(self):
        """Adjust network parameters based on background traffic levels."""
        traffic_factors = {
            "light": {
                "bandwidth_factor": 0.9,
                "latency_factor": 1.1,
                "jitter_factor": 1.2
            },
            "moderate": {
                "bandwidth_factor": 0.7,
                "latency_factor": 1.3,
                "jitter_factor": 1.5
            },
            "heavy": {
                "bandwidth_factor": 0.5,
                "latency_factor": 1.6,
                "jitter_factor": 2.0
            }
        }
        
        factors = traffic_factors.get(self.background_traffic, traffic_factors["moderate"])
        
        self.network_params["base_bandwidth"] *= factors["bandwidth_factor"]
        self.network_params["base_latency"] *= factors["latency_factor"]
        self.network_params["latency_variation"] *= factors["jitter_factor"]
        
    def simulate_channel_variations(self, time_steps):
        """Simulate 5G channel variations over time."""
        # Base channel capacity in Mbps
        base_capacity = self.network_params["base_bandwidth"]
        
        # Fading parameters
        fading_severity = {
            "mild": 0.1,
            "moderate": 0.2,
            "severe": 0.3,
            "very severe": 0.5
        }.get(self.network_params["fading_severity"], 0.2)
        
        # Simulate slow fading using a combination of sinusoids
        slow_fading = np.zeros(time_steps)
        for i in range(3):  # Combine 3 sinusoids for realism
            freq = 0.01 * (i + 1)  # Different frequencies for each component
            slow_fading += np.sin(np.linspace(0, freq * 2 * np.pi * time_steps, time_steps))
        
        # Normalize and scale by severity
        slow_fading = (slow_fading / 3) * fading_severity
        
        # Simulate fast fading (Rayleigh)
        fast_fading = np.random.rayleigh(scale=0.5, size=time_steps) - 0.5
        fast_fading *= fading_severity / 2
        
        # Combine fading effects
        fading_factor = 1 + slow_fading + fast_fading
        
        # Ensure no negative capacity
        fading_factor = np.maximum(0.1, fading_factor)
        
        # Calculate channel capacity over time
        channel_capacity = base_capacity * fading_factor
        
        return channel_capacity
    
    def simulate_handovers(self, time_steps):
        """Simulate handover events based on mobility pattern."""
        handover_probability = self.network_params["handover_probability"]
        
        # Adjust handover probability based on mobility pattern
        if self.mobility_pattern == "vehicular":
            handover_probability *= 2
        elif self.mobility_pattern == "pedestrian":
            handover_probability *= 1.5
        
        # Generate handover events
        handover_events = np.random.random(time_steps) < handover_probability
        
        # Handover effect on latency (ms)
        handover_latency = np.zeros(time_steps)
        
        # Apply latency effect for each handover
        for i in range(time_steps):
            if handover_events[i]:
                # Handover causes latency spike for a few seconds
                effect_duration = min(10, time_steps - i)
                handover_latency[i:i+effect_duration] += 50 * np.exp(-np.arange(effect_duration)/3)
                
        return handover_latency
    
    def simulate_tcp_dynamics(self, tcp_params, channel_capacity, handover_latency, time_steps):
        """
        Simulate TCP dynamics over the 5G channel.
        
        Args:
            tcp_params: Dict of TCP parameters to evaluate
            channel_capacity: Array of channel capacity over time
            handover_latency: Array of additional latency due to handovers
            time_steps: Number of time steps to simulate
            
        Returns:
            Dict of performance metrics
        """
        # Extract TCP parameters
        cwnd_init = tcp_params['cwnd_init']
        cwnd_max = tcp_params['cwnd_max']
        rtt_alpha = tcp_params['rtt_alpha']
        rto_beta = tcp_params['rto_beta']
        rto_min = tcp_params['rto_min']
        buffer_size = tcp_params['buffer_size']
        pacing_rate = tcp_params['pacing_rate']
        ack_frequency = tcp_params['ack_frequency']
        
        # Initialize TCP state
        cwnd = cwnd_init
        ssthresh = cwnd_max
        rtt_estimate = self.network_params["base_latency"]
        rto = max(rto_min, 4 * rtt_estimate)
        
        # Initialize performance metrics
        throughput = np.zeros(time_steps)
        latency = np.zeros(time_steps)
        packet_loss = np.zeros(time_steps)
        
        # Segment size in bytes
        mss = 1460
        
        # Buffer in segments
        buffer_segments = buffer_size * 1024 // mss
        
        # Packet queue
        buffer_occupancy = 0
        
        # Time step in seconds
        dt = self.simulation_duration / time_steps
        
        # Base RTT in seconds
        base_rtt = self.network_params["base_latency"] / 1000
        
        # Simulate TCP dynamics over time
        for t in range(time_steps):
            # Current channel conditions
            current_capacity = channel_capacity[t]  # Mbps
            additional_latency = handover_latency[t]  # ms
            
            # Calculate current network RTT (ms)
            network_rtt = self.network_params["base_latency"] + additional_latency
            
            # Add jitter
            jitter = np.random.normal(0, self.network_params["latency_variation"])
            network_rtt = max(1, network_rtt + jitter)
            
            # Calculate queuing delay based on buffer occupancy
            if current_capacity > 0:
                # Buffer drain rate in segments per second
                drain_rate = current_capacity * 1e6 / (8 * mss)
                
                # Queuing delay in ms
                queuing_delay = buffer_occupancy * 1000 / drain_rate if drain_rate > 0 else 0
            else:
                queuing_delay = 0
                
            # Total RTT including queuing
            total_rtt = network_rtt + queuing_delay
            
            # Update RTT estimate using EWMA
            rtt_estimate = (1 - rtt_alpha) * rtt_estimate + rtt_alpha * total_rtt
            
            # Update RTO
            rto = max(rto_min, rto_beta * rtt_estimate)
            
            # Calculate packet loss probability based on network conditions
            loss_prob = self.network_params["packet_loss_rate"]
            
            # Higher loss during handovers
            if handover_latency[t] > 0:
                loss_prob *= 2
                
            # Buffer overflow adds to loss probability
            if buffer_occupancy >= buffer_segments:
                loss_prob += 0.5  # High loss probability due to buffer overflow
            
            # Experience packet loss
            experienced_loss = np.random.random() < loss_prob
            
            # Congestion control
            if experienced_loss:
                # Fast retransmit/recovery
                ssthresh = max(2, cwnd // 2)
                cwnd = ssthresh
            else:
                # Normal ACK processing (simplified)
                if cwnd < ssthresh:
                    # Slow start
                    cwnd += 1
                else:
                    # Congestion avoidance
                    cwnd += 1 / cwnd
            
            # Apply pacing to smooth traffic
            cwnd = min(cwnd, cwnd_max)
            effective_cwnd = min(cwnd, cwnd_max * pacing_rate)
            
            # Calculate send rate based on window and RTT
            send_rate_segments = effective_cwnd / (total_rtt / 1000)
            send_rate_mbps = send_rate_segments * mss * 8 / 1e6
            
            # Calculate actual throughput (limited by channel capacity)
            achieved_throughput = min(send_rate_mbps, current_capacity)
            
            # Update buffer occupancy
            segments_sent = achieved_throughput * 1e6 * dt / (8 * mss)
            buffer_occupancy = min(buffer_segments, buffer_occupancy + segments_sent)
            
            # Drain buffer based on channel capacity
            segments_drained = current_capacity * 1e6 * dt / (8 * mss)
            buffer_occupancy = max(0, buffer_occupancy - segments_drained)
            
            # Record metrics
            throughput[t] = achieved_throughput
            latency[t] = total_rtt
            packet_loss[t] = 1 if experienced_loss else 0
        
        # Calculate aggregate metrics
        avg_throughput = np.mean(throughput)
        avg_latency = np.mean(latency)
        max_latency = np.max(latency)
        p95_latency = np.percentile(latency, 95)
        jitter = np.std(latency)
        loss_rate = np.mean(packet_loss) * 100
        
        return {
            'throughput_mbps': avg_throughput,
            'avg_latency_ms': avg_latency,
            'max_latency_ms': max_latency,
            'p95_latency_ms': p95_latency,
            'jitter_ms': jitter,
            'packet_loss_percent': loss_rate,
            'time_series': {
                'throughput': throughput,
                'latency': latency,
                'packet_loss': packet_loss
            }
        }
    
    def run_simulation(self, tcp_params):
        """
        Run a full simulation of TCP over 5G with the given parameters.
        
        Args:
            tcp_params: Dict of TCP parameters to evaluate
            
        Returns:
            Dict of performance metrics
        """
        # Number of time steps in the simulation
        time_steps = int(self.simulation_duration * 10)  # 10 steps per second for detailed simulation
        
        # Simulate 5G channel variations
        channel_capacity = self.simulate_channel_variations(time_steps)
        
        # Simulate handover events
        handover_latency = self.simulate_handovers(time_steps)
        
        # Simulate TCP dynamics
        results = self.simulate_tcp_dynamics(tcp_params, channel_capacity, handover_latency, time_steps)
        
        return results
    
    def plot_simulation_results(self, results):
        """
        Plot detailed simulation results.
        
        Args:
            results: Dict of simulation results with time series data
        """
        time_series = results['time_series']
        time_steps = len(time_series['throughput'])
        time_axis = np.linspace(0, self.simulation_duration, time_steps)
        
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Plot throughput
        axs[0].plot(time_axis, time_series['throughput'], 'b-')
        axs[0].set_ylabel('Throughput (Mbps)')
        axs[0].set_title('TCP Performance over 5G')
        axs[0].grid(True)
        
        # Plot latency
        axs[1].plot(time_axis, time_series['latency'], 'r-')
        axs[1].set_ylabel('Latency (ms)')
        axs[1].grid(True)
        
        # Plot packet loss
        axs[2].plot(time_axis, time_series['packet_loss'], 'k-')
        axs[2].set_ylabel('Packet Loss')
        axs[2].set_xlabel('Time (s)')
        axs[2].grid(True)
        
        plt.tight_layout()
        return fig


def main():
    """Run a full optimization experiment in various 5G scenarios."""
    # Define network scenarios to test
    scenarios = [
        {"name": "Urban Static", "profile": "urban", "mobility": "static", "traffic": "moderate"},
        {"name": "Urban Mobile", "profile": "urban", "mobility": "vehicular", "traffic": "moderate"},
        {"name": "mmWave Dense Urban", "profile": "mmWave", "mobility": "pedestrian", "traffic": "heavy"}
    ]
    
    # Store results for each scenario
    scenario_results = []
    
    for scenario in scenarios:
        print(f"\nOptimizing for scenario: {scenario['name']}")
        print("=" * 50)
        
        # Create simulator for this scenario
        simulator = Network5GSimulator(
            simulation_duration=30,
            network_profile=scenario["profile"],
            mobility_pattern=scenario["mobility"],
            background_traffic=scenario["traffic"]
        )
        
        # Create optimizer
        optimizer = TabuSearchTCP5GOptimizer(
            tabu_tenure=15,
            max_iterations=50,
            neighborhood_size=15,
            aspiration_criteria=True,
            diversification=True
        )
        
        # Run optimization
        best_solution, best_objective = optimizer.run_optimization(simulator)
        
        # Evaluate final solution in detail
        final_results = simulator.run_simulation(best_solution)
        
        # Record results
        scenario_results.append({
            "scenario": scenario["name"],
            "best_solution": best_solution,
            "metrics": {
                "latency_ms": final_results["avg_latency_ms"],
                "throughput_mbps": final_results["throughput_mbps"],
                "packet_loss_percent": final_results["packet_loss_percent"],
                "jitter_ms": final_results["jitter_ms"]
            }
        })
        
        # Print results
        print("\nOptimization Results:")
        print(f"Best Objective Value: {best_objective:.2f}")
        print("\nBest TCP Configuration:")
        for param, value in best_solution.items():
            print(f"  {param}: {value:.2f if isinstance(value, float) else value}")
        
        print("\nPerformance Metrics:")
        print(f"  Average Latency: {final_results['avg_latency_ms']:.2f} ms")
        print(f"  P95 Latency: {final_results['p95_latency_ms']:.2f} ms")
        print(f"  Throughput: {final_results['throughput_mbps']:.2f} Mbps")
        print(f"  Packet Loss: {final_results['packet_loss_percent']:.4f}%")
        print(f"  Jitter: {final_results['jitter_ms']:.2f} ms")
        
        # Plot convergence
        optimizer.plot_convergence()
        plt.savefig(f"convergence_{scenario['name'].replace(' ', '_')}.png")
        
        # Plot simulation results
        simulator.plot_simulation_results(final_results)
        plt.savefig(f"simulation_{scenario['name'].replace(' ', '_')}.png")
        
    # Compare results across scenarios
    print("\nComparison of Optimized Solutions Across Scenarios:")
