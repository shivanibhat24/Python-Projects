# Cooperative NOMA-CR Simulation Engine

A comprehensive Python simulation framework for analyzing and optimizing Cooperative Non-Orthogonal Multiple Access (NOMA) integrated with Cognitive Radio (CR) networks.

## Overview

This simulation engine provides a complete framework for modeling, analyzing, and optimizing cooperative NOMA-CR systems. It includes realistic channel models, primary user activity patterns, NOMA power allocation schemes, cooperative communication protocols, and multi-objective optimization algorithms.

## Features

### 🌐 **System Models**
- **Channel Modeling**: Rayleigh fading with temporal correlation
- **Primary User Activity**: Markov-based ON/OFF models
- **NOMA Power Allocation**: Channel-gain based power coefficients
- **Cooperative Communication**: Relay selection and cooperative transmission
- **Interference Management**: Primary user protection constraints

### 📊 **Performance Metrics**
- **Spectral Efficiency**: bits/s/Hz
- **Energy Efficiency**: bits/J
- **Throughput**: Effective data rates
- **Outage Probability**: QoS reliability
- **Latency**: End-to-end delays

### 🔧 **Optimization Engine**
- **Multi-objective Optimization**: Spectral efficiency, energy efficiency, throughput
- **Dynamic Resource Allocation**: Power, spectrum, and user assignment
- **Constraint Handling**: Primary user interference protection
- **Global Optimization**: Differential evolution algorithm

### 📈 **Visualization & Analysis**
- Real-time performance tracking
- Statistical analysis with mean, std, min, max
- Comprehensive plotting of all metrics
- Convergence analysis

## Installation

### Prerequisites
```bash
pip install numpy scipy matplotlib
```

### Dependencies
- `numpy`: Numerical computations
- `scipy`: Optimization algorithms
- `matplotlib`: Visualization
- `dataclasses`: Data structures (Python 3.7+)

## Quick Start

### Basic Usage

```python
from noma_cr_engine import SystemParameters, SimulationEngine

# Configure system parameters
params = SystemParameters(
    num_primary_users=2,
    num_secondary_users=4,
    num_channels=3,
    max_power_budget=2.0
)

# Create simulation engine
sim_engine = SimulationEngine(params)

# Run simulation
stats = sim_engine.run_simulation(num_iterations=1000)

# Display results
print("Simulation Results:")
for metric, values in stats.items():
    print(f"{metric}: Mean={values['mean']:.4f}, Std={values['std']:.4f}")

# Plot results
sim_engine.plot_results()
```

### Single Optimization Run

```python
# Perform single optimization
opt_result = sim_engine.optimizer.optimize_resources()

if opt_result['optimization_success']:
    print(f"Power allocation: {opt_result['power_allocation']}")
    print(f"Channel assignment: {opt_result['channel_assignment']}")
    print(f"Cooperation flags: {opt_result['cooperation_flags']}")
```

## Configuration

### System Parameters

```python
@dataclass
class SystemParameters:
    # Network topology
    num_primary_users: int = 2
    num_secondary_users: int = 6
    num_channels: int = 4
    
    # Power management
    max_power_budget: float = 1.0  # Watts
    noise_power: float = 1e-9  # Watts
    primary_interference_threshold: float = 1e-8  # Watts
    
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
```

## Architecture

### Core Components

1. **ChannelModel**: Wireless channel modeling with fading
2. **PrimaryUserModel**: Primary user activity patterns
3. **NOMAModel**: Power allocation and SIC operations
4. **CooperativeModel**: Relay selection and cooperative transmission
5. **PerformanceMetrics**: Performance evaluation metrics
6. **OptimizationEngine**: Multi-objective resource optimization
7. **SimulationEngine**: Main simulation orchestrator

### Class Hierarchy

```
SimulationEngine
├── ChannelModel
├── PrimaryUserModel
├── NOMAModel
├── CooperativeModel
├── PerformanceMetrics
└── OptimizationEngine
```

## Key Algorithms

### 1. NOMA Power Allocation
- **Strategy**: Inverse channel gain allocation
- **Constraint**: Minimum power guarantee
- **Objective**: Maximize sum rate with fairness

### 2. Cooperative Relay Selection
- **Criteria**: Channel quality thresholds
- **Protocol**: Decode-and-Forward (DF)
- **Selection**: Best relay based on combined channel gains

### 3. Spectrum Management
- **Sensing**: Primary user activity detection
- **Access**: Opportunistic spectrum utilization
- **Protection**: Interference threshold enforcement

### 4. Multi-objective Optimization
- **Variables**: Power allocation, channel assignment, cooperation modes
- **Objectives**: Spectral efficiency, energy efficiency, throughput
- **Constraints**: Power budget, interference limits
- **Algorithm**: Differential evolution

## Performance Metrics

### Spectral Efficiency
```python
SE = Σ log₂(1 + SINR_i) / bandwidth
```

### Energy Efficiency
```python
EE = Total_Throughput / Total_Power_Consumption
```

### Outage Probability
```python
P_out = P(Achieved_Rate < Required_Rate)
```

### Throughput
```python
Throughput = Σ (Rate_i × Success_Probability_i)
```

## Optimization Problem Formulation

### Objective Function
```
Maximize: α₁×SE + α₂×EE + α₃×Throughput - α₄×Outage_Probability
```

### Decision Variables
- **Power Allocation**: P = [p₁, p₂, ..., pₙ]
- **Channel Assignment**: C = [c₁, c₂, ..., cₙ]
- **Cooperation Modes**: M = [m₁, m₂, ..., mₙ]

### Constraints
- Power budget: Σpᵢ ≤ P_max
- Interference: Σ(pᵢ × hᵢⱼ) ≤ I_th
- NOMA ordering: p_weak ≥ p_strong
- Channel occupancy: Primary user protection

## Customization

### Adding New Metrics

```python
class CustomMetrics(PerformanceMetrics):
    def calculate_fairness_index(self, rates):
        """Calculate Jain's fairness index"""
        sum_rates = np.sum(rates)
        sum_squared_rates = np.sum(rates**2)
        n = len(rates)
        return (sum_rates**2) / (n * sum_squared_rates)
```

### Custom Optimization Objectives

```python
def custom_objective_function(self, x):
    # Your custom optimization logic here
    # Return negative value for maximization
    return -custom_performance_metric
```

### Channel Model Extensions

```python
class CustomChannelModel(ChannelModel):
    def nakagami_fading(self, m_parameter=2):
        """Nakagami fading channel model"""
        return np.random.gamma(m_parameter, 1/m_parameter, self.shape)
```

## Example Scenarios

### Scenario 1: Dense Urban Environment
```python
params = SystemParameters(
    num_primary_users=3,
    num_secondary_users=8,
    num_channels=5,
    max_power_budget=5.0,
    primary_interference_threshold=1e-7
)
```

### Scenario 2: Rural Low-Power Network
```python
params = SystemParameters(
    num_primary_users=1,
    num_secondary_users=4,
    num_channels=2,
    max_power_budget=0.5,
    cooperation_probability=0.9
)
```

### Scenario 3: High-Mobility Environment
```python
params = SystemParameters(
    # Standard parameters with high channel coherence
    channel_coherence_blocks=5,
    relay_selection_threshold=0.7
)
```

## Results Interpretation

### Spectral Efficiency
- **High values**: Efficient spectrum utilization
- **Low values**: Poor channel conditions or interference

### Energy Efficiency
- **High values**: Power-efficient transmission
- **Low values**: Excessive power consumption

### Outage Probability
- **Low values**: Reliable communication
- **High values**: Frequent communication failures

### Cooperation Usage
- **High values**: Frequent cooperative transmission
- **Low values**: Mainly direct transmission

## Troubleshooting

### Common Issues

1. **Optimization Convergence**
   - Reduce `max_iterations` for faster results
   - Increase `convergence_threshold` for stricter convergence

2. **Memory Issues**
   - Reduce `num_iterations` in simulation
   - Decrease number of users/channels

3. **Performance Issues**
   - Use vectorized operations
   - Reduce optimization population size

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
sim_engine.run_simulation(num_iterations=10, verbose=True)
```

## Extensions & Future Work

### Possible Enhancements

1. **Machine Learning Integration**
   - Deep Q-Network (DQN) for resource allocation
   - Reinforcement learning for adaptive strategies

2. **Advanced Channel Models**
   - Correlated fading
   - Multi-path propagation
   - Mobility models

3. **Security Considerations**
   - Physical layer security
   - Jamming resistance
   - Secure cooperative protocols

4. **5G/6G Integration**
   - Massive MIMO compatibility
   - Network slicing support
   - Ultra-reliable low-latency communications (URLLC)

### Research Applications

- **Academic Research**: Performance analysis and comparison
- **Industry Development**: System design and optimization
- **Standards Development**: Protocol evaluation and validation
- **Network Planning**: Deployment strategy optimization


## Acknowledgments

- Research community for theoretical foundations
- Open-source contributors for tools and libraries
- Academic institutions for research support

---

**Note**: This simulation engine is designed for research and educational purposes. For production deployments, additional considerations for robustness, security, and compliance may be required.
