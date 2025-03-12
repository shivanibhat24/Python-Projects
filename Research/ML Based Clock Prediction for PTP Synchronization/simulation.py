import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_processing import PTPDataProcessor
from models import DriftPredictionModel
from clock_servo import AdaptiveClockServo

class PTPSimulator:
    """Simulator for PTP clock synchronization with ML-based correction."""
    
    def __init__(self, model_path, data_path, sequence_length=100, prediction_horizon=20):
        """Initialize the simulator with model and data."""
        self.sequence_length = sequence_length
        self.prediction_horizon = prediction_horizon
        self.model_path = model_path
        self.data_path = data_path
        
        # Load data
        self.processor = PTPDataProcessor(sequence_length=sequence_length, prediction_horizon=prediction_horizon)
        self.data = self.processor.load_data(data_path)
        
        if self.data is None:
            raise ValueError("Failed to load data")
        
        # Extract feature dimensions
        feature_columns = [col for col in self.data.columns if col not in ['timestamp', 'offset']]
        self.n_features = len(feature_columns)
        
        # Load model
        self.model = DriftPredictionModel.load(
            model_path,
            sequence_length=sequence_length,
            n_features=self.n_features,
            prediction_horizon=prediction_horizon
        )
        
        # Create servo controller
        self.servo = AdaptiveClockServo(self.model)
    
    def run_simulation(self, target_col='offset', save_results=True):
        """Run simulation and return results."""
        print("Preparing data sequences...")
        X, y = self.processor.create_sequences(self.data, target_col=target_col)
        
        # Run simulation
        print("Running simulation...")
        corrections = []
        true_offsets = []
        predicted_offsets = []
        corrected_offsets = []
        
        for i in range(len(X)):
            # Current offset (using ground truth)
            true_offset = self.processor.target_scaler.inverse_transform(y[i].reshape(-1, 1))[0, 0]
            
            # Get model prediction
            prediction = self.model.predict(X[i:i+1])
            predicted_offset = self.processor.target_scaler.inverse_transform(prediction[0].reshape(-1, 1))[0, 0]
            
            # Calculate correction
            correction = self.servo.compute_correction(true_offset, prediction[0])
            
            # Calculate corrected offset
            corrected = true_offset - correction
            
            # Store results
            corrections.append(correction)
            true_offsets.append(true_offset)
            predicted_offsets.append(predicted_offset)
            corrected_offsets.append(corrected)
            
            # Print progress
            if i % 100 == 0:
                print(f"Processed {i}/{len(X)} samples")
        
        # Calculate performance metrics
        original_rmse = np.sqrt(np.mean(np.array(true_offsets) ** 2))
        corrected_rmse = np.sqrt(np.mean(np.array(corrected_offsets) ** 2))
        improvement = 100 * (original_rmse - corrected_rmse) / original_rmse
        
        results = {
            'true_offsets': true_offsets,
            'predicted_offsets': predicted_offsets,
            'corrections': corrections,
            'corrected_offsets': corrected_offsets,
            'original_rmse': original_rmse,
            'corrected_rmse': corrected_rmse,
            'improvement': improvement
        }
        
        print("\nSimulation Results:")
        print(f"Original RMSE: {original_rmse:.2f} ns")
        print(f"Corrected RMSE: {corrected_rmse:.2f} ns")
        print(f"Improvement: {improvement:.2f}%")
        
        if save_results:
            # Save as CSV
            results_df = pd.DataFrame({
                'true_offset': true_offsets,
                'predicted_offset': predicted_offsets,
                'correction': corrections,
                'corrected_offset': corrected_offsets
            })
            
            results_file = 'simulation_results.csv'
            results_df.to_csv(results_file, index=False)
            print(f"Results saved to {results_file}")
        
        return results
    
    def plot_results(self, results, n_samples=500):
        """Plot simulation results."""
        # Take a subset of samples for visualization
        indices = np.arange(min(n_samples, len(results['true_offsets'])))
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Original vs Corrected Offset
        ax1.plot(indices, np.array(results['true_offsets'])[indices], 'b-', alpha=0.7, label='Original Offset')
        ax1.plot(indices, np.array(results['corrected_offsets'])[indices], 'g-', label='Corrected Offset')
        ax1.set_title('PTP Clock Offset: Original vs Corrected')
        ax1.set_xlabel('Sample')
        ax1.set_ylabel('Offset (ns)')
        ax1.legend()
        ax1.grid(True)
        
        # Plot 2: Applied Corrections
        ax2.plot(indices, np.array(results['corrections'])[indices], 'r-', label='Applied Correction')
        ax2.set_title('ML-Based Corrections Applied')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('Correction (ns)')
        ax2.legend()
        ax2.grid(True)
        
        # Add performance metrics as text
        textstr = '\n'.join((
            f'Original RMSE: {results["original_rmse"]:.2f} ns',
            f'Corrected RMSE: {results["corrected_rmse"]:.2f} ns',
            f'Improvement: {results["improvement"]:.2f}%'
        ))
        
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax1.text(0.05, 0.95, textstr, transform=ax1.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig('simulation_results.png', dpi=300)
        plt.show()

    def compare_methods(self, target_col='offset'):
        """Compare different correction methods."""
        print("Preparing data sequences...")
        X, y = self.processor.create_sequences(self.data, target_col=target_col)
        
        # Define methods to compare
        methods = {
            'No Correction': lambda x, y: 0,
            'PI Controller': lambda x, y: 0.5 * x + 0.1 * self.servo.integral,
            'ML Only': lambda x, y: y[0] if y is not None else 0,
            'ML + PID': lambda x, y: self.servo.compute_correction(x, y)
        }
        
        # Run simulations for each method
        results = {}
        
        for method_name, correction_func in methods.items():
            print(f"\nSimulating {method_name}...")
            
            # Reset servo state
            self.servo.integral = 0
            self.servo.previous_error = 0
            
            corrected_offsets = []
            true_offsets = []
            
            for i in range(len(X)):
                # Current offset (using ground truth)
                true_offset = self.processor.target_scaler.inverse_transform(y[i].reshape(-1, 1))[0, 0]
                true_offsets.append(true_offset)
                
                # Get model prediction if needed
                prediction = None
                if 'ML' in method_name:
                    prediction = self.model.predict(X[i:i+1])[0]
                
                # Calculate correction using the current method
                correction = correction_func(true_offset, prediction)
                
                # Calculate corrected offset
                corrected = true_offset - correction
                corrected_offsets.append(corrected)
                
                # Update integral for PI controller
                if method_name == 'PI Controller':
                    self.servo.integral += true_offset * 0.001
                    self.servo.integral = np.clip(self.servo.integral, -100, 100)
                
                # Print progress
                if i % 100 == 0:
                    print(f"Processed {i}/{len(X)} samples")
            
            # Calculate performance metrics
            rmse = np.sqrt(np.mean(np.array(corrected_offsets) ** 2))
            results[method_name] = {
                'corrected_offsets': corrected_offsets,
                'rmse': rmse
            }
        
        # Store true offsets
        results['true_offsets'] = true_offsets
        
        # Calculate baseline RMSE
        baseline_rmse = np.sqrt(np.mean(np.array(true_offsets) ** 2))
        results['baseline_rmse'] = baseline_rmse
        
        # Print summary
        print("\nComparison Results:")
        print(f"Baseline RMSE (No Correction): {baseline_rmse:.2f} ns")
        
        for method_name in methods.keys():
            rmse = results[method_name]['rmse']
            improvement = 100 * (baseline_rmse - rmse) / baseline_rmse
            print(f"{method_name} RMSE: {rmse:.2f} ns (Improvement: {improvement:.2f}%)")
        
        # Plot comparison
        self.plot_comparison(results)
        
        return results
    
    def plot_comparison(self, results, n_samples=500):
        """Plot comparison of different correction methods."""
        # Take a subset of samples for visualization
        indices = np.arange(min(n_samples, len(results['true_offsets'])))
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot true offsets
        plt.plot(indices, np.array(results['true_offsets'])[indices], 'k-', alpha=0.5, label='Original Offset')
        
        # Plot corrected offsets for each method
        colors = ['r-', 'g-', 'b-', 'm-']
        for i, method_name in enumerate([m for m in results.keys() if m not in ['true_offsets', 'baseline_rmse']]):
            plt.plot(indices, np.array(results[method_name]['corrected_offsets'])[indices], 
                    colors[i], label=f"{method_name} (RMSE: {results[method_name]['rmse']:.2f} ns)")
        
        plt.title('Comparison of Clock Correction Methods')
        plt.xlabel('Sample')
        plt.ylabel('Offset (ns)')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig('method_comparison.png', dpi=300)
        plt.show()


# ml_ptp_sync/synthetic_data.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

class SyntheticPTPDataGenerator:
    """Generate synthetic PTP synchronization data with realistic drift patterns."""
    
    def __init__(self, 
                 duration_hours=24, 
                 sampling_rate_hz=1,
                 base_drift_ppm=1.0,
                 temp_sensitivity=0.1,
                 noise_level=5.0):
        """
        Initialize the synthetic data generator.
        
        Args:
            duration_hours: Duration of the dataset in hours
            sampling_rate_hz: Sampling rate in Hz
            base_drift_ppm: Base clock drift in parts per million
            temp_sensitivity: Sensitivity to temperature changes (ns/°C)
            noise_level: Standard deviation of Gaussian noise (ns)
        """
        self.duration_hours = duration_hours
        self.sampling_rate_hz = sampling_rate_hz
        self.base_drift_ppm = base_drift_ppm
        self.temp_sensitivity = temp_sensitivity
        self.noise_level = noise_level
        
        # Calculate total samples
        self.total_samples = int(duration_hours * 3600 * sampling_rate_hz)
        
        # Initialize random seed
        np.random.seed(42)
    
    def generate_timestamps(self):
        """Generate evenly spaced timestamps."""
        start_time = datetime.now()
        interval = timedelta(seconds=1.0/self.sampling_rate_hz)
        
        timestamps = [start_time + i * interval for i in range(self.total_samples)]
        unix_timestamps = [(t - datetime(1970, 1, 1)).total_seconds() for t in timestamps]
        
        return unix_timestamps
    
    def generate_temperature_profile(self):
        """Generate realistic temperature variations."""
        # Daily cycle with random variations
        time_points = np.linspace(0, 2*np.pi, self.total_samples)
        
        # Base daily cycle (cooler at night, warmer during day)
        base_temp = 25 + 5 * np.sin(time_points - np.pi/2)
        
        # Add some random variations
        random_walk = np.cumsum(np.random.normal(0, 0.02, self.total_samples))
        random_walk = random_walk - np.mean(random_walk)  # Center around zero
        
        # Fast variations
        fast_variations = np.random.normal(0, 0.5, self.total_samples)
        
        # Combine components
        temperature = base_temp + random_walk + fast_variations
        
        return temperature
    
    def generate_network_delay(self):
        """Generate realistic network delay variations."""
        # Base delay with some random packet jitter
        base_delay = 100  # 100 ns base network delay
        
        # Random jitter
        jitter = np.random.exponential(5, self.total_samples)
        
        # Occasional congestion events (higher delays)
        congestion_prob = 0.01  # 1% chance of congestion per sample
        congestion_mask = np.random.random(self.total_samples) < congestion_prob
        congestion_effect = np.zeros(self.total_samples)
        congestion_duration = 10  # Duration of congestion in samples
        
        # Apply congestion effects
        for i in np.where(congestion_mask)[0]:
            end_idx = min(i + congestion_duration, self.total_samples)
            # Exponential decay of congestion
            congestion_effect[i:end_idx] = np.maximum(
                congestion_effect[i:end_idx],
                50 * np.exp(-np.arange(end_idx-i)/5)
            )
        
        # Combine components
        delay = base_delay + jitter + congestion_effect
        
        return delay
    
    def generate_clock_drift(self, temperature):
        """Generate clock drift affected by temperature and random fluctuations."""
        # Base linear drift (clock running faster or slower than master)
        time_array = np.arange(self.total_samples) / (self.sampling_rate_hz * 3600)  # Time in hours
        base_drift = self.base_drift_ppm * 1e-6 * time_array * 1e9  # Convert ppm to ns
        
        # Temperature effect on drift
        temp_normalized = temperature - np.mean(temperature)
        temp_effect = self.temp_sensitivity * temp_normalized
        
        # Random walk component (frequency instability)
        random_walk = np.cumsum(np.random.normal(0, 0.01, self.total_samples))
        
        # Occasional frequency jumps
        jump_prob = 0.001  # 0.1% chance of frequency jump per sample
        jump_mask = np.random.random(self.total_samples) < jump_prob
        jump_points = np.where(jump_mask)[0]
        
        jumps = np.zeros(self.total_samples)
        for jump_idx in jump_points:
            jump_size = np.random.normal(0, 0.5)  # Random jump size
            jumps[jump_idx:] += jump_size
        
        # Combine all components
        drift = base_drift + temp_effect + random_walk + jumps
        
        # Add measurement noise
        drift += np.random.normal(0, self.noise_level, self.total_samples)
        
        return drift
    
    def generate_dataset(self):
        """Generate complete synthetic PTP dataset."""
        print("Generating synthetic PTP dataset...")
        
        # Generate timestamps
        timestamps = self.generate_timestamps()
        
        # Generate temperature profile
        temperature = self.generate_temperature_profile()
        
        # Generate network delay
        delay = self.generate_network_delay()
        
        # Generate clock drift (offset)
        offset = self.generate_clock_drift(temperature)
        
        # Combine into DataFrame
        data = pd.DataFrame({
            'timestamp': timestamps,
            'offset': offset,
            'delay': delay,
            'temperature': temperature
        })
        
        # Calculate derived features
        data['offset_rate'] = data['offset'].diff() / data['timestamp'].diff()
        data['delay_rate'] = data['delay'].diff() / data['timestamp'].diff()
        
        # Fill NaN values
        data = data.fillna(method='bfill')
        
        print(f"Generated dataset with {len(data)} samples")
        return data
    
    def plot_dataset(self, data, n_samples=1000):
        """Plot a preview of the generated dataset."""
        # Sample subset of data for visualization
        if len(data) > n_samples:
            sample_indices = np.linspace(0, len(data)-1, n_samples, dtype=int)
            data_sample = data.iloc[sample_indices]
        else:
            data_sample = data
        
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=(12, 16))
        
        # Plot 1: Clock Offset
        axes[0].plot(data_sample['timestamp'], data_sample['offset'], 'b-')
        axes[0].set_title('Clock Offset')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('Offset (ns)')
        axes[0].grid(True)
        
        # Plot 2: Network Delay
        axes[1].plot(data_sample['timestamp'], data_sample['delay'], 'r-')
        axes[1].set_title('Network Delay')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Delay (ns)')
        axes[1].grid(True)
        
        # Plot 3: Temperature
        axes[2].plot(data_sample['timestamp'], data_sample['temperature'], 'g-')
        axes[2].set_title('Temperature')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Temperature (°C)')
        axes[2].grid(True)
        
        # Plot 4: Offset Rate of Change
        axes[3].plot(data_sample['timestamp'], data_sample['offset_rate'], 'm-')
        axes[3].set_title('Offset Rate of Change')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Rate (ns/s)')
        axes[3].grid(True)
        
        plt.tight_layout()
        plt.savefig('synthetic_data_preview.png', dpi=300)
        plt.show()
    
    def save_dataset(self, data, filepath='synthetic_ptp_data.csv'):
        """Save dataset to CSV file."""
        data.to_csv(filepath, index=False)
        print(f"Dataset saved to {filepath}")

