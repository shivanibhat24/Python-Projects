#!/usr/bin/env python3
"""
GNSS Receiver Algorithms for GNU Radio
Implements key signal processing blocks for GPS/GNSS reception
"""

import numpy as np
from gnuradio import gr
import scipy.signal as signal
from scipy.fft import fft, ifft
import struct
import math

class GPSCACodeGenerator:
    """Generates GPS C/A codes for different satellites"""
    
    def __init__(self):
        # G1 and G2 generator polynomials for GPS C/A code
        self.g1_poly = [1, 0, 0, 1, 0, 0, 0, 0, 0, 1]  # x^10 + x^3 + 1
        self.g2_poly = [1, 0, 1, 1, 0, 1, 0, 1, 1, 1]  # x^10 + x^9 + x^8 + x^6 + x^3 + x^2 + 1
        
        # G2 delay for each satellite (1-32)
        self.g2_delays = {
            1: [2, 6], 2: [3, 7], 3: [4, 8], 4: [5, 9], 5: [1, 9],
            6: [2, 10], 7: [1, 8], 8: [2, 9], 9: [3, 10], 10: [2, 3],
            11: [3, 4], 12: [5, 6], 13: [6, 7], 14: [7, 8], 15: [8, 9],
            16: [9, 10], 17: [1, 4], 18: [2, 5], 19: [3, 6], 20: [4, 7],
            21: [5, 8], 22: [6, 9], 23: [1, 3], 24: [4, 6], 25: [5, 7],
            26: [6, 8], 27: [7, 9], 28: [8, 10], 29: [1, 6], 30: [2, 7],
            31: [3, 8], 32: [4, 9]
        }
    
    def generate_ca_code(self, prn, length=1023):
        """Generate C/A code for given PRN satellite"""
        if prn not in self.g2_delays:
            raise ValueError(f"PRN {prn} not supported")
        
        # Initialize shift registers
        g1_reg = [1] * 10
        g2_reg = [1] * 10
        
        ca_code = []
        delays = self.g2_delays[prn]
        
        for _ in range(length):
            # Output is XOR of G1 output and delayed G2 outputs
            g1_out = g1_reg[9]
            g2_out = g2_reg[delays[0] - 1] ^ g2_reg[delays[1] - 1]
            
            ca_code.append(g1_out ^ g2_out)
            
            # Update G1 register
            g1_feedback = g1_reg[2] ^ g1_reg[9]
            g1_reg = [g1_feedback] + g1_reg[:-1]
            
            # Update G2 register
            g2_feedback = (g2_reg[1] ^ g2_reg[2] ^ g2_reg[5] ^ 
                          g2_reg[7] ^ g2_reg[8] ^ g2_reg[9])
            g2_reg = [g2_feedback] + g2_reg[:-1]
        
        # Convert to bipolar (+1/-1)
        return np.array([1 if x == 0 else -1 for x in ca_code])

class GNSSAcquisition(gr.sync_block):
    """
    GNSS Signal Acquisition Block
    Performs coarse acquisition using FFT-based correlation
    """
    
    def __init__(self, sample_rate, if_freq, prn_list, coherent_ms=1):
        gr.sync_block.__init__(
            self,
            name="gnss_acquisition",
            in_sig=[np.complex64],
            out_sig=[]
        )
        
        self.sample_rate = sample_rate
        self.if_freq = if_freq
        self.prn_list = prn_list
        self.coherent_ms = coherent_ms
        
        # GPS parameters
        self.chip_rate = 1.023e6  # C/A code chip rate
        self.code_length = 1023
        self.samples_per_code = int(sample_rate / 1000)  # 1ms of samples
        
        # Generate local codes
        self.ca_generator = GPSCACodeGenerator()
        self.local_codes = {}
        for prn in prn_list:
            ca_code = self.ca_generator.generate_ca_code(prn)
            # Oversample to match sample rate
            oversampled_code = np.repeat(ca_code, self.samples_per_code // self.code_length)
            self.local_codes[prn] = np.conj(fft(oversampled_code))
        
        # Doppler search parameters
        self.doppler_range = 5000  # ±5kHz
        self.doppler_step = 250   # 250Hz steps
        self.doppler_bins = np.arange(-self.doppler_range, 
                                     self.doppler_range + self.doppler_step, 
                                     self.doppler_step)
        
        # Detection results
        self.acquisition_results = {}
    
    def work(self, input_items, output_items):
        in0 = input_items[0]
        
        if len(in0) < self.samples_per_code:
            return 0
        
        # Process one code period
        data = in0[:self.samples_per_code]
        
        # Perform acquisition for each PRN
        for prn in self.prn_list:
            max_corr = 0
            best_doppler = 0
            best_code_phase = 0
            
            # Search over Doppler frequencies
            for doppler in self.doppler_bins:
                # Remove carrier with Doppler offset
                t = np.arange(len(data)) / self.sample_rate
                carrier = np.exp(-1j * 2 * np.pi * (self.if_freq + doppler) * t)
                baseband = data * carrier
                
                # FFT-based correlation
                data_fft = fft(baseband)
                correlation = ifft(data_fft * self.local_codes[prn])
                corr_magnitude = np.abs(correlation)
                
                # Find peak
                peak_idx = np.argmax(corr_magnitude)
                peak_val = corr_magnitude[peak_idx]
                
                if peak_val > max_corr:
                    max_corr = peak_val
                    best_doppler = doppler
                    best_code_phase = peak_idx
            
            # Store acquisition results
            self.acquisition_results[prn] = {
                'correlation': max_corr,
                'doppler': best_doppler,
                'code_phase': best_code_phase,
                'acquired': max_corr > self.get_threshold()
            }
        
        return len(data)
    
    def get_threshold(self):
        """Simple threshold based on noise floor estimation"""
        return 2.5  # Adjust based on signal conditions
    
    def get_acquisition_results(self):
        return self.acquisition_results

class GNSSTrackingLoop(gr.sync_block):
    """
    GNSS Signal Tracking Block
    Implements DLL (Delay Lock Loop) and PLL (Phase Lock Loop)
    """
    
    def __init__(self, sample_rate, if_freq, prn, initial_doppler=0, initial_code_phase=0):
        gr.sync_block.__init__(
            self,
            name="gnss_tracking",
            in_sig=[np.complex64],
            out_sig=[np.complex64]  # Output demodulated data
        )
        
        self.sample_rate = sample_rate
        self.if_freq = if_freq
        self.prn = prn
        
        # GPS parameters
        self.chip_rate = 1.023e6
        self.code_length = 1023
        self.samples_per_chip = sample_rate / self.chip_rate
        
        # Generate local C/A code
        self.ca_generator = GPSCACodeGenerator()
        self.ca_code = self.ca_generator.generate_ca_code(prn)
        
        # Tracking loop parameters
        self.code_phase = initial_code_phase
        self.carrier_phase = 0
        self.doppler_freq = initial_doppler
        
        # Loop filter parameters
        self.dll_bandwidth = 2.0  # Hz
        self.pll_bandwidth = 25.0  # Hz
        
        # Initialize loop filters
        self.dll_filter = self.init_loop_filter(self.dll_bandwidth)
        self.pll_filter = self.init_loop_filter(self.pll_bandwidth)
        
        # Correlator spacing
        self.correlator_spacing = 0.5  # chips
        
        # Integration time
        self.integration_time = 0.001  # 1ms
        self.samples_per_integration = int(self.integration_time * sample_rate)
        
        self.sample_counter = 0
        self.integrate_buffer = np.zeros(self.samples_per_integration, dtype=np.complex64)
        
    def init_loop_filter(self, bandwidth):
        """Initialize 2nd order loop filter"""
        return {
            'prev_error': 0,
            'integrator': 0,
            'K1': 4 * bandwidth,
            'K2': 4 * bandwidth * bandwidth
        }
    
    def update_loop_filter(self, filter_state, error):
        """Update loop filter state"""
        filter_state['integrator'] += error
        output = filter_state['K1'] * error + filter_state['K2'] * filter_state['integrator']
        filter_state['prev_error'] = error
        return output
    
    def generate_local_code(self, code_phase, num_samples):
        """Generate local code replica"""
        code_indices = ((code_phase + np.arange(num_samples) / self.samples_per_chip) 
                       % self.code_length).astype(int)
        return self.ca_code[code_indices]
    
    def work(self, input_items, output_items):
        in0 = input_items[0]
        out = output_items[0]
        
        num_samples = min(len(in0), len(out))
        
        for i in range(num_samples):
            # Store sample in integration buffer
            self.integrate_buffer[self.sample_counter] = in0[i]
            self.sample_counter += 1
            
            # Process when integration buffer is full
            if self.sample_counter >= self.samples_per_integration:
                self.process_integration_period()
                self.sample_counter = 0
            
            # Generate output (simplified - normally would be navigation data)
            out[i] = in0[i] * np.exp(-1j * self.carrier_phase)
        
        return num_samples
    
    def process_integration_period(self):
        """Process one integration period"""
        data = self.integrate_buffer
        
        # Generate local carrier
        t = np.arange(len(data)) / self.sample_rate
        local_carrier = np.exp(-1j * 2 * np.pi * 
                              (self.if_freq + self.doppler_freq) * t + 
                              1j * self.carrier_phase)
        
        # Remove carrier
        baseband = data * local_carrier
        
        # Generate early, prompt, and late code replicas
        early_code = self.generate_local_code(
            self.code_phase - self.correlator_spacing, len(data))
        prompt_code = self.generate_local_code(self.code_phase, len(data))
        late_code = self.generate_local_code(
            self.code_phase + self.correlator_spacing, len(data))
        
        # Correlate with local codes
        I_early = np.sum(baseband * early_code).real
        Q_early = np.sum(baseband * early_code).imag
        I_prompt = np.sum(baseband * prompt_code).real
        Q_prompt = np.sum(baseband * prompt_code).imag
        I_late = np.sum(baseband * late_code).real
        Q_late = np.sum(baseband * late_code).imag
        
        # DLL discriminator (Early-Late power)
        early_power = I_early**2 + Q_early**2
        late_power = I_late**2 + Q_late**2
        dll_error = (early_power - late_power) / (early_power + late_power)
        
        # PLL discriminator (atan2)
        pll_error = np.arctan2(Q_prompt, I_prompt)
        
        # Update loop filters
        code_correction = self.update_loop_filter(self.dll_filter, dll_error)
        freq_correction = self.update_loop_filter(self.pll_filter, pll_error)
        
        # Update code phase and carrier frequency
        self.code_phase += code_correction
        self.doppler_freq += freq_correction
        
        # Update carrier phase
        self.carrier_phase += 2 * np.pi * (self.if_freq + self.doppler_freq) * self.integration_time

class GNSSNavigationDecoder(gr.sync_block):
    """
    GNSS Navigation Data Decoder
    Extracts navigation message from demodulated signal
    """
    
    def __init__(self, sample_rate):
        gr.sync_block.__init__(
            self,
            name="gnss_nav_decoder",
            in_sig=[np.complex64],
            out_sig=[]
        )
        
        self.sample_rate = sample_rate
        self.bit_rate = 50  # GPS navigation data rate (50 bps)
        self.samples_per_bit = int(sample_rate / self.bit_rate)
        
        # Navigation message parameters
        self.preamble = [1, 0, 0, 0, 1, 0, 1, 1]  # GPS preamble
        self.subframe_length = 300  # bits
        
        # Decoder state
        self.bit_buffer = []
        self.bit_sync = False
        self.subframe_buffer = []
        self.ephemeris_data = {}
        
        self.sample_counter = 0
        self.integration_buffer = []
    
    def work(self, input_items, output_items):
        in0 = input_items[0]
        
        for sample in in0:
            self.integration_buffer.append(sample)
            self.sample_counter += 1
            
            # Integrate over one bit period
            if self.sample_counter >= self.samples_per_bit:
                bit_value = self.demodulate_bit()
                self.process_bit(bit_value)
                self.sample_counter = 0
                self.integration_buffer = []
        
        return len(in0)
    
    def demodulate_bit(self):
        """Demodulate one navigation bit"""
        # Simple integration and dump
        integrated = np.sum(self.integration_buffer)
        return 1 if integrated.real > 0 else 0
    
    def process_bit(self, bit):
        """Process received navigation bit"""
        self.bit_buffer.append(bit)
        
        # Maintain sliding window for preamble detection
        if len(self.bit_buffer) > 8:
            self.bit_buffer.pop(0)
        
        # Check for preamble
        if len(self.bit_buffer) == 8 and self.bit_buffer == self.preamble:
            self.bit_sync = True
            self.subframe_buffer = []
            print("Preamble detected - bit sync acquired")
        
        # Collect subframe data
        if self.bit_sync:
            self.subframe_buffer.append(bit)
            
            # Process complete subframe
            if len(self.subframe_buffer) == self.subframe_length:
                self.process_subframe(self.subframe_buffer)
                self.subframe_buffer = []
    
    def process_subframe(self, subframe):
        """Process complete navigation subframe"""
        # Extract subframe ID (bits 50-52)
        subframe_id = self.extract_bits(subframe, 49, 3)
        
        print(f"Received subframe {subframe_id}")
        
        # Process based on subframe type
        if subframe_id == 1:
            self.process_subframe1(subframe)
        elif subframe_id == 2:
            self.process_subframe2(subframe)
        elif subframe_id == 3:
            self.process_subframe3(subframe)
    
    def extract_bits(self, data, start, length):
        """Extract bits from data array"""
        bits = data[start:start + length]
        value = 0
        for i, bit in enumerate(bits):
            value += bit * (2 ** (length - 1 - i))
        return value
    
    def process_subframe1(self, subframe):
        """Process subframe 1 - clock corrections"""
        # Extract clock parameters (simplified)
        self.ephemeris_data['week_number'] = self.extract_bits(subframe, 60, 10)
        self.ephemeris_data['sv_health'] = self.extract_bits(subframe, 76, 6)
        print(f"Week number: {self.ephemeris_data['week_number']}")
    
    def process_subframe2(self, subframe):
        """Process subframe 2 - ephemeris data"""
        # Extract orbital parameters (simplified)
        self.ephemeris_data['crs'] = self.extract_bits(subframe, 68, 16)
        self.ephemeris_data['delta_n'] = self.extract_bits(subframe, 90, 16)
        print("Subframe 2 processed - orbital parameters")
    
    def process_subframe3(self, subframe):
        """Process subframe 3 - more ephemeris data"""
        # Extract more orbital parameters (simplified)
        self.ephemeris_data['cic'] = self.extract_bits(subframe, 60, 16)
        self.ephemeris_data['omega0'] = self.extract_bits(subframe, 76, 32)
        print("Subframe 3 processed - more orbital parameters")
    
    def get_ephemeris_data(self):
        """Get decoded ephemeris data"""
        return self.ephemeris_data

class GNSSPositionSolver:
    """
    GNSS Position Solver
    Solves for position, velocity, and time using least squares
    """
    
    def __init__(self):
        self.WGS84_A = 6378137.0  # Semi-major axis (meters)
        self.WGS84_F = 1.0 / 298.257223563  # Flattening
        self.WGS84_E2 = 2 * self.WGS84_F - self.WGS84_F**2  # Eccentricity squared
        self.c = 299792458.0  # Speed of light (m/s)
        
        # Earth rotation rate (rad/s)
        self.omega_e = 7.2921151467e-5
    
    def satellite_position(self, ephemeris, time):
        """Calculate satellite position from ephemeris data"""
        # This is a simplified version - full implementation would use
        # all ephemeris parameters for precise orbit calculation
        
        # For demonstration, return a sample position
        return np.array([20000000.0, 10000000.0, 15000000.0])  # ECEF coordinates
    
    def pseudorange_residual(self, user_pos, sat_pos, pseudorange):
        """Calculate pseudorange residual"""
        distance = np.linalg.norm(sat_pos - user_pos[:3])
        return pseudorange - distance - self.c * user_pos[3]
    
    def solve_position(self, satellite_positions, pseudoranges):
        """Solve for user position using least squares"""
        # Initial guess [x, y, z, clock_bias]
        x = np.array([0.0, 0.0, 0.0, 0.0])
        
        # Iterative least squares
        for iteration in range(10):
            # Calculate residuals and Jacobian
            residuals = []
            jacobian = []
            
            for i, (sat_pos, pseudorange) in enumerate(zip(satellite_positions, pseudoranges)):
                # Range from user to satellite
                range_vec = sat_pos - x[:3]
                range_mag = np.linalg.norm(range_vec)
                
                # Residual
                residual = pseudorange - range_mag - self.c * x[3]
                residuals.append(residual)
                
                # Jacobian row
                unit_vec = range_vec / range_mag
                jac_row = [-unit_vec[0], -unit_vec[1], -unit_vec[2], self.c]
                jacobian.append(jac_row)
            
            residuals = np.array(residuals)
            jacobian = np.array(jacobian)
            
            # Least squares update
            try:
                dx = np.linalg.solve(jacobian.T @ jacobian, jacobian.T @ residuals)
                x += dx
                
                # Check convergence
                if np.linalg.norm(dx) < 1e-6:
                    break
            except np.linalg.LinAlgError:
                print("Singular matrix in position solve")
                break
        
        return x
    
    def ecef_to_lla(self, ecef_pos):
        """Convert ECEF coordinates to Latitude, Longitude, Altitude"""
        x, y, z = ecef_pos
        
        # Calculate longitude
        lon = np.arctan2(y, x)
        
        # Calculate latitude iteratively
        p = np.sqrt(x**2 + y**2)
        lat = np.arctan2(z, p)
        
        for _ in range(5):  # Iterate for precision
            N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat)**2)
            lat = np.arctan2(z + self.WGS84_E2 * N * np.sin(lat), p)
        
        # Calculate altitude
        N = self.WGS84_A / np.sqrt(1 - self.WGS84_E2 * np.sin(lat)**2)
        alt = p / np.cos(lat) - N
        
        return np.degrees(lat), np.degrees(lon), alt

# Example usage and test functions
def test_ca_code_generation():
    """Test C/A code generation"""
    generator = GPSCACodeGenerator()
    
    # Generate codes for first 5 satellites
    for prn in range(1, 6):
        code = generator.generate_ca_code(prn)
        print(f"PRN {prn}: First 10 chips: {code[:10]}")
        print(f"PRN {prn}: Code length: {len(code)}")
        print(f"PRN {prn}: Autocorrelation peak: {np.max(np.correlate(code, code, mode='full'))}")
        print()

def simulate_gnss_signal(prn, sample_rate=4e6, duration=0.001, snr_db=10):
    """Simulate GNSS signal for testing"""
    # Generate C/A code
    generator = GPSCACodeGenerator()
    ca_code = generator.generate_ca_code(prn)
    
    # Parameters
    if_freq = 1.57542e9  # L1 frequency
    doppler_freq = 1000  # 1 kHz Doppler
    samples_per_chip = sample_rate / 1.023e6
    
    # Generate time vector
    t = np.arange(0, duration, 1/sample_rate)
    
    # Oversample C/A code
    code_samples = np.repeat(ca_code, int(samples_per_chip))
    num_samples = len(t)
    
    # Repeat code as necessary
    code_signal = np.tile(code_samples, int(np.ceil(num_samples / len(code_samples))))[:num_samples]
    
    # Generate carrier
    carrier = np.exp(1j * 2 * np.pi * (if_freq + doppler_freq) * t)
    
    # Modulate signal
    signal = code_signal * carrier
    
    # Add noise
    noise_power = np.power(10, -snr_db/10)
    noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    
    return signal + noise

if __name__ == "__main__":
    print("GNSS Receiver Algorithm Test")
    print("=" * 40)
    
    # Test C/A code generation
    print("Testing C/A Code Generation:")
    test_ca_code_generation()
    
    # Test position solver
    print("Testing Position Solver:")
    solver = GNSSPositionSolver()
    
    # Example satellite positions and pseudoranges
    sat_positions = [
        np.array([20000000, 10000000, 15000000]),
        np.array([15000000, 20000000, 10000000]),
        np.array([10000000, 15000000, 20000000]),
        np.array([25000000, 5000000, 12000000])
    ]
    
    pseudoranges = [22000000, 23000000, 24000000, 25000000]
    
    position = solver.solve_position(sat_positions, pseudoranges)
    lla = solver.ecef_to_lla(position[:3])
    
    print(f"Solved position (ECEF): {position[:3]}")
    print(f"Clock bias: {position[3]/solver.c*1e9:.2f} ns")
    print(f"Position (LLA): {lla[0]:.6f}°, {lla[1]:.6f}°, {lla[2]:.2f}m")
    print()
    
    print("GNSS Receiver algorithms ready for GNU Radio integration!")
