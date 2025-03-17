import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft
import librosa
import soundfile as sf
import noisereduce as nr

class NoiseRemovalSystem:
    """
    Comprehensive system for noise cancellation in audio signals.
    Includes multiple methods that can be used individually or in combination.
    """
    
    def __init__(self):
        self.sample_rate = None
        self.audio_data = None
        self.original_audio = None
    
    def load_audio(self, file_path):
        """Load audio from a file path."""
        self.sample_rate, self.audio_data = wavfile.read(file_path)
        
        # Convert to float for processing
        if self.audio_data.dtype != np.float32:
            self.audio_data = self.audio_data.astype(np.float32)
            
            # Normalize to [-1, 1] range if integer data
            if np.issubdtype(wavfile.read(file_path)[1].dtype, np.integer):
                max_value = np.iinfo(wavfile.read(file_path)[1].dtype).max
                self.audio_data = self.audio_data / max_value
        
        # Store original for comparison
        self.original_audio = np.copy(self.audio_data)
        
        return self.audio_data, self.sample_rate
    
    def save_audio(self, file_path, data=None):
        """Save audio to a file."""
        if data is None:
            data = self.audio_data
            
        # Convert back to int16 for saving
        if data.dtype == np.float32:
            data = np.int16(data * 32767)
            
        wavfile.write(file_path, self.sample_rate, data)
    
    def spectral_subtraction(self, noise_sample, smoothing_factor=1.5, min_snr=0.01):
        """
        Basic spectral subtraction noise reduction.
        
        Args:
            noise_sample: Section of audio containing only noise
            smoothing_factor: Oversubtraction factor for noise spectrum
            min_snr: Minimum signal-to-noise ratio to prevent musical noise
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
        
        # Calculate noise spectrum
        noise_fft = fft(noise_sample)
        noise_power = np.abs(noise_fft)**2
        noise_power_mean = np.mean(noise_power)
        
        # Calculate signal spectrum
        signal_fft = fft(self.audio_data)
        signal_phase = np.angle(signal_fft)
        signal_power = np.abs(signal_fft)**2
        
        # Subtract noise from signal
        power_diff = signal_power - smoothing_factor * noise_power_mean
        
        # Apply minimum SNR to avoid musical noise
        mask = (power_diff / noise_power_mean) > min_snr
        power_diff[~mask] = min_snr * noise_power_mean
        
        # Ensure all values are non-negative
        power_diff = np.maximum(power_diff, 0)
        
        # Reconstruct signal
        magnitude = np.sqrt(power_diff)
        recovered_fft = magnitude * np.exp(1j * signal_phase)
        self.audio_data = np.real(ifft(recovered_fft))
        
        return self.audio_data
    
    def wiener_filter(self, noise_sample=None, noise_estimate=None):
        """
        Wiener filter for noise reduction.
        
        Args:
            noise_sample: Section of audio containing only noise
            noise_estimate: Pre-calculated noise power spectrum estimate
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
        
        # Get noise power spectrum
        if noise_estimate is not None:
            noise_power = noise_estimate
        elif noise_sample is not None:
            noise_fft = fft(noise_sample)
            noise_power = np.abs(noise_fft)**2
        else:
            raise ValueError("Either noise_sample or noise_estimate must be provided")
            
        # Process the audio with Wiener filter using scipy
        self.audio_data = signal.wiener(self.audio_data, noise=noise_power.mean())
        
        return self.audio_data
    
    def adaptive_noise_cancellation(self, reference_noise, step_size=0.01, filter_length=32):
        """
        Adaptive noise cancellation using LMS algorithm.
        
        Args:
            reference_noise: Reference noise signal (must be synchronized with main audio)
            step_size: Step size for LMS adaptation
            filter_length: Length of the adaptive filter
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
            
        if len(reference_noise) != len(self.audio_data):
            raise ValueError("Reference noise must have the same length as audio data")
        
        # Initialize filter weights
        weights = np.zeros(filter_length)
        y = np.zeros_like(self.audio_data)
        
        # LMS algorithm
        for i in range(filter_length, len(self.audio_data)):
            x_segment = reference_noise[i-filter_length:i]
            y[i] = np.dot(weights, x_segment)
            e = self.audio_data[i] - y[i]
            weights = weights + step_size * e * x_segment
        
        # The error signal e is our clean signal
        self.audio_data = self.audio_data - y
        
        return self.audio_data
    
    def time_frequency_masking(self, noise_sample=None, threshold=2.0):
        """
        Time-frequency masking for noise reduction.
        
        Args:
            noise_sample: Section of audio containing only noise
            threshold: Threshold for binary masking
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
        
        # Calculate STFT
        stft = librosa.stft(self.audio_data)
        mag = np.abs(stft)
        phase = np.angle(stft)
        
        # If noise sample is provided, use it for estimating noise profile
        if noise_sample is not None:
            noise_stft = librosa.stft(noise_sample)
            noise_mag = np.abs(noise_stft)
            noise_mean = np.mean(noise_mag, axis=1, keepdims=True)
        else:
            # Estimate noise from signal itself (first few frames)
            noise_mean = np.mean(mag[:, :5], axis=1, keepdims=True)
        
        # Create mask based on SNR
        mask = mag > (threshold * noise_mean)
        
        # Apply mask
        mag_masked = mag * mask
        
        # Reconstruct signal
        stft_reconstructed = mag_masked * np.exp(1j * phase)
        self.audio_data = librosa.istft(stft_reconstructed)
        
        return self.audio_data
    
    def noisereduce_library(self, noise_sample=None, prop_decrease=0.75, n_fft=2048):
        """
        Use the noisereduce library for noise reduction.
        
        Args:
            noise_sample: Section of audio containing only noise
            prop_decrease: Proportion to decrease noise by
            n_fft: FFT window size
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
        
        if noise_sample is not None:
            # Noise sample available - perform noise reduction with it
            self.audio_data = nr.reduce_noise(
                y=self.audio_data, 
                y_noise=noise_sample,
                sr=self.sample_rate,
                prop_decrease=prop_decrease,
                n_fft=n_fft
            )
        else:
            # No noise sample - use statistical method
            self.audio_data = nr.reduce_noise(
                y=self.audio_data,
                sr=self.sample_rate,
                prop_decrease=prop_decrease,
                n_fft=n_fft
            )
        
        return self.audio_data
    
    def median_filtering(self, window_size=3):
        """
        Apply median filtering to reduce impulsive noise.
        
        Args:
            window_size: Size of the median filter window
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
        
        self.audio_data = signal.medfilt(self.audio_data, window_size)
        return self.audio_data
    
    def butterworth_filter(self, lowcut=100, highcut=8000, order=5):
        """
        Apply Butterworth bandpass filter.
        
        Args:
            lowcut: Low frequency cutoff in Hz
            highcut: High frequency cutoff in Hz
            order: Filter order
        """
        if self.audio_data is None or self.sample_rate is None:
            raise ValueError("Please load audio data first")
        
        nyquist = 0.5 * self.sample_rate
        low = lowcut / nyquist
        high = highcut / nyquist
        
        b, a = signal.butter(order, [low, high], btype='band')
        self.audio_data = signal.filtfilt(b, a, self.audio_data)
        
        return self.audio_data
    
    def plot_spectrogram(self, title="Spectrogram", show_original=False):
        """Plot spectrogram of the audio data."""
        plt.figure(figsize=(12, 8))
        
        if show_original and self.original_audio is not None:
            plt.subplot(2, 1, 1)
            plt.title("Original Audio Spectrogram")
            plt.specgram(self.original_audio, Fs=self.sample_rate, NFFT=1024)
            plt.colorbar(format='%+2.0f dB')
            
            plt.subplot(2, 1, 2)
            plt.title("Processed Audio Spectrogram")
            plt.specgram(self.audio_data, Fs=self.sample_rate, NFFT=1024)
            plt.colorbar(format='%+2.0f dB')
        else:
            plt.title(title)
            plt.specgram(self.audio_data, Fs=self.sample_rate, NFFT=1024)
            plt.colorbar(format='%+2.0f dB')
            
        plt.tight_layout()
        plt.show()
    
    def evaluate_snr(self, clean_reference=None):
        """
        Calculate Signal-to-Noise Ratio improvement.
        
        Args:
            clean_reference: Clean reference signal without noise (ground truth)
        """
        if clean_reference is None and self.original_audio is None:
            raise ValueError("Either clean reference or original audio is required")
        
        if clean_reference is None:
            # If no clean reference, calculate improvement relative to original
            original_power = np.mean(self.original_audio**2)
            noise_power = np.mean((self.original_audio - self.audio_data)**2)
        else:
            # If clean reference available, calculate true SNR
            signal_power = np.mean(clean_reference**2)
            noise_power = np.mean((clean_reference - self.audio_data)**2)
            
        if noise_power == 0:
            return float('inf')  # Perfect noise cancellation
            
        snr = 10 * np.log10(signal_power / noise_power)
        return snr
    
    def pipeline(self, methods, **kwargs):
        """
        Apply a pipeline of noise reduction methods.
        
        Args:
            methods: List of method names to apply in sequence
            **kwargs: Arguments for each method
        """
        if self.audio_data is None:
            raise ValueError("Please load audio data first")
            
        for method in methods:
            if not hasattr(self, method):
                raise ValueError(f"Method {method} not found")
                
            # Extract arguments for this method
            method_args = {k.split('_', 1)[1]: v for k, v in kwargs.items() 
                           if k.startswith(f"{method}_")}
            
            # Call the method with the extracted arguments
            getattr(self, method)(**method_args)
            
        return self.audio_data


# Example usage:
def example_usage():
    # Create an instance of the noise removal system
    noise_remover = NoiseRemovalSystem()
    
    # Load audio file
    noise_remover.load_audio("noisy_speech.wav")
    
    # Extract noise sample (first 1 second assumes it's silence/noise)
    noise_sample = noise_remover.audio_data[:int(noise_remover.sample_rate)]
    
    # Method 1: Apply spectral subtraction
    # noise_remover.spectral_subtraction(noise_sample)
    
    # Method 2: Apply Wiener filter
    # noise_remover.wiener_filter(noise_sample=noise_sample)
    
    # Method 3: Use noisereduce library
    # noise_remover.noisereduce_library(noise_sample)
    
    # Method 4: Apply bandpass filter to remove out-of-band noise
    # noise_remover.butterworth_filter(lowcut=100, highcut=8000)
    
    # Pipeline approach - apply multiple methods in sequence
    noise_remover.pipeline(
        methods=['butterworth_filter', 'noisereduce_library', 'median_filtering'],
        butterworth_filter_lowcut=100,
        butterworth_filter_highcut=8000,
        noisereduce_library_noise_sample=noise_sample,
        noisereduce_library_prop_decrease=0.75,
        median_filtering_window_size=3
    )
    
    # Plot spectrograms to compare
    noise_remover.plot_spectrogram(show_original=True)
    
    # Save the processed audio
    noise_remover.save_audio("cleaned_speech.wav")
    
    # Print SNR improvement
    print(f"SNR improvement: {noise_remover.evaluate_snr()} dB")

# Uncomment to run the example
# example_usage()
