"""
Audio processing module for meeting transcription system.
Handles audio loading, preprocessing, and feature extraction.
"""

import librosa
import soundfile as sf
import numpy as np
from typing import Tuple, Optional, List
import webrtcvad
from pydub import AudioSegment
import logging

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Handles audio preprocessing and feature extraction."""
    
    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1024):
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.vad = webrtcvad.Vad(3)  # Aggressive mode
        
    def load_audio(self, file_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file and return waveform and sample rate."""
        try:
            # Load audio using librosa
            waveform, sr = librosa.load(file_path, sr=self.sample_rate)
            logger.info(f"Loaded audio: {file_path}, duration: {len(waveform)/sr:.2f}s")
            return waveform, sr
        except Exception as e:
            logger.error(f"Error loading audio {file_path}: {e}")
            raise
    
    def preprocess_audio(self, waveform: np.ndarray, sr: int) -> np.ndarray:
        """Preprocess audio: normalize, denoise, and filter."""
        # Normalize audio
        waveform = librosa.util.normalize(waveform)
        
        # Apply high-pass filter to remove low-frequency noise
        waveform = librosa.effects.preemphasis(waveform)
        
        # Trim silence
        waveform, _ = librosa.effects.trim(waveform, top_db=20)
        
        return waveform
    
    def extract_features(self, waveform: np.ndarray, sr: int) -> dict:
        """Extract audio features for analysis."""
        features = {}
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=waveform, sr=sr, n_mfcc=13)
        features['mfcc'] = mfccs
        
        # Spectral centroid
        spectral_centroid = librosa.feature.spectral_centroid(y=waveform, sr=sr)
        features['spectral_centroid'] = spectral_centroid
        
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(waveform)
        features['zcr'] = zcr
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=waveform, sr=sr)
        features['spectral_rolloff'] = spectral_rolloff
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        features['chroma'] = chroma
        
        return features
    
    def detect_voice_activity(self, waveform: np.ndarray, sr: int) -> List[Tuple[float, float]]:
        """Detect voice activity in audio."""
        # Convert to 16-bit PCM
        audio_int16 = (waveform * 32767).astype(np.int16)
        
        # Frame duration in ms
        frame_duration = 30
        frame_length = int(sr * frame_duration / 1000)
        
        voiced_frames = []
        timestamps = []
        
        for i in range(0, len(audio_int16), frame_length):
            frame = audio_int16[i:i+frame_length]
            
            # Pad if necessary
            if len(frame) < frame_length:
                frame = np.pad(frame, (0, frame_length - len(frame)), mode='constant')
            
            # Convert to bytes
            frame_bytes = frame.tobytes()
            
            # VAD detection
            is_speech = self.vad.is_speech(frame_bytes, sr)
            voiced_frames.append(is_speech)
            timestamps.append(i / sr)
        
        # Group consecutive voiced frames
        voice_segments = []
        start_time = None
        
        for i, (is_voiced, timestamp) in enumerate(zip(voiced_frames, timestamps)):
            if is_voiced and start_time is None:
                start_time = timestamp
            elif not is_voiced and start_time is not None:
                voice_segments.append((start_time, timestamp))
                start_time = None
        
        # Handle case where speech continues to end
        if start_time is not None:
            voice_segments.append((start_time, timestamps[-1]))
        
        return voice_segments
    
    def segment_audio(self, waveform: np.ndarray, sr: int, segments: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Segment audio based on provided time segments."""
        audio_segments = []
        
        for start_time, end_time in segments:
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            
            segment = waveform[start_sample:end_sample]
            if len(segment) > 0:
                audio_segments.append(segment)
        
        return audio_segments
    
    def save_audio(self, waveform: np.ndarray, sr: int, output_path: str):
        """Save audio waveform to file."""
        sf.write(output_path, waveform, sr)
        logger.info(f"Saved audio to {output_path}")
    
    def convert_audio_format(self, input_path: str, output_path: str, 
                           target_format: str = "wav") -> str:
        """Convert audio to target format."""
        try:
            audio = AudioSegment.from_file(input_path)
            
            # Convert to mono and target sample rate
            audio = audio.set_channels(1)
            audio = audio.set_frame_rate(self.sample_rate)
            
            # Export in target format
            audio.export(output_path, format=target_format)
            logger.info(f"Converted {input_path} to {output_path}")
            
            return output_path
            
        except Exception as e:
            logger.error(f"Error converting audio format: {e}")
            raise
    
    def get_audio_info(self, file_path: str) -> dict:
        """Get audio file information."""
        try:
            audio = AudioSegment.from_file(file_path)
            info = {
                'duration': len(audio) / 1000.0,  # seconds
                'channels': audio.channels,
                'sample_rate': audio.frame_rate,
                'frame_width': audio.frame_width,
                'format': file_path.split('.')[-1].lower()
            }
            return info
        except Exception as e:
            logger.error(f"Error getting audio info: {e}")
            return {}


class AudioBuffer:
    """Real-time audio buffer for streaming audio processing."""
    
    def __init__(self, sample_rate: int = 16000, buffer_duration: float = 2.0):
        self.sample_rate = sample_rate
        self.buffer_size = int(sample_rate * buffer_duration)
        self.buffer = np.zeros(self.buffer_size)
        self.write_pos = 0
        
    def add_audio(self, audio_chunk: np.ndarray):
        """Add audio chunk to buffer."""
        chunk_size = len(audio_chunk)
        
        if self.write_pos + chunk_size <= self.buffer_size:
            self.buffer[self.write_pos:self.write_pos + chunk_size] = audio_chunk
            self.write_pos += chunk_size
        else:
            # Wrap around buffer
            remaining = self.buffer_size - self.write_pos
            self.buffer[self.write_pos:] = audio_chunk[:remaining]
            self.buffer[:chunk_size - remaining] = audio_chunk[remaining:]
            self.write_pos = chunk_size - remaining
    
    def get_buffer(self) -> np.ndarray:
        """Get current buffer contents."""
        if self.write_pos == 0:
            return self.buffer
        else:
            return np.concatenate([
                self.buffer[self.write_pos:],
                self.buffer[:self.write_pos]
            ])
    
    def clear(self):
        """Clear the buffer."""
        self.buffer.fill(0)
        self.write_pos = 0
