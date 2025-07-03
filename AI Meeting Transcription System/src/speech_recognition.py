"""
Speech recognition module for meeting transcription system.
Converts speech to text using state-of-the-art models.
"""

import torch
import whisper
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from transformers import pipeline, AutoProcessor, AutoModelForSpeechSeq2Seq
from faster_whisper import WhisperModel
import librosa
import soundfile as sf
import tempfile
import os

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Whisper-based speech recognition."""
    
    def __init__(self, model_name: str = "large-v3", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Whisper model."""
        try:
            self.model = whisper.load_model(self.model_name, device=self.device)
            logger.info(f"Loaded Whisper model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading Whisper model: {e}")
            raise
    
    def transcribe(self, audio_file: str, language: str = "en") -> Dict:
        """Transcribe audio file."""
        try:
            result = self.model.transcribe(
                audio_file,
                language=language,
                word_timestamps=True,
                initial_prompt="This is a meeting recording with multiple speakers."
            )
            
            logger.info(f"Transcription completed for {audio_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise
    
    def transcribe_segment(self, audio_segment: np.ndarray, sr: int, 
                          language: str = "en") -> Dict:
        """Transcribe audio segment."""
        try:
            # Save segment to temporary file
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_segment, sr)
                
                # Transcribe
                result = self.model.transcribe(
                    tmp_file.name,
                    language=language,
                    word_timestamps=True
                )
                
                # Clean up
                os.unlink(tmp_file.name)
                
                return result
                
        except Exception as e:
            logger.error(f"Error transcribing segment: {e}")
            raise


class FasterWhisperTranscriber:
    """Faster Whisper implementation for better performance."""
    
    def __init__(self, model_name: str = "large-v3", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu",
                 compute_type: str = "float16"):
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load Faster Whisper model."""
        try:
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info(f"Loaded Faster Whisper model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading Faster Whisper model: {e}")
            raise
    
    def transcribe(self, audio_file: str, language: str = "en") -> Dict:
        """Transcribe audio file."""
        try:
            segments, info = self.model.transcribe(
                audio_file,
                language=language,
                word_timestamps=True,
                initial_prompt="This is a meeting recording with multiple speakers."
            )
            
            # Convert segments to list
            segments_list = list(segments)
            
            # Format result similar to Whisper
            result = {
                'text': ' '.join([segment.text for segment in segments_list]),
                'segments': [
                    {
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text,
                        'words': [
                            {
                                'word': word.word,
                                'start': word.start,
                                'end': word.end,
                                'probability': word.probability
                            } for word in segment.words
                        ] if segment.words else []
                    } for segment in segments_list
                ],
                'language': info.language
            }
            
            logger.info(f"Transcription completed for {audio_file}")
            return result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise


class HuggingFaceTranscriber:
    """HuggingFace transformers-based speech recognition."""
    
    def __init__(self, model_name: str = "openai/whisper-large-v3"):
        self.model_name = model_name
        self.pipe = None
        self._load_model()
    
    def _load_model(self):
        """Load HuggingFace model."""
        try:
            self.pipe = pipeline(
                "automatic-speech-recognition",
                model=self.model_name,
                return_timestamps=True,
                device=0 if torch.cuda.is_available() else -1
            )
            logger.info(f"Loaded HuggingFace model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading HuggingFace model: {e}")
            raise
    
    def transcribe(self, audio_file: str) -> Dict:
        """Transcribe audio file."""
        try:
            result = self.pipe(audio_file)
            
            # Format result
            formatted_result = {
                'text': result['text'],
                'chunks': result.get('chunks', [])
            }
            
            logger.info(f"Transcription completed for {audio_file}")
            return formatted_result
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            raise


class SpeechRecognitionPipeline:
    """Complete speech recognition pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.transcriber = self._initialize_transcriber()
    
    def _initialize_transcriber(self):
        """Initialize transcriber based on config."""
        model_name = self.config.get('model_name', 'openai/whisper-large-v3')
        device = self.config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if 'faster-whisper' in model_name.lower():
            return FasterWhisperTranscriber(model_name, device)
        elif 'huggingface' in model_name.lower():
            return HuggingFaceTranscriber(model_name)
        else:
            return WhisperTranscriber(model_name, device)
    
    def transcribe_audio(self, audio_file: str, language: str = "en") -> Dict:
        """Transcribe complete audio file."""
        return self.transcriber.transcribe(audio_file, language)
    
    def transcribe_segments(self, audio_file: str, segments: List[Tuple[float, float]], 
                          language: str = "en") -> List[Dict]:
        """Transcribe specific audio segments."""
        results = []
        
        # Load audio
        waveform, sr = librosa.load(audio_file, sr=16000)
        
        for start_time, end_time in segments:
            # Extract segment
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            segment = waveform[start_sample:end_sample]
            
            # Transcribe segment
            if hasattr(self.transcriber, 'transcribe_segment'):
                result = self.transcriber.transcribe_segment(segment, sr, language)
            else:
                # Save segment and transcribe
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                    sf.write(tmp_file.name, segment, sr)
                    result = self.transcriber.transcribe(tmp_file.name, language)
                    os.unlink(tmp_file.name)
            
            # Add timing information
            result['segment_start'] = start_time
            result['segment_end'] = end_time
            results.append(result)
        
        return results
    
    def process_with_speaker_segments(self, audio_file: str, 
                                    speaker_segments: Dict[str, List[Tuple[float, float]]],
                                    language: str = "en") -> Dict:
        """Process audio with speaker segment information."""
        results = {
            'speakers': {},
            'timeline': []
        }
        
        # Process each speaker's segments
        for speaker, segments in speaker_segments.items():
            speaker_transcripts = []
            
            for segment in segments:
                transcript = self.transcribe_segments(audio_file, [segment], language)[0]
                speaker_transcripts.append({
                    'start': segment[0],
                    'end': segment[1],
                    'text': transcript.get('text', ''),
                    'confidence': self._calculate_confidence(transcript)
                })
            
            results['speakers'][speaker] = speaker_transcripts
        
        # Create timeline
        all_segments = []
        for speaker, transcripts in results['speakers'].items():
            for transcript in transcripts:
                all_segments.append({
                    'speaker': speaker,
                    'start': transcript['start'],
                    'end': transcript['end'],
                    'text': transcript['text'],
                    'confidence': transcript['confidence']
                })
        
        # Sort by start time
        results['timeline'] = sorted(all_segments, key=lambda x: x['start'])
        
        return results
    
    def _calculate_confidence(self, transcript: Dict) -> float:
        """Calculate average confidence score."""
        if 'segments' in transcript:
            confidences = []
            for segment in transcript['segments']:
                if 'words' in segment:
                    word_confidences = [word.get('probability', 0.5) 
                                      for word in segment['words']]
                    if word_confidences:
                        confidences.extend(word_confidences)
            
            return np.mean(confidences) if confidences else 0.5
        
        return 0.5  # Default confidence


class TranscriptionPostProcessor:
    """Post-process transcription results."""
    
    def __init__(self):
        self.punctuation_marks = ['.', '!', '?', ',', ';', ':']
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize transcribed text."""
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Fix common transcription errors
        text = text.replace(' um ', ' ')
        text = text.replace(' uh ', ' ')
        text = text.replace(' ah ', ' ')
        
        # Capitalize first letter of sentences
        sentences = text.split('.')
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                cleaned_sentences.append(sentence)
        
        return '. '.join(cleaned_sentences)
    
    def add_punctuation(self, text: str) -> str:
        """Add punctuation to text using heuristics."""
        # This is a simple implementation
        # In practice, you might want to use a specialized model
        
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            
            # Add comma after transition words
            if word.lower() in ['however', 'therefore', 'moreover', 'furthermore']:
                if i < len(words) - 1:
                    result.append(',')
            
            # Add period at end of sentences (heuristic)
            if i < len(words) - 1:
                next_word = words[i + 1]
                if next_word[0].isupper() and word[-1] not in self.punctuation_marks:
                    result.append('.')
        
        return ' '.join(result)
    
    def merge_overlapping_segments(self, segments: List[Dict], 
                                 overlap_threshold: float = 0.1) -> List[Dict]:
        """Merge overlapping transcription segments."""
        if not segments:
            return segments
        
        # Sort by start time
        segments = sorted(segments, key=lambda x: x['start'])
        merged = [segments[0]]
        
        for current in segments[1:]:
            last = merged[-1]
            
            # Check for overlap
            if current['start'] <= last['end'] + overlap_threshold:
                # Merge segments
                merged[-1] = {
                    'start': last['start'],
                    'end': max(last['end'], current['end']),
                    'text': last['text'] + ' ' + current['text'],
                    'speaker': last.get('speaker', current.get('speaker')),
                    'confidence': (last.get('confidence', 0.5) + current.get('confidence', 0.5)) / 2
                }
            else:
                merged.append(current)
        
        return merged
    
    def generate_summary(self, transcript: str, max_length: int = 200) -> str:
        """Generate a summary of the transcript."""
        sentences = transcript.split('.')
        if len(sentences) <= 3:
            return transcript
        
        # Simple extractive summarization
        # Take first, middle, and last sentences
        summary_sentences = [
            sentences[0],
            sentences[len(sentences) // 2],
            sentences[-1]
        ]
        
        summary = '. '.join(sentence.strip() for sentence in summary_sentences if sentence.strip())
        
        # Truncate if too long
        if len(summary) > max_length:
            summary = summary[:max_length].rsplit(' ', 1)[0] + '...'
        
        return summary
