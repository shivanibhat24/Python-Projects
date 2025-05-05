"""
Twitter Spaces to Podcast Summaries
-----------------------------------
This script converts Twitter Spaces recordings into concise podcast summaries with highlight reels.
It uses Tweepy to access Twitter API, OpenAI's Whisper for speech-to-text transcription,
spaCy for natural language processing to extract key points, and a TTS engine to generate audio summaries.
"""

import os
import json
import time
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import tweepy
import whisper
import spacy
import numpy as np
from pydub import AudioSegment
import torch
from TTS.api import TTS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("twitter_spaces_podcast")

class TwitterSpacesProcessor:
    """Processes Twitter Spaces recordings and converts them to podcast summaries."""
    
    def __init__(self, config_path: str):
        """
        Initialize the Twitter Spaces processor.
        
        Args:
            config_path: Path to the configuration file containing API keys and settings.
        """
        self.config = self._load_config(config_path)
        self.twitter_client = self._setup_twitter_client()
        self.nlp = spacy.load("en_core_web_lg")  # Load a more comprehensive spaCy model
        self.transcription_model = whisper.load_model("medium")  # medium model for better accuracy
        self.tts_engine = self._setup_tts_engine()
        
        # Create output directories if they don't exist
        os.makedirs(self.config["output_dir"], exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "audio"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "transcripts"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "summaries"), exist_ok=True)
        os.makedirs(os.path.join(self.config["output_dir"], "highlights"), exist_ok=True)
        
    def _load_config(self, config_path: str) -> dict:
        """
        Load configuration from JSON file.
        
        Args:
            config_path: Path to the configuration file.
            
        Returns:
            Dictionary containing configuration parameters.
        """
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            # Validate required configuration fields
            required_fields = [
                "twitter_api_key", "twitter_api_secret", 
                "twitter_access_token", "twitter_access_token_secret",
                "output_dir", "min_highlight_duration", "max_highlight_duration",
                "summary_length", "num_highlights"
            ]
            
            for field in required_fields:
                if field not in config:
                    raise ValueError(f"Missing required configuration field: {field}")
                    
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
            
    def _setup_twitter_client(self) -> tweepy.API:
        """
        Set up Twitter API client using tweepy.
        
        Returns:
            Authenticated tweepy API client.
        """
        try:
            auth = tweepy.OAuth1UserHandler(
                self.config["twitter_api_key"],
                self.config["twitter_api_secret"],
                self.config["twitter_access_token"],
                self.config["twitter_access_token_secret"]
            )
            
            client = tweepy.API(auth)
            client.verify_credentials()
            logger.info("Twitter API authentication successful")
            return client
        except Exception as e:
            logger.error(f"Twitter API authentication failed: {e}")
            raise
            
    def _setup_tts_engine(self) -> TTS:
        """
        Set up text-to-speech engine.
        
        Returns:
            TTS engine instance.
        """
        try:
            # Use a high-quality TTS model 
            tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC")
            return tts
        except Exception as e:
            logger.error(f"Failed to initialize TTS engine: {e}")
            raise
    
    def get_twitter_spaces(self, user_id: Optional[str] = None, space_id: Optional[str] = None) -> List[Dict]:
        """
        Fetch Twitter Spaces data using the Twitter API.
        
        Args:
            user_id: Twitter user ID to fetch spaces from (optional).
            space_id: Specific space ID to fetch (optional).
            
        Returns:
            List of Twitter Spaces data.
        """
        spaces = []
        try:
            if space_id:
                # Get a specific Space by ID
                space = self.twitter_client.get_space(space_id=space_id)
                spaces.append(space.data)
                logger.info(f"Retrieved Space: {space.data.title}")
            elif user_id:
                # Get Spaces created by a specific user
                user_spaces = self.twitter_client.get_spaces(user_ids=[user_id], state="all")
                for space in user_spaces.data:
                    spaces.append(space)
                    logger.info(f"Retrieved Space: {space.title}")
            else:
                # Get currently active Spaces
                active_spaces = self.twitter_client.get_spaces(state="live")
                for space in active_spaces.data:
                    spaces.append(space)
                    logger.info(f"Retrieved active Space: {space.title}")
                    
            return spaces
        except Exception as e:
            logger.error(f"Failed to fetch Twitter Spaces: {e}")
            return []
    
    def download_space_audio(self, space_id: str) -> str:
        """
        Download the audio recording of a Twitter Space.
        
        Args:
            space_id: ID of the Twitter Space.
            
        Returns:
            Path to the downloaded audio file.
        """
        try:
            # In a real implementation, you would use the Twitter API to get the audio URL
            # This is a simplified version as Twitter's API for this is subject to change
            space_data = self.twitter_client.get_space(space_id=space_id)
            
            # For now, let's assume we have a URL to the audio recording
            # In practice, you would need to implement this based on Twitter's API
            audio_url = space_data.data.audio_url
            
            output_path = os.path.join(self.config["output_dir"], "audio", f"{space_id}.mp3")
            
            # Download the audio file
            # In a real implementation, you would use requests or similar to download the file
            # For now, we'll simulate this process
            logger.info(f"Downloading audio for Space {space_id} to {output_path}")
            
            return output_path
        except Exception as e:
            logger.error(f"Failed to download Space audio: {e}")
            raise
    
    def transcribe_audio(self, audio_path: str) -> Dict:
        """
        Transcribe an audio file using OpenAI's Whisper.
        
        Args:
            audio_path: Path to the audio file.
            
        Returns:
            Dictionary containing the transcription with timestamps.
        """
        try:
            logger.info(f"Transcribing audio: {audio_path}")
            
            # Perform transcription with word-level timestamps
            result = self.transcription_model.transcribe(
                audio_path, 
                language="en", 
                word_timestamps=True,
                verbose=False
            )
            
            # Process and structure the transcription
            transcript = {
                "text": result["text"],
                "segments": []
            }
            
            # Process segments with speaker identification (if available)
            for segment in result["segments"]:
                transcript_segment = {
                    "id": segment["id"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "words": segment.get("words", [])
                }
                transcript["segments"].append(transcript_segment)
            
            # Save transcript to file
            output_path = os.path.join(
                self.config["output_dir"], 
                "transcripts", 
                f"{os.path.basename(audio_path).split('.')[0]}.json"
            )
            
            with open(output_path, 'w') as f:
                json.dump(transcript, f, indent=2)
                
            logger.info(f"Transcription saved to {output_path}")
            return transcript
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise
    
    def identify_speakers(self, transcript: Dict) -> Dict:
        """
        Identify different speakers in the transcript using NLP techniques.
        
        Args:
            transcript: Transcription dictionary.
            
        Returns:
            Transcript with speaker labels.
        """
        # In a real implementation, you would use a more sophisticated speaker diarization model
        # Here we use a simplified approach based on timing and content
        
        # Clone the transcript
        enhanced_transcript = transcript.copy()
        enhanced_transcript["segments"] = []
        
        current_speaker = 0
        speaker_change_threshold = 1.5  # seconds of silence that might indicate speaker change
        
        for i, segment in enumerate(transcript["segments"]):
            # Check for potential speaker change
            if i > 0:
                prev_segment = transcript["segments"][i-1]
                gap = segment["start"] - prev_segment["end"]
                
                # Check for significant pause or change in speaking style
                if gap > speaker_change_threshold:
                    current_speaker = (current_speaker + 1) % 5  # Rotate between 5 potential speakers
            
            # Add speaker information
            enhanced_segment = segment.copy()
            enhanced_segment["speaker"] = f"Speaker_{current_speaker}"
            enhanced_transcript["segments"].append(enhanced_segment)
        
        return enhanced_transcript
    
    def extract_key_points(self, transcript: Dict) -> List[Dict]:
        """
        Extract key points from the transcript using NLP.
        
        Args:
            transcript: Transcription dictionary.
            
        Returns:
            List of key points with timestamps.
        """
        key_points = []
        
        # Combine segments into larger chunks for better context
        full_text = " ".join([segment["text"] for segment in transcript["segments"]])
        doc = self.nlp(full_text)
        
        # Extract sentences that contain important entities and high-information content
        important_sentences = []
        
        for sent in doc.sents:
            # Check if sentence contains named entities
            has_entities = any(ent.label_ in ["PERSON", "ORG", "GPE", "PRODUCT", "EVENT"] for ent in sent.ents)
            
            # Check if sentence contains important keywords
            # This is a simple approach; in practice, you might use TF-IDF or other techniques
            important_keywords = ["announce", "launch", "important", "significant", "breakthrough", "new"]
            has_keywords = any(token.lemma_ in important_keywords for token in sent)
            
            # Score the sentence based on various factors
            score = 0
            if has_entities:
                score += 2
            if has_keywords:
                score += 1
            if len(sent) > 10:  # Longer sentences might contain more information
                score += 1
            
            if score >= 2:  # Threshold for importance
                important_sentences.append((sent.text, score))
        
        # Sort by score and take top N
        important_sentences.sort(key=lambda x: x[1], reverse=True)
        top_sentences = important_sentences[:self.config["summary_length"]]
        
        # Find the corresponding timestamps for each important sentence
        for sentence_text, score in top_sentences:
            # Find which segment(s) contain this sentence
            for segment in transcript["segments"]:
                if sentence_text in segment["text"]:
                    key_point = {
                        "text": sentence_text,
                        "start": segment["start"],
                        "end": segment["end"],
                        "score": score,
                        "speaker": segment.get("speaker", "Unknown")
                    }
                    key_points.append(key_point)
                    break
        
        return key_points
    
    def generate_summary(self, key_points: List[Dict]) -> str:
        """
        Generate a textual summary from key points.
        
        Args:
            key_points: List of key points with timestamps.
            
        Returns:
            Summary text.
        """
        # Sort key points by score
        sorted_points = sorted(key_points, key=lambda x: x["score"], reverse=True)
        
        # Take top N points and sort them by timestamp
        top_points = sorted(sorted_points[:self.config["summary_length"]], key=lambda x: x["start"])
        
        # Generate a coherent summary
        summary_parts = []
        
        for i, point in enumerate(top_points):
            speaker = point.get("speaker", "The speaker")
            text = point["text"].strip()
            
            # Format the point
            if i == 0:
                summary_parts.append(f"{speaker} started by discussing {text}")
            elif i == len(top_points) - 1:
                summary_parts.append(f"Finally, {speaker} concluded with {text}")
            else:
                summary_parts.append(f"{speaker} then mentioned {text}")
        
        summary = " ".join(summary_parts)
        return summary
    
    def extract_highlights(self, transcript: Dict, key_points: List[Dict]) -> List[Dict]:
        """
        Extract highlight segments from the transcript.
        
        Args:
            transcript: Transcription dictionary.
            key_points: List of key points.
            
        Returns:
            List of highlight segments with start and end times.
        """
        highlights = []
        min_duration = self.config["min_highlight_duration"]
        max_duration = self.config["max_highlight_duration"]
        
        # Use key points as the basis for highlights
        for point in key_points:
            start_time = point["start"]
            
            # Extend the highlight to include context
            # Find the segment containing this start time
            containing_segments = [
                s for s in transcript["segments"] 
                if s["start"] <= start_time and s["end"] >= start_time
            ]
            
            if containing_segments:
                segment = containing_segments[0]
                highlight_start = max(segment["start"] - 2, 0)  # 2 seconds before for context
                highlight_end = min(segment["end"] + 2, segment["end"] + 5)  # 2-5 seconds after for context
                
                # Ensure minimum and maximum duration
                duration = highlight_end - highlight_start
                if duration < min_duration:
                    highlight_end = highlight_start + min_duration
                if duration > max_duration:
                    highlight_end = highlight_start + max_duration
                
                highlight = {
                    "start": highlight_start,
                    "end": highlight_end,
                    "text": segment["text"],
                    "speaker": segment.get("speaker", "Unknown")
                }
                
                highlights.append(highlight)
        
        # Sort by start time and limit to requested number
        highlights.sort(key=lambda x: x["start"])
        highlights = highlights[:self.config["num_highlights"]]
        
        return highlights
    
    def create_highlight_reel(self, audio_path: str, highlights: List[Dict], space_id: str) -> str:
        """
        Create a highlight reel audio file from the original recording.
        
        Args:
            audio_path: Path to the original audio file.
            highlights: List of highlight segments.
            space_id: ID of the Twitter Space.
            
        Returns:
            Path to the highlight reel audio file.
        """
        try:
            logger.info(f"Creating highlight reel for Space {space_id}")
            
            # Load the original audio
            original_audio = AudioSegment.from_file(audio_path)
            
            # Create a new audio file with highlights
            highlight_reel = AudioSegment.silent(duration=500)  # Start with a short silence
            
            # Add an intro
            intro_text = f"Highlights from Twitter Space number {space_id}"
            intro_path = os.path.join(self.config["output_dir"], "temp_intro.wav")
            self.tts_engine.tts_to_file(text=intro_text, file_path=intro_path)
            intro_audio = AudioSegment.from_file(intro_path)
            highlight_reel += intro_audio + AudioSegment.silent(duration=1000)
            
            # Add each highlight
            for i, highlight in enumerate(highlights):
                # Extract the highlight segment
                start_ms = int(highlight["start"] * 1000)
                end_ms = int(highlight["end"] * 1000)
                segment = original_audio[start_ms:end_ms]
                
                # Add a short introduction for each highlight
                highlight_intro_text = f"Highlight {i+1}: {highlight['speaker']}"
                highlight_intro_path = os.path.join(self.config["output_dir"], f"temp_highlight_intro_{i}.wav")
                self.tts_engine.tts_to_file(text=highlight_intro_text, file_path=highlight_intro_path)
                highlight_intro = AudioSegment.from_file(highlight_intro_path)
                
                # Combine intro and highlight
                highlight_reel += highlight_intro + AudioSegment.silent(duration=500) + segment + AudioSegment.silent(duration=1000)
                
                # Clean up temporary file
                os.remove(highlight_intro_path)
            
            # Clean up intro file
            os.remove(intro_path)
            
            # Save the highlight reel
            output_path = os.path.join(self.config["output_dir"], "highlights", f"{space_id}_highlights.mp3")
            highlight_reel.export(output_path, format="mp3")
            
            logger.info(f"Highlight reel saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create highlight reel: {e}")
            raise
    
    def create_summary_audio(self, summary: str, space_id: str) -> str:
        """
        Create an audio summary using TTS.
        
        Args:
            summary: Summary text.
            space_id: ID of the Twitter Space.
            
        Returns:
            Path to the summary audio file.
        """
        try:
            logger.info(f"Creating audio summary for Space {space_id}")
            
            # Generate audio from the summary text
            output_path = os.path.join(self.config["output_dir"], "summaries", f"{space_id}_summary.wav")
            self.tts_engine.tts_to_file(text=summary, file_path=output_path)
            
            logger.info(f"Audio summary saved to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to create audio summary: {e}")
            raise
    
    def process_space(self, space_id: str) -> Dict:
        """
        Process a Twitter Space from start to finish.
        
        Args:
            space_id: ID of the Twitter Space.
            
        Returns:
            Dictionary with paths to all generated files.
        """
        try:
            logger.info(f"Processing Twitter Space {space_id}")
            
            # Download the audio
            audio_path = self.download_space_audio(space_id)
            
            # Transcribe the audio
            transcript = self.transcribe_audio(audio_path)
            
            # Identify speakers
            enhanced_transcript = self.identify_speakers(transcript)
            
            # Extract key points
            key_points = self.extract_key_points(enhanced_transcript)
            
            # Generate text summary
            summary = self.generate_summary(key_points)
            
            # Save the summary text
            summary_text_path = os.path.join(self.config["output_dir"], "summaries", f"{space_id}_summary.txt")
            with open(summary_text_path, 'w') as f:
                f.write(summary)
            
            # Extract highlights
            highlights = self.extract_highlights(enhanced_transcript, key_points)
            
            # Create highlight reel
            highlight_reel_path = self.create_highlight_reel(audio_path, highlights, space_id)
            
            # Create audio summary
            summary_audio_path = self.create_summary_audio(summary, space_id)
            
            # Return paths to all generated files
            result = {
                "space_id": space_id,
                "original_audio": audio_path,
                "transcript": os.path.join(self.config["output_dir"], "transcripts", f"{space_id}.json"),
                "text_summary": summary_text_path,
                "audio_summary": summary_audio_path,
                "highlight_reel": highlight_reel_path
            }
            
            logger.info(f"Successfully processed Twitter Space {space_id}")
            return result
        except Exception as e:
            logger.error(f"Failed to process Twitter Space {space_id}: {e}")
            raise

def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description='Process Twitter Spaces into podcast summaries.')
    parser.add_argument('--config', type=str, required=True, help='Path to configuration file')
    parser.add_argument('--space-id', type=str, help='ID of the Twitter Space to process')
    parser.add_argument('--user-id', type=str, help='ID of the Twitter user to fetch spaces from')
    
    args = parser.parse_args()
    
    try:
        processor = TwitterSpacesProcessor(args.config)
        
        if args.space_id:
            # Process a specific Space
            result = processor.process_space(args.space_id)
            print(f"Processing complete. Results:")
            print(json.dumps(result, indent=2))
        elif args.user_id:
            # Process all Spaces from a user
            spaces = processor.get_twitter_spaces(user_id=args.user_id)
            results = []
            
            for space in spaces:
                result = processor.process_space(space.id)
                results.append(result)
            
            print(f"Processed {len(results)} Spaces. Results:")
            print(json.dumps(results, indent=2))
        else:
            # Process currently active Spaces
            spaces = processor.get_twitter_spaces()
            results = []
            
            for space in spaces:
                result = processor.process_space(space.id)
                results.append(result)
            
            print(f"Processed {len(results)} Spaces. Results:")
            print(json.dumps(results, indent=2))
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
