"""
Main pipeline for meeting transcription system.
Orchestrates the complete workflow from audio to analyzed transcript.
"""
import os
import json
import yaml
import logging
from typing import Dict, List, Optional
from datetime import datetime
import argparse
from audio_processor import AudioProcessor
from speaker_diarization import SpeakerDiarizationPipeline
from speech_recognition import SpeechRecognitionPipeline, TranscriptionPostProcessor
from nlp_processor import NLPProcessor, MeetingAnalyzer, TranscriptFormatter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MeetingTranscriptionPipeline:
    """Complete meeting transcription pipeline."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.audio_processor = None
        self.diarization_pipeline = None
        self.speech_recognition_pipeline = None
        self.nlp_processor = None
        self.meeting_analyzer = None
        self.transcript_formatter = None
        self.post_processor = None
        
        self._initialize_components()
    
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration from YAML file."""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            raise
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Audio processor
            self.audio_processor = AudioProcessor(
                sample_rate=self.config['audio']['sample_rate'],
                chunk_size=self.config['audio']['chunk_size']
            )
            
            # Speaker diarization
            self.diarization_pipeline = SpeakerDiarizationPipeline(
                self.config['diarization']
            )
            
            # Speech recognition
            self.speech_recognition_pipeline = SpeechRecognitionPipeline(
                self.config['speech_recognition']
            )
            
            # NLP processor
            self.nlp_processor = NLPProcessor(
                self.config['nlp']['model_name']
            )
            
            # Meeting analyzer
            self.meeting_analyzer = MeetingAnalyzer(self.nlp_processor)
            
            # Transcript formatter
            self.transcript_formatter = TranscriptFormatter()
            
            # Post processor
            self.post_processor = TranscriptionPostProcessor()
            
            logger.info("All pipeline components initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing components: {e}")
            raise
    
    def process_audio_file(self, audio_file: str, output_dir: str = None) -> Dict:
        """Process a single audio file through the complete pipeline."""
        logger.info(f"Starting transcription pipeline for: {audio_file}")
        
        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        if output_dir is None:
            output_dir = os.path.join(os.path.dirname(audio_file), "transcripts")
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate output filename
        base_name = os.path.splitext(os.path.basename(audio_file))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"{base_name}_{timestamp}.json")
        
        try:
            # Step 1: Audio preprocessing
            logger.info("Step 1: Processing audio file")
            processed_audio = self.audio_processor.process_file(audio_file)
            
            # Step 2: Speaker diarization
            logger.info("Step 2: Performing speaker diarization")
            speaker_segments = self.diarization_pipeline.diarize(processed_audio)
            
            # Step 3: Speech recognition
            logger.info("Step 3: Performing speech recognition")
            transcription_results = self.speech_recognition_pipeline.transcribe_segments(
                processed_audio, speaker_segments
            )
            
            # Step 4: Post-process transcription
            logger.info("Step 4: Post-processing transcription")
            cleaned_transcription = self.post_processor.process(transcription_results)
            
            # Step 5: NLP analysis
            logger.info("Step 5: Analyzing transcript with NLP")
            nlp_analysis = self.nlp_processor.analyze_transcript(cleaned_transcription)
            
            # Step 6: Meeting analysis
            logger.info("Step 6: Performing meeting analysis")
            meeting_analysis = self.meeting_analyzer.analyze_meeting(
                cleaned_transcription, nlp_analysis
            )
            
            # Step 7: Format final transcript
            logger.info("Step 7: Formatting final transcript")
            formatted_transcript = self.transcript_formatter.format_transcript(
                cleaned_transcription, speaker_segments, meeting_analysis
            )
            
            # Compile results
            results = {
                "metadata": {
                    "audio_file": audio_file,
                    "processed_at": datetime.now().isoformat(),
                    "pipeline_version": "1.0.0",
                    "duration_seconds": processed_audio.get("duration", 0),
                    "speakers_detected": len(set(seg["speaker"] for seg in speaker_segments))
                },
                "transcript": formatted_transcript,
                "speaker_segments": speaker_segments,
                "nlp_analysis": nlp_analysis,
                "meeting_analysis": meeting_analysis,
                "processing_stats": {
                    "audio_processing_time": processed_audio.get("processing_time", 0),
                    "diarization_time": speaker_segments.get("processing_time", 0),
                    "transcription_time": transcription_results.get("processing_time", 0),
                    "nlp_processing_time": nlp_analysis.get("processing_time", 0)
                }
            }
            
            # Save results
            self._save_results(results, output_file)
            
            logger.info(f"Transcription pipeline completed successfully. Results saved to: {output_file}")
            return results
            
        except Exception as e:
            logger.error(f"Error in transcription pipeline: {e}")
            raise
    
    def process_batch(self, audio_files: List[str], output_dir: str = None) -> List[Dict]:
        """Process multiple audio files."""
        logger.info(f"Starting batch processing for {len(audio_files)} files")
        
        results = []
        failed_files = []
        
        for i, audio_file in enumerate(audio_files, 1):
            logger.info(f"Processing file {i}/{len(audio_files)}: {audio_file}")
            
            try:
                result = self.process_audio_file(audio_file, output_dir)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process {audio_file}: {e}")
                failed_files.append({"file": audio_file, "error": str(e)})
        
        # Save batch summary
        if output_dir:
            batch_summary = {
                "processed_at": datetime.now().isoformat(),
                "total_files": len(audio_files),
                "successful": len(results),
                "failed": len(failed_files),
                "failed_files": failed_files
            }
            
            summary_file = os.path.join(output_dir, f"batch_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(summary_file, 'w') as f:
                json.dump(batch_summary, f, indent=2)
            
            logger.info(f"Batch processing completed. Summary saved to: {summary_file}")
        
        return results
    
    def _save_results(self, results: Dict, output_file: str):
        """Save results to JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Error saving results: {e}")
            raise
    
    def get_pipeline_status(self) -> Dict:
        """Get status of all pipeline components."""
        status = {
            "audio_processor": self.audio_processor is not None,
            "diarization_pipeline": self.diarization_pipeline is not None,
            "speech_recognition_pipeline": self.speech_recognition_pipeline is not None,
            "nlp_processor": self.nlp_processor is not None,
            "meeting_analyzer": self.meeting_analyzer is not None,
            "transcript_formatter": self.transcript_formatter is not None,
            "post_processor": self.post_processor is not None
        }
        return status
    
    def update_config(self, new_config: Dict):
        """Update configuration and reinitialize components."""
        logger.info("Updating configuration")
        self.config.update(new_config)
        self._initialize_components()
        logger.info("Configuration updated and components reinitialized")


def main():
    """Main function for command line interface."""
    parser = argparse.ArgumentParser(description="Meeting Transcription Pipeline")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    parser.add_argument("--audio", required=True, help="Audio file or directory path")
    parser.add_argument("--output", help="Output directory path")
    parser.add_argument("--batch", action="store_true", help="Process multiple files in directory")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # Initialize pipeline
        pipeline = MeetingTranscriptionPipeline(args.config)
        
        if args.batch:
            # Process directory of audio files
            if not os.path.isdir(args.audio):
                raise ValueError("Audio path must be a directory when using --batch")
            
            audio_files = []
            for ext in ['.wav', '.mp3', '.mp4', '.m4a', '.flac']:
                audio_files.extend(
                    [os.path.join(args.audio, f) for f in os.listdir(args.audio) 
                     if f.lower().endswith(ext)]
                )
            
            if not audio_files:
                raise ValueError("No audio files found in directory")
            
            logger.info(f"Found {len(audio_files)} audio files")
            results = pipeline.process_batch(audio_files, args.output)
            
        else:
            # Process single audio file
            results = pipeline.process_audio_file(args.audio, args.output)
        
        logger.info("Pipeline execution completed successfully")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        raise


if __name__ == "__main__":
    main()
