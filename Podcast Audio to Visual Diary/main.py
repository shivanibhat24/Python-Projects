import os
import sys
import numpy as np
import pandas as pd
import librosa
import torch
import matplotlib.pyplot as plt
from PIL import Image
from datetime import datetime
from tqdm import tqdm
import argparse
import logging
import json
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from diffusers import StableDiffusionPipeline
import whisper
from TTS.api import TTS

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("podcast_diary.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PodcastVisualDiaryGenerator:
    """
    A class to generate visual diaries from podcast audio by analyzing voice tone
    and extracting emotions.
    """
    
    def __init__(self, config=None):
        """
        Initialize the Podcast Visual Diary Generator.
        
        Args:
            config (dict, optional): Configuration parameters.
        """
        # Default configuration
        self.config = {
            'segment_length': 30,  # in seconds
            'emotion_threshold': 0.6,
            'output_dir': 'visual_diary_output',
            'use_gpu': torch.cuda.is_available(),
            'image_size': 512,
            'image_format': 'png',
            'voice_features': ['pitch', 'intensity', 'speech_rate', 'energy'],
            'emotions': ['neutral', 'happy', 'sad', 'angry', 'fearful', 'surprise', 'disgust']
        }
        
        # Update with custom config if provided
        if config:
            self.config.update(config)
        
        # Create output directory if it doesn't exist
        os.makedirs(self.config['output_dir'], exist_ok=True)
        
        # Initialize models
        self._init_models()
        
        logger.info("Podcast Visual Diary Generator initialized successfully")
    
    def _init_models(self):
        """Initialize all required models."""
        logger.info("Initializing models...")
        
        # Device configuration
        self.device = "cuda" if self.config['use_gpu'] and torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Initialize Whisper for speech-to-text
        logger.info("Loading Whisper model for transcription...")
        self.whisper_model = whisper.load_model("base")
        
        # Initialize emotion classifier
        logger.info("Loading emotion classifier...")
        self.emotion_classifier = pipeline(
            "text-classification", 
            model="j-hartmann/emotion-english-distilroberta-base",
            device=0 if self.device == "cuda" else -1
        )
        
        # Initialize stable diffusion for image generation
        logger.info("Loading Stable Diffusion model...")
        self.image_generator = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
        )
        if self.device == "cuda":
            self.image_generator = self.image_generator.to(self.device)
        
        # Initialize TTS model
        logger.info("Loading TTS model...")
        self.tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DDC", progress_bar=False)
        
        logger.info("All models initialized successfully")
    
    def _extract_audio_features(self, audio_path, segment_length=30):
        """
        Extract audio features using librosa.
        
        Args:
            audio_path (str): Path to the audio file.
            segment_length (int): Length of each segment in seconds.
            
        Returns:
            list: A list of dictionaries containing segment features.
        """
        logger.info(f"Extracting audio features from {audio_path}")
        
        # Load audio
        y, sr = librosa.load(audio_path, sr=None)
        
        # Convert segment length to samples
        segment_samples = segment_length * sr
        
        segments = []
        
        # Process each segment
        for i in range(0, len(y), segment_samples):
            if i + segment_samples > len(y):
                # Skip incomplete segments or pad them
                continue
                
            segment = y[i:i+segment_samples]
            segment_start_time = i / sr
            
            # Extract features for the segment
            features = {
                'start_time': segment_start_time,
                'end_time': segment_start_time + segment_length,
                'pitch': np.mean(librosa.yin(segment, fmin=75, fmax=600)),
                'intensity': np.mean(librosa.feature.rms(y=segment)[0]),
                'speech_rate': len(librosa.onset.onset_detect(y=segment, sr=sr)) / segment_length,
                'energy': np.mean(librosa.feature.spectral_contrast(y=segment, sr=sr)[0])
            }
            
            segments.append(features)
        
        logger.info(f"Extracted features for {len(segments)} segments")
        return segments
    
    def _transcribe_audio(self, audio_path):
        """
        Transcribe the audio using Whisper.
        
        Args:
            audio_path (str): Path to the audio file.
            
        Returns:
            dict: Transcription result containing 'text' and 'segments'.
        """
        logger.info(f"Transcribing audio: {audio_path}")
        try:
            result = self.whisper_model.transcribe(audio_path)
            logger.info(f"Transcription successful: {len(result['text'])} characters")
            return result
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            return {"text": "", "segments": []}
    
    def _analyze_emotions(self, text):
        """
        Analyze emotions in the text.
        
        Args:
            text (str): Text to analyze.
            
        Returns:
            dict: Emotion scores.
        """
        if not text:
            return {"label": "neutral", "score": 1.0}
        
        try:
            result = self.emotion_classifier(text)
            return result[0]
        except Exception as e:
            logger.error(f"Emotion analysis failed: {e}")
            return {"label": "neutral", "score": 1.0}
    
    def _generate_prompt_from_emotion(self, emotion, intensity, context="podcast"):
        """
        Generate an image prompt based on emotion and intensity.
        
        Args:
            emotion (str): Detected emotion.
            intensity (float): Emotion intensity.
            context (str): Context for the prompt.
            
        Returns:
            str: Image generation prompt.
        """
        # Base prompts for different emotions
        emotion_prompts = {
            "joy": "vibrant landscape with bright colors, sunshine, and blooming flowers",
            "happy": "sunny day in a colorful garden with butterflies and clear blue sky",
            "neutral": "balanced and calm landscape with soft clouds and gentle light",
            "sad": "rainy day with muted colors, foggy atmosphere, and still water",
            "anger": "stormy seascape with dark clouds, lightning, and crashing waves",
            "fear": "dark forest with twisted trees, deep shadows, and mist",
            "surprise": "magical scene with unexpected elements, light bursts, and wonder",
            "disgust": "decaying landscape with murky colors and unsettling atmosphere"
        }
        
        # Get the base prompt for the emotion or default to neutral
        base_prompt = emotion_prompts.get(emotion.lower(), emotion_prompts["neutral"])
        
        # Add intensity modifiers
        if intensity > 0.8:
            intensity_modifier = "extremely powerful and overwhelming"
        elif intensity > 0.6:
            intensity_modifier = "strong and distinctive"
        else:
            intensity_modifier = "subtle and gentle"
        
        # Combine into final prompt
        prompt = f"A {intensity_modifier} artistic illustration of {base_prompt}, representing {emotion} mood in a {context} diary, digital art"
        return prompt
    
    def _generate_image(self, prompt):
        """
        Generate an image using Stable Diffusion.
        
        Args:
            prompt (str): Image generation prompt.
            
        Returns:
            PIL.Image: Generated image.
        """
        logger.info(f"Generating image with prompt: {prompt}")
        
        try:
            with torch.autocast("cuda" if self.device == "cuda" else "cpu"):
                image = self.image_generator(
                    prompt, 
                    height=self.config['image_size'],
                    width=self.config['image_size'],
                    num_inference_steps=50
                ).images[0]
            
            logger.info("Image generation successful")
            return image
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            # Return a blank image as fallback
            return Image.new('RGB', (self.config['image_size'], self.config['image_size']), (255, 255, 255))
    
    def _create_text_summary(self, segment_data, transcription):
        """
        Create a text summary of the podcast segment.
        
        Args:
            segment_data (dict): Segment features and emotion data.
            transcription (str): Transcribed text.
            
        Returns:
            str: Generated summary.
        """
        emotion = segment_data['emotion']['label']
        emotion_score = segment_data['emotion']['score']
        
        # Create a summary based on the features and transcription
        summary = (f"Segment from {segment_data['start_time']:.1f}s to {segment_data['end_time']:.1f}s.\n"
                  f"Primary emotion: {emotion.capitalize()} (confidence: {emotion_score:.2f})\n\n"
                  f"Voice characteristics:\n"
                  f"- Pitch: {'High' if segment_data['pitch'] > 0.5 else 'Low'}\n"
                  f"- Intensity: {'Strong' if segment_data['intensity'] > 0.5 else 'Soft'}\n"
                  f"- Speech rate: {'Fast' if segment_data['speech_rate'] > 2.5 else 'Moderate' if segment_data['speech_rate'] > 1.5 else 'Slow'}\n\n"
                  f"Excerpt: \"{transcription[:100]}{'...' if len(transcription) > 100 else ''}\"")
        
        return summary
    
    def _generate_audio_summary(self, summary_text, output_path):
        """
        Generate an audio summary using TTS.
        
        Args:
            summary_text (str): Text to convert to speech.
            output_path (str): Path to save the audio file.
            
        Returns:
            str: Path to the generated audio file.
        """
        try:
            self.tts.tts_to_file(summary_text, file_path=output_path)
            logger.info(f"Generated audio summary: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Failed to generate audio summary: {e}")
            return None
    
    def process_podcast(self, audio_path):
        """
        Process a podcast audio file and generate a visual diary.
        
        Args:
            audio_path (str): Path to the podcast audio file.
            
        Returns:
            dict: Information about the generated visual diary.
        """
        logger.info(f"Processing podcast: {audio_path}")
        
        # Create session folder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = os.path.join(self.config['output_dir'], f"session_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)
        
        # Extract audio features
        segments = self._extract_audio_features(audio_path, self.config['segment_length'])
        
        # Transcribe the audio
        transcription = self._transcribe_audio(audio_path)
        
        # Process each segment
        diary_entries = []
        
        for i, segment in enumerate(tqdm(segments, desc="Processing segments")):
            segment_id = f"segment_{i:03d}"
            
            # Find corresponding transcription segment
            segment_text = ""
            for ts_segment in transcription['segments']:
                if (ts_segment['start'] >= segment['start_time'] and 
                    ts_segment['start'] < segment['end_time']):
                    segment_text += ts_segment['text'] + " "
            
            # Analyze emotions
            emotion_result = self._analyze_emotions(segment_text)
            segment['emotion'] = emotion_result
            
            # Generate image prompt and create image
            prompt = self._generate_prompt_from_emotion(
                emotion_result['label'], 
                emotion_result['score']
            )
            
            image = self._generate_image(prompt)
            image_path = os.path.join(session_dir, f"{segment_id}.{self.config['image_format']}")
            image.save(image_path)
            
            # Create text summary
            summary = self._create_text_summary(segment, segment_text)
            summary_path = os.path.join(session_dir, f"{segment_id}_summary.txt")
            with open(summary_path, 'w') as f:
                f.write(summary)
            
            # Generate audio summary
            audio_summary_path = os.path.join(session_dir, f"{segment_id}_summary.wav")
            self._generate_audio_summary(summary, audio_summary_path)
            
            # Add to diary entries
            diary_entries.append({
                'segment_id': segment_id,
                'start_time': segment['start_time'],
                'end_time': segment['end_time'],
                'emotion': emotion_result['label'],
                'emotion_score': emotion_result['score'],
                'image_path': image_path,
                'summary_path': summary_path,
                'audio_summary_path': audio_summary_path,
                'prompt': prompt
            })
        
        # Create diary index
        index_path = os.path.join(session_dir, "diary_index.json")
        with open(index_path, 'w') as f:
            json.dump({
                'podcast': os.path.basename(audio_path),
                'processed_date': timestamp,
                'segments': len(diary_entries),
                'entries': diary_entries
            }, f, indent=2)
        
        # Create a visual summary of the entire podcast
        self._create_visual_summary(diary_entries, session_dir)
        
        logger.info(f"Podcast processing complete. Output directory: {session_dir}")
        return {
            'session_dir': session_dir,
            'entries': len(diary_entries),
            'index_path': index_path
        }
    
    def _create_visual_summary(self, diary_entries, session_dir):
        """
        Create a visual summary of the podcast emotions.
        
        Args:
            diary_entries (list): List of diary entries.
            session_dir (str): Session directory.
        """
        try:
            # Extract emotion data
            emotions = [entry['emotion'] for entry in diary_entries]
            timestamps = [entry['start_time'] / 60 for entry in diary_entries]  # Convert to minutes
            
            # Count emotions
            emotion_counts = {}
            for emotion in emotions:
                if emotion in emotion_counts:
                    emotion_counts[emotion] += 1
                else:
                    emotion_counts[emotion] = 1
            
            # Create plots
            plt.figure(figsize=(15, 10))
            
            # Emotion timeline
            plt.subplot(2, 1, 1)
            emotion_map = {
                'joy': 0, 'happy': 0,
                'neutral': 1,
                'sad': 2, 'fear': 3,
                'anger': 4, 'disgust': 5,
                'surprise': 6
            }
            
            # Map emotions to numerical values for plotting
            emotion_values = [emotion_map.get(e, 1) for e in emotions]
            
            # Create scatter plot
            plt.scatter(timestamps, emotion_values, c=emotion_values, cmap='viridis', s=100, alpha=0.7)
            plt.yticks(list(set(emotion_values)), [k for k, v in sorted(emotion_map.items(), key=lambda x: x[1])])
            plt.xlabel('Time (minutes)')
            plt.ylabel('Emotion')
            plt.title('Emotional Journey Through the Podcast')
            plt.grid(True, alpha=0.3)
            
            # Emotion distribution pie chart
            plt.subplot(2, 1, 2)
            labels = list(emotion_counts.keys())
            sizes = list(emotion_counts.values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=plt.cm.tab10.colors)
            plt.axis('equal')
            plt.title('Emotion Distribution')
            
            # Save the summary plot
            summary_path = os.path.join(session_dir, "emotional_summary.png")
            plt.tight_layout()
            plt.savefig(summary_path)
            plt.close()
            
            logger.info(f"Created visual summary: {summary_path}")
        except Exception as e:
            logger.error(f"Failed to create visual summary: {e}")
    
    def generate_html_report(self, session_dir):
        """
        Generate an HTML report of the visual diary.
        
        Args:
            session_dir (str): Session directory.
            
        Returns:
            str: Path to the HTML report.
        """
        try:
            # Load diary index
            index_path = os.path.join(session_dir, "diary_index.json")
            with open(index_path, 'r') as f:
                diary_data = json.load(f)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Podcast Visual Diary - {diary_data['podcast']}</title>
                <style>
                    body {{
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                        background-color: #f5f5f5;
                    }}
                    h1, h2, h3 {{
                        color: #333;
                    }}
                    .summary-container {{
                        display: flex;
                        justify-content: center;
                        margin: 20px 0;
                    }}
                    .summary-image {{
                        max-width: 100%;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                    }}
                    .entries-grid {{
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 20px;
                        margin-top: 30px;
                    }}
                    .entry-card {{
                        background-color: white;
                        border-radius: 8px;
                        overflow: hidden;
                        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
                        transition: transform 0.3s ease;
                    }}
                    .entry-card:hover {{
                        transform: translateY(-5px);
                    }}
                    .entry-image {{
                        width: 100%;
                        height: 200px;
                        object-fit: cover;
                    }}
                    .entry-details {{
                        padding: 15px;
                    }}
                    .emotion-tag {{
                        display: inline-block;
                        padding: 5px 10px;
                        border-radius: 15px;
                        font-size: 14px;
                        color: white;
                        margin-bottom: 10px;
                    }}
                    .time-stamp {{
                        color: #666;
                        font-size: 14px;
                    }}
                    .audio-control {{
                        width: 100%;
                        margin-top: 10px;
                    }}
                    footer {{
                        margin-top: 50px;
                        text-align: center;
                        color: #666;
                    }}
                </style>
            </head>
            <body>
                <h1>Podcast Visual Diary</h1>
                <h2>{diary_data['podcast']}</h2>
                <p>Processed on {diary_data['processed_date']}</p>
                
                <div class="summary-container">
                    <img src="emotional_summary.png" alt="Emotional Summary" class="summary-image">
                </div>
                
                <h2>Diary Entries</h2>
                <div class="entries-grid">
            """
            
            # Emotion colors for tags
            emotion_colors = {
                'joy': '#FFD700',
                'happy': '#32CD32',
                'neutral': '#87CEEB',
                'sad': '#6495ED',
                'fear': '#9370DB',
                'anger': '#FF6347',
                'disgust': '#8B4513',
                'surprise': '#FF69B4'
            }
            
            # Add entries
            for entry in diary_data['entries']:
                # Get relative paths
                image_path = os.path.basename(entry['image_path'])
                audio_path = os.path.basename(entry['audio_summary_path'])
                
                # Get summary text
                with open(entry['summary_path'], 'r') as f:
                    summary = f.read().replace('\n', '<br>')
                
                # Get emotion color
                emotion_color = emotion_colors.get(entry['emotion'], '#888888')
                
                html_content += f"""
                <div class="entry-card">
                    <img src="{image_path}" alt="Visual for {entry['segment_id']}" class="entry-image">
                    <div class="entry-details">
                        <span class="emotion-tag" style="background-color: {emotion_color};">
                            {entry['emotion'].capitalize()} ({entry['emotion_score']:.2f})
                        </span>
                        <p class="time-stamp">
                            {entry['start_time']/60:.1f} min - {entry['end_time']/60:.1f} min
                        </p>
                        <p>{summary}</p>
                        <audio controls class="audio-control">
                            <source src="{audio_path}" type="audio/wav">
                            Your browser does not support the audio element.
                        </audio>
                    </div>
                </div>
                """
            
            html_content += """
                </div>
                
                <footer>
                    <p>Generated with Podcast Audio to Visual Diary Generator</p>
                </footer>
            </body>
            </html>
            """
            
            # Save HTML report
            report_path = os.path.join(session_dir, "visual_diary_report.html")
            with open(report_path, 'w') as f:
                f.write(html_content)
            
            logger.info(f"Generated HTML report: {report_path}")
            return report_path
        except Exception as e:
            logger.error(f"Failed to generate HTML report: {e}")
            return None

def main():
    """Main function to run the podcast visual diary generator."""
    parser = argparse.ArgumentParser(description='Generate a visual diary from podcast audio.')
    parser.add_argument('--audio', required=True, help='Path to the podcast audio file')
    parser.add_argument('--output', default='visual_diary_output', help='Output directory')
    parser.add_argument('--segment-length', type=int, default=30, help='Segment length in seconds')
    parser.add_argument('--image-size', type=int, default=512, help='Size of generated images')
    parser.add_argument('--no-gpu', action='store_true', help='Disable GPU usage')
    
    args = parser.parse_args()
    
    # Configure the generator
    config = {
        'segment_length': args.segment_length,
        'output_dir': args.output,
        'use_gpu': not args.no_gpu,
        'image_size': args.image_size
    }
    
    # Create generator instance
    generator = PodcastVisualDiaryGenerator(config)
    
    # Process the podcast
    result = generator.process_podcast(args.audio)
    
    # Generate HTML report
    html_report = generator.generate_html_report(result['session_dir'])
    
    print(f"\nProcessing complete!")
    print(f"Processed {result['entries']} segments")
    print(f"Output directory: {result['session_dir']}")
    
    if html_report:
        print(f"HTML report: {html_report}")
        print(f"Open the HTML report in your browser to view the visual diary.")

if __name__ == "__main__":
    main()
