### Setup Instructions

1. Install Python 3.8 or higher
2. Install required packages:
   pip install -r requirements.txt

3. Install FFmpeg:
   - Windows: Download from https://ffmpeg.org/download.html
   - macOS: brew install ffmpeg
   - Linux: sudo apt-get install ffmpeg

4. For advanced features, get API keys:
   - Musixmatch API: https://developer.musixmatch.com/
   - Genius API: https://genius.com/api-clients
   - Google Cloud Speech-to-Text: For better transcription accuracy

5. Run the application:
   python smart_karaoke_app.py

Features:
- Audio and Video mode processing
- 150+ language support with focus on Indian languages
- YouTube download integration
- Voice pitch calibration and adjustment
- Automatic lyrics extraction and timing
- Real-time audio visualization
- Advanced audio processing (vocal isolation, reverb, noise reduction)
- Video generation with synchronized lyrics
- Export functionality for multiple formats
- User-friendly GUI with progress tracking

API Integration:
- YouTube-DL for video/audio download
- Google Translate for multilingual support
- Speech Recognition for voice analysis
- Lyrics APIs (Musixmatch, Genius, Lyrics.ovh)
- Google Cloud Speech-to-Text (optional for better accuracy)

Advanced Features:
- Harmonic-percussive separation for vocal isolation
- Phase vocoder for high-quality pitch shifting
- Spectral subtraction for noise reduction
- Animated lyrics with visual effects
- Multi-format export support
- Real-time audio analysis and visualization
