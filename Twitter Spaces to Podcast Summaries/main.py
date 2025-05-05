# Twitter Spaces to Podcast Summaries

This tool converts Twitter Spaces recordings into concise podcast summaries with highlight reels. It uses a combination of technologies to create high-quality audio summaries of Twitter Spaces conversations.

## Features

- **Automatic Audio Transcription**: Uses OpenAI's Whisper to convert spoken conversations into text
- **Speaker Identification**: Attempts to identify different speakers in the conversation
- **Key Points Extraction**: Analyzes the transcript using spaCy to find the most important parts
- **Summary Generation**: Creates concise text summaries of the Space
- **Highlight Reel Creation**: Extracts the most interesting segments into a highlight audio file
- **Text-to-Speech Summary**: Converts the text summary into a spoken overview

## Prerequisites

Before using this tool, you'll need:

1. **Python 3.8+**
2. **Twitter API Access**: You'll need to create a Twitter Developer account and obtain API keys
3. **Required Libraries**: Install all dependencies using pip

## Installation

1. Clone or download the code to your local machine
2. Install the required dependencies:

```bash
pip install tweepy openai-whisper spacy pydub torch TTS numpy
python -m spacy download en_core_web_lg
```

3. Create a configuration file (see example `config.json`)
4. Make sure you have enough disk space for audio processing

## Configuration

Create a `config.json` file with the following structure:

```json
{
  "twitter_api_key": "YOUR_TWITTER_API_KEY",
  "twitter_api_secret": "YOUR_TWITTER_API_SECRET",
  "twitter_access_token": "YOUR_TWITTER_ACCESS_TOKEN",
  "twitter_access_token_secret": "YOUR_TWITTER_ACCESS_TOKEN_SECRET",
  "output_dir": "./output",
  "min_highlight_duration": 15,
  "max_highlight_duration": 60,
  "summary_length": 10,
  "num_highlights": 5
}
```

- `min_highlight_duration` and `max_highlight_duration`: Control the length of highlight clips (in seconds)
- `summary_length`: Number of key points to include in the summary
- `num_highlights`: Number of audio highlights to extract

## Usage

You can use the tool in three different ways:

### 1. Process a Specific Twitter Space

```bash
python twitter_spaces_to_podcast.py --config config.json --space-id 1AvGPyzneWlwX
```

### 2. Process All Spaces from a Specific User

```bash
python twitter_spaces_to_podcast.py --config config.json --user-id 12345678
```

### 3. Process Currently Active Spaces

```bash
python twitter_spaces_to_podcast.py --config config.json
```

## Output Files

The tool generates several output files in the specified output directory:

- `audio/`: Contains the original downloaded audio
- `transcripts/`: Contains JSON files with the full transcription
- `summaries/`: Contains text and audio summaries
- `highlights/`: Contains the highlight audio reels

## Technical Notes

- **Twitter API**: This tool uses Tweepy to interact with the Twitter API. Note that Twitter's API for accessing Spaces might change, so you may need to adjust the code accordingly.

- **Speech Recognition**: We use Whisper's "medium" model for transcription. You can switch to other models (tiny, base, small, large) depending on your accuracy needs and computational resources.

- **Speaker Diarization**: The current implementation uses a simplified approach for speaker identification. For production use, you might want to integrate a more sophisticated diarization model.

- **NLP Processing**: The tool uses spaCy to extract important content. You can adjust the scoring mechanism in the `extract_key_points` method to better suit your needs.

## Limitations

- The speaker identification is approximative and may not always correctly identify different speakers
- Twitter API access to Spaces content may be limited or change over time
- Processing long audio files requires significant computational resources
- The TTS voice is synthetic and may not sound natural for all content

## Future Improvements

- Add more advanced speaker diarization
- Implement topic modeling to better organize content
- Add support for multi-language Spaces
- Create a web interface for easier usage
- Add more customization options for the summary generation
