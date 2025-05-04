# Pitch Deck Reconstructor - Usage Guide

This tool automatically converts meeting transcripts or audio recordings of pitch presentations into polished PowerPoint pitch decks using AI.

## Features

- **Audio to Text Conversion**: Processes audio files using Whisper to create accurate transcripts
- **Intelligent Content Analysis**: Uses LLaMA 3 to extract key pitch deck components from transcripts
- **Automatic Slide Generation**: Creates complete pitch decks with proper structure and formatting
- **Multiple Format Support**: Works with text files (.txt, .md) or audio files (.mp3, .wav, .m4a, .ogg)
- **Template Support**: Can use custom PowerPoint templates for brand consistency

## Installation

1. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. Install the required dependencies:
   ```bash
   pip install langchain python-pptx transformers torch huggingface_hub librosa
   ```

3. Download the necessary models (first run will download them automatically):
   - LLaMA 3 (8B parameter version)
   - Whisper (for audio transcription)

## Usage

### Basic Usage

```bash
python pitch_deck_reconstructor.py --input transcript.txt --output my_pitch_deck.pptx
```

### Processing Audio Files

```bash
python pitch_deck_reconstructor.py --input pitch_recording.mp3
```

### Using a Company Name

```bash
python pitch_deck_reconstructor.py --input transcript.txt --company "TechStartup Inc."
```

### Using a Custom Template

```bash
python pitch_deck_reconstructor.py --input transcript.txt --template template.pptx
```

## Generated Slide Structure

The tool creates a comprehensive pitch deck with these slides:

1. **Title Slide**: Company name, tagline
2. **Problem Statement**: What problem is being solved
3. **Solution**: How the company solves this problem
4. **Market Opportunity**: Market size, trends, growth potential
5. **Product/Service**: Details about what they offer
6. **Business Model**: How they make money
7. **Competitive Landscape**: Competitors and advantages
8. **Traction & Metrics**: Current progress, customers, revenue
9. **Team**: Key team members and background
10. **Financial Projections**: Revenue forecasts, funding needs
11. **Go-to-Market Strategy**: How they'll acquire customers
12. **Call to Action**: What they're asking for

## Example

```bash
python pitch_deck_reconstructor.py --input sample_pitch.txt --company "GreenTech Solutions" --template corporate_template.pptx
```

## Tips for Best Results

- Ensure audio recordings are clear with minimal background noise
- For better accuracy, provide the company name via the `--company` parameter
- To improve formatting, use a corporate template that matches the company's branding
- For complete presentations, ensure the transcript covers all key aspects of a pitch (problem, solution, market, etc.)

## Troubleshooting

- **Memory Issues**: If you encounter memory errors with LLaMA 3, use a smaller model or ensure your system has adequate RAM
- **Audio Processing Failures**: Ensure audio files are in a supported format and properly encoded
- **Missing Content**: If slides are missing content, check that your transcript adequately covers those topics
