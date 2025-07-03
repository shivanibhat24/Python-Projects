"""
NLP processing module for meeting transcription system.
Handles text analysis, named entity recognition, and sentiment analysis.
"""

import spacy
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.chunk import ne_chunk
from nltk.tag import pos_tag
from textblob import TextBlob
import re
from typing import List, Dict, Tuple, Optional, Set
import logging
from collections import Counter, defaultdict
import numpy as np

logger = logging.getLogger(__name__)


class NLPProcessor:
    """Main NLP processing class."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.model_name = model_name
        self.nlp = None
        self.stop_words = set()
        self._load_models()
    
    def _load_models(self):
        """Load NLP models and resources."""
        try:
            # Load spaCy model
            self.nlp = spacy.load(self.model_name)
            logger.info(f"Loaded spaCy model: {self.model_name}")
            
            # Download NLTK data if not already present
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                nltk.download('punkt')
            
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                nltk.download('stopwords')
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                nltk.download('averaged_perceptron_tagger')
            
            try:
                nltk.data.find('chunkers/maxent_ne_chunker')
            except LookupError:
                nltk.download('maxent_ne_chunker')
            
            try:
                nltk.data.find('corpora/words')
            except LookupError:
                nltk.download('words')
            
            # Load stop words
            self.stop_words = set(stopwords.words('english'))
            
            logger.info("NLP models loaded successfully")
            
        except Exception as e:
            logger.error(f"Error loading NLP models: {e}")
            raise
    
    def process_text(self, text: str) -> Dict:
        """Process text through complete NLP pipeline."""
        results = {
            'original_text': text,
            'sentences': self.segment_sentences(text),
            'entities': self.extract_entities(text),
            'sentiment': self.analyze_sentiment(text),
            'keywords': self.extract_keywords(text),
            'topics': self.extract_topics(text),
            'pos_tags': self.get_pos_tags(text),
            'summary': self.generate_summary(text)
        }
        
        return results
    
    def segment_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        try:
            # Use spaCy for sentence segmentation
            doc = self.nlp(text)
            sentences = [sent.text.strip() for sent in doc.sents]
            return sentences
        except Exception as e:
            logger.error(f"Error in sentence segmentation: {e}")
            # Fallback to NLTK
            return sent_tokenize(text)
    
    def extract_entities(self, text: str) -> List[Dict]:
        """Extract named entities from text."""
        entities = []
        
        try:
            # Use spaCy for NER
            doc = self.nlp(text)
            
            for ent in doc.ents:
                entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start_char,
                    'end': ent.end_char,
                    'confidence': float(ent._.get('confidence', 0.8))
                })
        
        except Exception as e:
            logger.error(f"Error in entity extraction: {e}")
        
        return entities
    
    def analyze_sentiment(self, text: str) -> Dict:
        """Analyze sentiment of text."""
        try:
            # Use TextBlob for sentiment analysis
            blob = TextBlob(text)
            
            sentiment = {
                'polarity': blob.sentiment.polarity,  # -1 to 1
                'subjectivity': blob.sentiment.subjectivity,  # 0 to 1
                'label': self._get_sentiment_label(blob.sentiment.polarity)
            }
            
            return sentiment
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'polarity': 0.0, 'subjectivity': 0.0, 'label': 'neutral'}
    
    def _get_sentiment_label(self, polarity: float) -> str:
        """Convert polarity score to sentiment label."""
        if polarity > 0.1:
            return 'positive'
        elif polarity < -0.1:
            return 'negative'
        else:
            return 'neutral'
    
    def extract_keywords(self, text: str, top_k: int = 10) -> List[Dict]:
        """Extract keywords from text."""
        try:
            doc = self.nlp(text)
            
            # Extract tokens (excluding stop words and punctuation)
            tokens = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct 
                     and token.is_alpha and len(token.text) > 2]
            
            # Count frequency
            word_freq = Counter(tokens)
            
            # Get keywords with frequency
            keywords = [
                {'word': word, 'frequency': freq, 'score': freq / len(tokens)}
                for word, freq in word_freq.most_common(top_k)
            ]
            
            return keywords
            
        except Exception as e:
            logger.error(f"Error in keyword extraction: {e}")
            return []
    
    def extract_topics(self, text: str) -> List[str]:
        """Extract topics from text using simple heuristics."""
        try:
            doc = self.nlp(text)
            
            # Look for noun phrases
            noun_phrases = [chunk.text.lower() for chunk in doc.noun_chunks 
                          if len(chunk.text.split()) >= 2]
            
            # Count frequency
            phrase_freq = Counter(noun_phrases)
            
            # Get top topics
            topics = [phrase for phrase, freq in phrase_freq.most_common(5)]
            
            return topics
            
        except Exception as e:
            logger.error(f"Error in topic extraction: {e}")
            return []
    
    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """Get part-of-speech tags for text."""
        try:
            doc = self.nlp(text)
            pos_tags = [(token.text, token.pos_) for token in doc]
            return pos_tags
        except Exception as e:
            logger.error(f"Error in POS tagging: {e}")
            # Fallback to NLTK
            tokens = word_tokenize(text)
            return pos_tag(tokens)
    
    def generate_summary(self, text: str, max_sentences: int = 3) -> str:
        """Generate extractive summary of text."""
        try:
            sentences = self.segment_sentences(text)
            
            if len(sentences) <= max_sentences:
                return text
            
            # Simple extractive summarization
            # Score sentences based on keyword frequency
            keywords = self.extract_keywords(text, top_k=20)
            keyword_set = set(kw['word'] for kw in keywords)
            
            sentence_scores = []
            for sentence in sentences:
                doc = self.nlp(sentence.lower())
                words = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct]
                score = sum(1 for word in words if word in keyword_set)
                sentence_scores.append((sentence, score))
            
            # Get top sentences
            top_sentences = sorted(sentence_scores, key=lambda x: x[1], reverse=True)[:max_sentences]
            
            # Maintain original order
            summary_sentences = []
            for sentence in sentences:
                if any(sentence == s[0] for s in top_sentences):
                    summary_sentences.append(sentence)
            
            return ' '.join(summary_sentences)
            
        except Exception as e:
            logger.error(f"Error in summary generation: {e}")
            return text[:500] + "..." if len(text) > 500 else text


class MeetingAnalyzer:
    """Analyze meeting transcripts for insights."""
    
    def __init__(self, nlp_processor: NLPProcessor):
        self.nlp = nlp_processor
    
    def analyze_meeting(self, transcript_data: Dict) -> Dict:
        """Analyze complete meeting transcript."""
        results = {
            'meeting_summary': '',
            'key_topics': [],
            'action_items': [],
            'decisions': [],
            'speaker_analysis': {},
            'overall_sentiment': {},
            'meeting_stats': {}
        }
        
        # Combine all text
        all_text = []
        speaker_texts = defaultdict(list)
        
        for segment in transcript_data.get('timeline', []):
            text = segment.get('text', '')
            speaker = segment.get('speaker', 'Unknown')
            
            all_text.append(text)
            speaker_texts[speaker].append(text)
        
        combined_text = ' '.join(all_text)
        
        # Generate overall summary
        results['meeting_summary'] = self.nlp.generate_summary(combined_text)
        
        # Extract key topics
        results['key_topics'] = self.nlp.extract_topics(combined_text)
        
        # Extract action items and decisions
        results['action_items'] = self.extract_action_items(combined_text)
        results['decisions'] = self.extract_decisions(combined_text)
        
        # Analyze each speaker
        for speaker, texts in speaker_texts.items():
            speaker_text = ' '.join(texts)
            speaker_analysis = self.nlp.process_text(speaker_text)
            
            results['speaker_analysis'][speaker] = {
                'word_count': len(speaker_text.split()),
                'sentiment': speaker_analysis['sentiment'],
                'key_topics': speaker_analysis['topics'],
                'speaking_time_ratio': len(texts) / len(all_text)
            }
        
        # Overall sentiment
        results['overall_sentiment'] = self.nlp.analyze_sentiment(combined_text)
        
        # Meeting statistics
        results['meeting_stats'] = self.calculate_meeting_stats(transcript_data)
        
        return results
    
    def extract_action_items(self, text: str) -> List[str]:
        """Extract action items from meeting text."""
        action_items = []
        
        # Common action item patterns
        patterns = [
            r'(?:will|shall|should|need to|must|have to|going to)\s+([^.!?]*)',
            r'action item[:\s]*([^.!?]*)',
            r'(?:todo|to do|task)[:\s]*([^.!?]*)',
            r'(?:assign|assigned|responsibility)[:\s]*([^.!?]*)'
        ]
        
        sentences = self.nlp.segment_sentences(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check for action item indicators
            if any(indicator in sentence_lower for indicator in 
                   ['will', 'should', 'need to', 'action item', 'todo', 'assign']):
                
                # Extract using patterns
                for pattern in patterns:
                    matches = re.finditer(pattern, sentence_lower)
                    for match in matches:
                        action_text = match.group(1).strip()
                        if len(action_text) > 10:  # Filter out very short matches
                            action_items.append(sentence.strip())
                            break
        
        return list(set(action_items))  # Remove duplicates
    
    def extract_decisions(self, text: str) -> List[str]:
        """Extract decisions from meeting text."""
        decisions = []
        
        # Decision indicators
        decision_words = [
            'decided', 'decision', 'agree', 'agreed', 'conclude', 'concluded',
            'resolve', 'resolved', 'determine', 'determined', 'choose', 'chosen'
        ]
        
        sentences = self.nlp.segment_sentences(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            if any(word in sentence_lower for word in decision_words):
                decisions.append(sentence.strip())
        
        return decisions
    
    def calculate_meeting_stats(self, transcript_data: Dict) -> Dict:
        """Calculate meeting statistics."""
        timeline = transcript_data.get('timeline', [])
        
        if not timeline:
            return {}
        
        # Calculate duration
        start_time = min(segment['start'] for segment in timeline)
        end_time = max(segment['end'] for segment in timeline)
        duration = end_time - start_time
        
        # Speaker statistics
        speaker_stats = defaultdict(lambda: {'segments': 0, 'words': 0, 'time': 0})
        
        for segment in timeline:
            speaker = segment.get('speaker', 'Unknown')
            speaker_stats[speaker]['segments'] += 1
            speaker_stats[speaker]['words'] += len(segment.get('text', '').split())
            speaker_stats[speaker]['time'] += segment['end'] - segment['start']
        
        # Calculate talk time ratios
        total_talk_time = sum(stats['time'] for stats in speaker_stats.values())
        
        for speaker in speaker_stats:
            speaker_stats[speaker]['talk_ratio'] = (
                speaker_stats[speaker]['time'] / total_talk_time if total_talk_time > 0 else 0
            )
        
        return {
            'duration_minutes': duration / 60,
            'total_segments': len(timeline),
            'num_speakers': len(speaker_stats),
            'speaker_stats': dict(speaker_stats),
            'words_per_minute': sum(stats['words'] for stats in speaker_stats.values()) / (duration / 60) if duration > 0 else 0
        }


class TranscriptFormatter:
    """Format transcripts for different output types."""
    
    def __init__(self):
        pass
    
    def format_as_text(self, transcript_data: Dict, include_timestamps: bool = True) -> str:
        """Format transcript as plain text."""
        lines = []
        
        if include_timestamps:
            lines.append("MEETING TRANSCRIPT")
            lines.append("=" * 50)
            lines.append("")
        
        timeline = transcript_data.get('timeline', [])
        
        for segment in timeline:
            speaker = segment.get('speaker', 'Unknown')
            text = segment.get('text', '')
            start_time = segment.get('start', 0)
            
            if include_timestamps:
                timestamp = self._format_timestamp(start_time)
                lines.append(f"[{timestamp}] {speaker}: {text}")
            else:
                lines.append(f"{speaker}: {text}")
        
        return '\n'.join(lines)
    
    def format_as_srt(self, transcript_data: Dict) -> str:
        """Format transcript as SRT subtitle file."""
        lines = []
        
        timeline = transcript_data.get('timeline', [])
        
        for i, segment in enumerate(timeline, 1):
            start_time = segment.get('start', 0)
            end_time = segment.get('end', 0)
            text = segment.get('text', '')
            speaker = segment.get('speaker', 'Unknown')
            
            # SRT format
            lines.append(str(i))
            lines.append(f"{self._format_srt_time(start_time)} --> {self._format_srt_time(end_time)}")
            lines.append(f"{speaker}: {text}")
            lines.append("")
        
        return '\n'.join(lines)
    
    def format_as_json(self, transcript_data: Dict, analysis_data: Dict = None) -> Dict:
        """Format transcript as structured JSON."""
        result = {
            'transcript': transcript_data,
            'metadata': {
                'duration': transcript_data.get('duration', 0),
                'speakers': transcript_data.get('speakers', []),
                'total_segments': len(transcript_data.get('timeline', []))
            }
        }
        
        if analysis_data:
            result['analysis'] = analysis_data
        
        return result
    
    def _format_timestamp(self, seconds: float) -> str:
        """Format seconds as MM:SS timestamp."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format seconds as SRT time format (HH:MM:SS,mmm)."""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int((seconds % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
