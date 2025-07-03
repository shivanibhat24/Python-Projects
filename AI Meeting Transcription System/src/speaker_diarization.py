"""
Speaker diarization module for meeting transcription system.
Identifies and segments speakers in audio recordings.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from pyannote.audio import Pipeline
from pyannote.core import Annotation, Segment
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import speechbrain as sb
from speechbrain.pretrained import SpeakerRecognition

logger = logging.getLogger(__name__)


class SpeakerDiarizer:
    """Handles speaker diarization using pyannote.audio."""
    
    def __init__(self, model_name: str = "pyannote/speaker-diarization-3.1",
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.model_name = model_name
        self.device = device
        self.pipeline = None
        self._load_model()
    
    def _load_model(self):
        """Load the speaker diarization model."""
        try:
            self.pipeline = Pipeline.from_pretrained(self.model_name)
            if torch.cuda.is_available():
                self.pipeline.to(torch.device(self.device))
            logger.info(f"Loaded diarization model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading diarization model: {e}")
            raise
    
    def diarize(self, audio_file: str, min_speakers: int = 2, 
                max_speakers: int = 10) -> Annotation:
        """Perform speaker diarization on audio file."""
        try:
            # Run diarization
            diarization = self.pipeline(audio_file, 
                                      min_speakers=min_speakers,
                                      max_speakers=max_speakers)
            
            logger.info(f"Diarization completed. Found {len(diarization.labels())} speakers")
            return diarization
            
        except Exception as e:
            logger.error(f"Error in diarization: {e}")
            raise
    
    def get_speaker_segments(self, diarization: Annotation) -> Dict[str, List[Tuple[float, float]]]:
        """Extract speaker segments from diarization result."""
        speaker_segments = {}
        
        for segment, _, speaker in diarization.itertracks(yield_label=True):
            if speaker not in speaker_segments:
                speaker_segments[speaker] = []
            
            speaker_segments[speaker].append((segment.start, segment.end))
        
        return speaker_segments
    
    def merge_consecutive_segments(self, segments: List[Tuple[float, float]], 
                                 gap_threshold: float = 0.5) -> List[Tuple[float, float]]:
        """Merge consecutive segments with small gaps."""
        if not segments:
            return segments
        
        # Sort segments by start time
        segments = sorted(segments, key=lambda x: x[0])
        merged = [segments[0]]
        
        for current_start, current_end in segments[1:]:
            last_start, last_end = merged[-1]
            
            # If gap is small, merge segments
            if current_start - last_end <= gap_threshold:
                merged[-1] = (last_start, max(last_end, current_end))
            else:
                merged.append((current_start, current_end))
        
        return merged


class SpeakerEmbedder:
    """Extract speaker embeddings for identification."""
    
    def __init__(self, model_name: str = "speechbrain/spkrec-ecapa-voxceleb"):
        self.model_name = model_name
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load speaker embedding model."""
        try:
            self.model = SpeakerRecognition.from_hparams(
                source=self.model_name,
                savedir=f"models/{self.model_name.replace('/', '_')}"
            )
            logger.info(f"Loaded speaker embedding model: {self.model_name}")
        except Exception as e:
            logger.error(f"Error loading speaker embedding model: {e}")
            raise
    
    def extract_embedding(self, audio_file: str) -> np.ndarray:
        """Extract speaker embedding from audio file."""
        try:
            embedding = self.model.encode_batch(audio_file)
            return embedding.squeeze().cpu().numpy()
        except Exception as e:
            logger.error(f"Error extracting embedding: {e}")
            raise
    
    def extract_embeddings_from_segments(self, audio_file: str, 
                                       segments: List[Tuple[float, float]]) -> List[np.ndarray]:
        """Extract embeddings from audio segments."""
        embeddings = []
        
        for start_time, end_time in segments:
            try:
                # Extract segment embedding
                embedding = self.model.encode_batch(audio_file, 
                                                  start_time=start_time,
                                                  end_time=end_time)
                embeddings.append(embedding.squeeze().cpu().numpy())
            except Exception as e:
                logger.warning(f"Error extracting embedding for segment {start_time}-{end_time}: {e}")
                continue
        
        return embeddings
    
    def compute_similarity(self, embedding1: np.ndarray, 
                          embedding2: np.ndarray) -> float:
        """Compute cosine similarity between two embeddings."""
        return cosine_similarity([embedding1], [embedding2])[0][0]


class SpeakerIdentifier:
    """Identify speakers using embeddings and clustering."""
    
    def __init__(self, similarity_threshold: float = 0.7):
        self.similarity_threshold = similarity_threshold
        self.known_speakers = {}  # speaker_id -> embedding
        self.speaker_counter = 0
    
    def add_known_speaker(self, speaker_id: str, embedding: np.ndarray):
        """Add a known speaker with their embedding."""
        self.known_speakers[speaker_id] = embedding
        logger.info(f"Added known speaker: {speaker_id}")
    
    def identify_speaker(self, embedding: np.ndarray) -> Optional[str]:
        """Identify speaker from embedding."""
        if not self.known_speakers:
            return None
        
        max_similarity = -1
        best_match = None
        
        for speaker_id, known_embedding in self.known_speakers.items():
            similarity = cosine_similarity([embedding], [known_embedding])[0][0]
            
            if similarity > max_similarity and similarity > self.similarity_threshold:
                max_similarity = similarity
                best_match = speaker_id
        
        return best_match
    
    def cluster_speakers(self, embeddings: List[np.ndarray], 
                        n_clusters: Optional[int] = None) -> List[int]:
        """Cluster speaker embeddings."""
        if len(embeddings) < 2:
            return [0] * len(embeddings)
        
        # Stack embeddings
        X = np.stack(embeddings)
        
        # Determine number of clusters
        if n_clusters is None:
            # Use distance threshold for clustering
            clustering = AgglomerativeClustering(
                distance_threshold=1 - self.similarity_threshold,
                n_clusters=None,
                linkage='average',
                metric='cosine'
            )
        else:
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                linkage='average',
                metric='cosine'
            )
        
        labels = clustering.fit_predict(X)
        return labels.tolist()
    
    def assign_speaker_labels(self, embeddings: List[np.ndarray], 
                            segments: List[Tuple[float, float]]) -> Dict[Tuple[float, float], str]:
        """Assign speaker labels to segments."""
        if len(embeddings) != len(segments):
            raise ValueError("Number of embeddings must match number of segments")
        
        # First try to identify known speakers
        segment_labels = {}
        unknown_embeddings = []
        unknown_segments = []
        
        for embedding, segment in zip(embeddings, segments):
            speaker_id = self.identify_speaker(embedding)
            if speaker_id:
                segment_labels[segment] = speaker_id
            else:
                unknown_embeddings.append(embedding)
                unknown_segments.append(segment)
        
        # Cluster unknown speakers
        if unknown_embeddings:
            cluster_labels = self.cluster_speakers(unknown_embeddings)
            
            # Assign new speaker IDs
            cluster_to_speaker = {}
            for cluster_id in set(cluster_labels):
                if cluster_id not in cluster_to_speaker:
                    speaker_id = f"Speaker_{self.speaker_counter}"
                    cluster_to_speaker[cluster_id] = speaker_id
                    self.speaker_counter += 1
            
            # Map clusters to segments
            for embedding, segment, cluster_id in zip(unknown_embeddings, unknown_segments, cluster_labels):
                speaker_id = cluster_to_speaker[cluster_id]
                segment_labels[segment] = speaker_id
                
                # Add to known speakers
                self.known_speakers[speaker_id] = embedding
        
        return segment_labels


class SpeakerDiarizationPipeline:
    """Complete speaker diarization pipeline."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.diarizer = SpeakerDiarizer(
            model_name=config.get('model_name', 'pyannote/speaker-diarization-3.1'),
            device=config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        )
        self.embedder = SpeakerEmbedder()
        self.identifier = SpeakerIdentifier(
            similarity_threshold=config.get('threshold', 0.7)
        )
    
    def process_audio(self, audio_file: str) -> Dict:
        """Process audio file for speaker diarization."""
        logger.info(f"Processing audio file: {audio_file}")
        
        # Step 1: Perform diarization
        diarization = self.diarizer.diarize(
            audio_file,
            min_speakers=self.config.get('min_speakers', 2),
            max_speakers=self.config.get('max_speakers', 10)
        )
        
        # Step 2: Extract speaker segments
        speaker_segments = self.diarizer.get_speaker_segments(diarization)
        
        # Step 3: Merge consecutive segments
        for speaker in speaker_segments:
            speaker_segments[speaker] = self.diarizer.merge_consecutive_segments(
                speaker_segments[speaker]
            )
        
        # Step 4: Extract embeddings and identify speakers
        all_segments = []
        all_embeddings = []
        
        for speaker, segments in speaker_segments.items():
            for segment in segments:
                all_segments.append(segment)
                # Extract embedding for this segment
                embedding = self.embedder.extract_embedding(audio_file)
                all_embeddings.append(embedding)
        
        # Step 5: Assign final speaker labels
        segment_labels = self.identifier.assign_speaker_labels(
            all_embeddings, all_segments
        )
        
        # Step 6: Organize results
        results = {
            'speakers': list(set(segment_labels.values())),
            'segments': [],
            'speaker_times': {}
        }
        
        # Calculate speaker speaking times
        for speaker in results['speakers']:
            results['speaker_times'][speaker] = 0
        
        for segment, speaker in segment_labels.items():
            start_time, end_time = segment
            duration = end_time - start_time
            
            results['segments'].append({
                'start': start_time,
                'end': end_time,
                'speaker': speaker,
                'duration': duration
            })
            
            results['speaker_times'][speaker] += duration
        
        # Sort segments by start time
        results['segments'].sort(key=lambda x: x['start'])
        
        logger.info(f"Diarization completed. Found {len(results['speakers'])} speakers")
        return results
