import pandas as pd
import numpy as np
from typing import Dict, Any, List
import re
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

class DataQualityAssessment:
    def __init__(self):
        """
        Comprehensive data quality assessment module
        """
        self.quality_metrics = {}
    
    def assess_requirements_quality(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Perform comprehensive quality assessment on requirements
        """
        # Convert to DataFrame for easier analysis
        df = pd.DataFrame(requirements)
        
        # Quality metrics dictionary
        quality_metrics = {
            'total_requirements': len(df),
            'completeness': {},
            'consistency': {},
            'uniqueness': {},
            'complexity': {}
        }
        
        # Completeness assessment
        quality_metrics['completeness'] = self._assess_completeness(df)
        
        # Consistency check
        quality_metrics['consistency'] = self._check_consistency(df)
        
        # Uniqueness assessment
        quality_metrics['uniqueness'] = self._assess_uniqueness(df)
        
        # Complexity analysis
        quality_metrics['complexity'] = self._analyze_complexity(df)
        
        # Linguistic complexity
        quality_metrics['linguistic_complexity'] = self._linguistic_complexity_analysis(df)
        
        return quality_metrics
    
    def _assess_completeness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess requirements completeness
        """
        completeness = {}
        
        # Check for missing or empty fields
        for column in df.columns:
            total_count = len(df)
            non_empty_count = df[column].notna().sum()
            completeness[column] = non_empty_count / total_count * 100
        
        return completeness
    
    def _check_consistency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Check requirements consistency
        """
        consistency = {}
        
        # Check for consistent length
        consistency['length_variation'] = {
            'mean_length': df['text'].str.len().mean(),
            'std_length': df['text'].str.len().std()
        }
        
        # Keyword consistency
        key_phrases = ['shall', 'must', 'should', 'will']
        consistency['keyword_usage'] = {
            phrase: df['text'].str.contains(phrase, case=False).mean() * 100 
            for phrase in key_phrases
        }
        
        return consistency
    
    def _assess_uniqueness(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Assess requirement uniqueness
        """
        # Use cosine similarity to detect similar requirements
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['text'])
        similarity_matrix = cosine_similarity(tfidf_matrix)
        
        # Count highly similar requirements
        threshold = 0.8
        similar_count = np.sum(similarity_matrix > threshold) - len(df)
        
        return {
            'total_unique_requirements': len(df),
            'similar_requirements_percentage': similar_count / (len(df)**2) * 100
        }
    
    def _analyze_complexity(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze requirement complexity
        """
        complexity = {}
        
        # Word count complexity
        df['word_count'] = df['text'].str.split().str.len()
        complexity['word_count'] = {
            'mean': df['word_count'].mean(),
            'median': df['word_count'].median(),
            'max': df['word_count'].max()
        }
        
        # Sentence complexity
        df['sentence_count'] = df['text'].str.count(r'[.!?]')
        complexity['sentence_complexity'] = {
            'mean_sentences': df['sentence_count'].mean(),
            'requirements_with_multiple_sentences_ratio': 
                (df['sentence_count'] > 1).mean() * 100
        }
        
        return complexity
    
    def _linguistic_complexity_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform linguistic complexity analysis
        """
        import textstat
        
        linguistic_complexity = {}
        
        # Readability scores
        df['flesch_reading_ease'] = df['text'].apply(textstat.flesch_reading_ease)
        df['flesch_kincaid_grade'] = df['text'].apply(textstat.flesch_kincaid_grade)
        
        linguistic_complexity['readability'] = {
            'mean_flesch_reading_ease': df['flesch_reading_ease'].mean(),
            'mean_flesch_kincaid_grade': df['flesch_kincaid_grade'].mean(),
            'readability_difficulty_distribution': {
                'very_easy': (df['flesch_reading_ease'] > 80).mean() * 100,
                'easy': ((df['flesch_reading_ease'] > 50) & (df['flesch_reading_ease'] <= 80)).mean() * 100,
                'medium': ((df['flesch_reading_ease'] > 30) & (df['flesch_reading_ease'] <= 50)).mean() * 100,
                'difficult': ((df['flesch_reading_ease'] > 0) & (df['flesch_reading_ease'] <= 30)).mean() * 100,
                'very_difficult': (df['flesch_reading_ease'] <= 0).mean() * 100
            }
        }
        
        return linguistic_complexity
    
    def generate_data_lineage(self, requirements: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate data lineage tracking
        """
        import hashlib
        import datetime
        
        lineage = {
            'metadata': {
                'generation_timestamp': datetime.datetime.now().isoformat(),
                'total_requirements': len(requirements)
            },
            'requirements': []
        }
        
        for req in requirements:
            # Generate unique hash for traceability
            req_hash = hashlib.md5(
                json.dumps(req, sort_keys=True).encode()
            ).hexdigest()
            
            lineage['requirements'].append({
                'id': req_hash,
                'source': req.get('source', 'Unknown'),
                'timestamp': datetime.datetime.now().isoformat()
            })
        
        return lineage

# Example usage
def main():
    # Sample requirements data
    requirements = [
        {'text': 'The system shall provide user authentication', 'category': 'functional'},
        {'text': 'Performance must be optimized for high load', 'category': 'non-functional'}
    ]
    
    quality_assessor = DataQualityAssessment()
    
    # Assess quality
    quality_metrics = quality_assessor.assess_requirements_quality(requirements)
    print("Quality Metrics:", json.dumps(quality_metrics, indent=2))
    
    # Generate data lineage
    lineage = quality_assessor.generate_data_lineage(requirements)
    print("Data Lineage:", json.dumps(lineage, indent=2))

if __name__ == "__main__":
    main()
