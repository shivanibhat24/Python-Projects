import os
import json
import numpy as np
import pandas as pd
import spacy
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

class RequirementClassifier:
    def __init__(self, model_path='./models/requirement_classifier'):
        """
        Advanced requirement classification using machine learning
        """
        self.nlp = spacy.load('en_core_web_sm')
        self.vectorizer = TfidfVectorizer(
            stop_words='english', 
            max_features=5000,
            ngram_range=(1, 2)
        )
        self.classifier = LinearSVC(random_state=42)
        self.label_encoder = LabelEncoder()
        self.model_path = model_path
        
        # Ensure model directory exists
        os.makedirs(model_path, exist_ok=True)

    def preprocess_text(self, text):
        """
        Advanced text preprocessing
        """
        doc = self.nlp(text)
        
        # Remove stop words, lemmatize
        processed_tokens = [
            token.lemma_.lower() 
            for token in doc 
            if not token.is_stop and token.is_alpha
        ]
        
        return ' '.join(processed_tokens)

    def prepare_training_data(self, training_data_path):
        """
        Prepare training data from a JSON or CSV file
        Expected format: [{'text': 'requirement text', 'category': 'functional/non-functional'}]
        """
        if training_data_path.endswith('.json'):
            with open(training_data_path, 'r') as f:
                data = json.load(f)
        elif training_data_path.endswith('.csv'):
            data = pd.read_csv(training_data_path).to_dict('records')
        else:
            raise ValueError("Unsupported file format. Use JSON or CSV.")
        
        # Preprocess texts
        texts = [self.preprocess_text(item['text']) for item in data]
        labels = [item['category'] for item in data]
        
        return texts, labels

    def train_model(self, training_data_path):
        """
        Train the requirement classification model
        """
        # Prepare data
        X, y = self.prepare_training_data(training_data_path)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42
        )
        
        # Create pipeline
        self.pipeline = Pipeline([
            ('vectorizer', self.vectorizer),
            ('classifier', self.classifier)
        ])
        
        # Train model
        self.pipeline.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.pipeline.predict(X_test)
        print(classification_report(
            y_test, 
            y_pred, 
            target_names=self.label_encoder.classes_
        ))
        
        # Save model
        self._save_model()

    def _save_model(self):
        """
        Save trained model components
        """
        import joblib
        
        joblib.dump(self.vectorizer, 
                    os.path.join(self.model_path, 'vectorizer.joblib'))
        joblib.dump(self.pipeline, 
                    os.path.join(self.model_path, 'classifier.joblib'))
        joblib.dump(self.label_encoder, 
                    os.path.join(self.model_path, 'label_encoder.joblib'))

    def load_model(self):
        """
        Load pre-trained model components
        """
        import joblib
        
        try:
            self.vectorizer = joblib.load(
                os.path.join(self.model_path, 'vectorizer.joblib')
            )
            self.pipeline = joblib.load(
                os.path.join(self.model_path, 'classifier.joblib')
            )
            self.label_encoder = joblib.load(
                os.path.join(self.model_path, 'label_encoder.joblib')
            )
            return True
        except FileNotFoundError:
            print("No pre-trained model found.")
            return False

    def classify_requirement(self, text):
        """
        Classify a new requirement
        """
        preprocessed_text = self.preprocess_text(text)
        
        # Predict
        prediction = self.pipeline.predict([preprocessed_text])[0]
        
        # Decode label
        category = self.label_encoder.inverse_transform([prediction])[0]
        
        # Confidence
        probabilities = self.pipeline.predict_proba([preprocessed_text])[0]
        confidence = max(probabilities)
        
        return {
            'category': category,
            'confidence': confidence
        }

# Example usage
def main():
    classifier = RequirementClassifier()
    
    # Train model (first time)
    classifier.train_model('training_data.json')
    
    # Or load existing model
    classifier.load_model()
    
    # Classify a new requirement
    result = classifier.classify_requirement(
        "The system shall provide user authentication"
    )
    print(result)

if __name__ == "__main__":
    main()
