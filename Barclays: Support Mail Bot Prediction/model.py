import pandas as pd
import numpy as np
import re
import nltk
import pickle
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download necessary NLTK resources
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

class SupportMailClassifier:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.categories = None
        
    def preprocess_text(self, text):
        """Clean and preprocess the text data"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and numbers
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stopwords and lemmatize
        cleaned_tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        
        return ' '.join(cleaned_tokens)
    
    def train(self, texts, labels):
        """Train the model on preprocessed data"""
        # Preprocess the texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Save unique categories
        self.categories = sorted(list(set(labels)))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, labels, test_size=0.2, random_state=42
        )
        
        # Vectorize the text
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train the model
        self.model.fit(X_train_vec, y_train)
        
        # Evaluate the model
        y_pred = self.model.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        
        print(f"Model Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy
    
    def predict(self, text):
        """Predict the category of a support email"""
        processed_text = self.preprocess_text(text)
        text_vec = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_vec)[0]
        
        # Get prediction probabilities
        probabilities = self.model.predict_proba(text_vec)[0]
        confidence = max(probabilities)
        
        return {
            'category': prediction,
            'confidence': confidence,
            'all_probabilities': {
                self.categories[i]: prob for i, prob in enumerate(probabilities)
            }
        }
    
    def save_model(self, filepath):
        """Save the trained model and vectorizer"""
        model_data = {
            'vectorizer': self.vectorizer,
            'model': self.model,
            'categories': self.categories
        }
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """Load a trained model and vectorizer"""
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
            
        self.vectorizer = model_data['vectorizer']
        self.model = model_data['model']
        self.categories = model_data['categories']
        print(f"Model loaded from {filepath}")

# Example usage
if __name__ == "__main__":
    # Sample data - replace with your actual training data
    sample_data = {
        "emails": [
            "I can't login to my account, it says invalid password",
            "How do I reset my password?",
            "The application is very slow today",
            "I'm getting an error when I try to upload a file",
            "When will the new version be released?",
            "The system crashed while processing my report",
            "I need to update my billing information",
            "How do I export my data?",
            "The dashboard is not showing current data",
            "I need help setting up my account"
        ],
        "categories": [
            "login_issues",
            "password_reset",
            "performance",
            "error",
            "release_info",
            "system_crash",
            "billing",
            "data_export",
            "dashboard",
            "account_setup"
        ]
    }
    
    # Create and train the classifier
    classifier = SupportMailClassifier()
    classifier.train(sample_data["emails"], sample_data["categories"])
    
    # Save the model
    classifier.save_model("support_mail_classifier.pkl")
    
    # Test prediction
    test_email = "I forgot my password and can't log in"
    prediction = classifier.predict(test_email)
    print(f"\nTest Prediction for: '{test_email}'")
    print(f"Predicted Category: {prediction['category']}")
    print(f"Confidence: {prediction['confidence']:.4f}")
    print("All Probabilities:")
    for category, prob in prediction['all_probabilities'].items():
        print(f"  {category}: {prob:.4f}")
