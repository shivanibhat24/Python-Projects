import numpy as np
import librosa
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Function to extract features from an audio file
def extract_features(file_path):
    """
    Extract features using librosa from an audio file
    
    Args:
        file_path (str): Path to the audio file
        
    Returns:
        feature_vector (np.ndarray): Vector containing extracted features
    """
    # Load the audio file
    y, sr = librosa.load(file_path, sr=None)
    
    # Extract features
    # 1. MFCCs (Mel-Frequency Cepstral Coefficients)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfccs, axis=1)
    
    # 2. Spectral Centroid
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_centroid_mean = np.mean(spectral_centroid)
    
    # 3. Spectral Rolloff
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_rolloff_mean = np.mean(spectral_rolloff)
    
    # 4. Zero Crossing Rate
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
    zero_crossing_rate_mean = np.mean(zero_crossing_rate)
    
    # 5. Chroma Features (represent the 12 different pitch classes)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    
    # 6. Tempo (Beats Per Minute)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    
    # Combine all features into a single vector
    feature_vector = np.concatenate([
        mfcc_mean, 
        [spectral_centroid_mean], 
        [spectral_rolloff_mean],
        [zero_crossing_rate_mean],
        chroma_mean,
        [tempo]
    ])
    
    return feature_vector

# Function to process multiple files
def build_dataset(file_paths, genres):
    """
    Build a dataset from multiple audio files
    
    Args:
        file_paths (list): List of file paths
        genres (list): Corresponding genre labels
        
    Returns:
        features_df (pd.DataFrame): DataFrame containing features and labels
    """
    features = []
    
    for file_path in file_paths:
        feature_vector = extract_features(file_path)
        features.append(feature_vector)
    
    # Create feature names
    feature_names = [f'mfcc_{i}' for i in range(13)]
    feature_names += ['spectral_centroid', 'spectral_rolloff', 'zero_crossing_rate']
    feature_names += [f'chroma_{i}' for i in range(12)]
    feature_names += ['tempo']
    
    # Convert to DataFrame
    features_df = pd.DataFrame(features, columns=feature_names)
    features_df['genre'] = genres
    
    return features_df

# Train a model
def train_model(features_df):
    """
    Train a machine learning model on the extracted features
    
    Args:
        features_df (pd.DataFrame): DataFrame containing features and labels
        
    Returns:
        model: Trained model
        scaler: Fitted scaler
    """
    # Split features and labels
    X = features_df.drop('genre', axis=1)
    y = features_df['genre']
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate the model
    y_pred = model.predict(X_test_scaled)
    print(classification_report(y_test, y_pred))
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    print("\nTop 10 most important features:")
    print(feature_importance.head(10))
    
    return model, scaler

# Visualize features
def visualize_audio_features(file_path, title="Audio Features Visualization"):
    """
    Visualize various audio features
    
    Args:
        file_path (str): Path to the audio file
        title (str): Title for the plot
    """
    y, sr = librosa.load(file_path, sr=None)
    
    plt.figure(figsize=(15, 10))
    
    # Plot waveform
    plt.subplot(3, 1, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    
    # Plot spectrogram
    plt.subplot(3, 1, 2)
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Plot MFCCs
    plt.subplot(3, 1, 3)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCCs')
    
    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.9)
    plt.show()

# Example usage
def classify_song(model, scaler, file_path):
    """
    Classify a single song
    
    Args:
        model: Trained model
        scaler: Fitted scaler
        file_path (str): Path to the audio file
        
    Returns:
        genre (str): Predicted genre
    """
    # Extract features
    features = extract_features(file_path)
    
    # Scale features
    features_scaled = scaler.transform([features])
    
    # Make prediction
    prediction = model.predict(features_scaled)
    
    return prediction[0]

# Main execution flow example
if __name__ == "__main__":
    # Example paths - you would replace these with actual file paths
    file_paths = [
        'path/to/rock_song1.mp3',
        'path/to/rock_song2.mp3',
        'path/to/jazz_song1.mp3',
        'path/to/jazz_song2.mp3',
        'path/to/classical_song1.mp3',
        'path/to/classical_song2.mp3'
    ]
    
    genres = [
        'rock',
        'rock',
        'jazz',
        'jazz',
        'classical',
        'classical'
    ]
    
    # Build dataset
    features_df = build_dataset(file_paths, genres)
    
    # Train model
    model, scaler = train_model(features_df)
    
    # Classify a new song
    new_song_path = 'path/to/unknown_song.mp3'
    predicted_genre = classify_song(model, scaler, new_song_path)
    print(f"The predicted genre is: {predicted_genre}")
    
    # Visualize features of the new song
    visualize_audio_features(new_song_path, f"Audio Features - Predicted as {predicted_genre}")
