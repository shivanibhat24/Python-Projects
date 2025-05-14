import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Constants
SAMPLE_RATE = 22050  # Standard audio sample rate
DURATION = 3  # Duration of audio segments in seconds
N_MELS = 128  # Number of Mel bands for the spectrogram
N_FFT = 2048  # Length of the FFT window
HOP_LENGTH = 512  # Number of samples between successive frames
BATCH_SIZE = 32
EPOCHS = 20
LEARNING_RATE = 0.001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Dataset class for audio files
class DroneAudioDataset(Dataset):
    def __init__(self, audio_files, labels, transform=None):
        self.audio_files = audio_files
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_file = self.audio_files[idx]
        label = self.labels[idx]
        
        # Load and preprocess audio
        features = self.extract_features(audio_file)
        
        if self.transform:
            features = self.transform(features)
            
        return features, label
    
    def extract_features(self, audio_file):
        # Load audio file
        try:
            y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
            
            # If audio is shorter than DURATION, pad with zeros
            if len(y) < DURATION * SAMPLE_RATE:
                y = np.pad(y, (0, DURATION * SAMPLE_RATE - len(y)))
                
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=y, 
                sr=sr, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / log_mel_spec.std()
            
            return torch.FloatTensor(log_mel_spec)
        except Exception as e:
            print(f"Error processing {audio_file}: {e}")
            # Return an empty tensor of the right shape
            return torch.zeros((N_MELS, 1 + DURATION * SAMPLE_RATE // HOP_LENGTH))

# CNN Model for Drone Detection
class DroneDetectionCNN(nn.Module):
    def __init__(self, num_classes=2):  # Default: drone vs no-drone
        super(DroneDetectionCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.dropout1 = nn.Dropout(0.25)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.dropout2 = nn.Dropout(0.25)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(kernel_size=2)
        self.dropout3 = nn.Dropout(0.25)
        
        # Calculate the size of the flattened features after convolutions
        self.feature_size = self._get_feature_size()
        
        # Fully connected layers
        self.fc1 = nn.Linear(self.feature_size, 256)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(256, num_classes)
        
    def _get_feature_size(self):
        # Calculate the output size after convolutions and pooling
        height = N_MELS
        width = 1 + DURATION * SAMPLE_RATE // HOP_LENGTH
        
        # Apply pooling operations
        height //= 8  # Three max pooling operations with kernel_size=2
        width //= 8   # Three max pooling operations with kernel_size=2
        
        return 128 * height * width
        
    def forward(self, x):
        # Input shape: (batch_size, 1, n_mels, time)
        x = x.unsqueeze(1)  # Add channel dimension
        
        # Convolutional blocks
        x = self.pool1(torch.relu(self.bn1(self.conv1(x))))
        x = self.dropout1(x)
        
        x = self.pool2(torch.relu(self.bn2(self.conv2(x))))
        x = self.dropout2(x)
        
        x = self.pool3(torch.relu(self.bn3(self.conv3(x))))
        x = self.dropout3(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected layers
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)
        
        return x

# Function to train the model
def train_model(model, train_loader, val_loader, criterion, optimizer, epochs):
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Zero the gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)
        
        # Validation phase
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                # Statistics
                val_running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)
        
        print(f"Epoch {epoch+1}/{epochs}, "
              f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}, "
              f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
    
    return train_losses, val_losses, train_accuracies, val_accuracies

# Function to evaluate the model
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # Statistics
            test_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    test_loss = test_loss / len(test_loader.dataset)
    
    # Create and print classification report
    report = classification_report(all_labels, all_preds, target_names=['No Drone', 'Drone'])
    print("\nTest Loss: {:.4f}".format(test_loss))
    print("\nClassification Report:")
    print(report)
    
    # Create confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Drone', 'Drone'], 
                yticklabels=['No Drone', 'Drone'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()
    
    return test_loss, all_labels, all_preds

# Function to visualize the spectrograms
def visualize_spectrogram(audio_file, title):
    # Load audio file
    y, sr = librosa.load(audio_file, sr=SAMPLE_RATE, duration=DURATION)
    
    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_fft=N_FFT, 
        hop_length=HOP_LENGTH, 
        n_mels=N_MELS
    )
    
    # Convert to log scale (dB)
    log_mel_spec = librosa.power_to_db(mel_spec)
    
    # Plot
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(
        log_mel_spec, 
        sr=sr, 
        hop_length=HOP_LENGTH, 
        x_axis='time', 
        y_axis='mel'
    )
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.tight_layout()
    plt.savefig(f"{title.replace(' ', '_')}.png")
    plt.close()

# Real-time drone detection class
class RealtimeDroneDetector:
    def __init__(self, model_path, threshold=0.5):
        # Load the trained model
        self.model = DroneDetectionCNN()
        self.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        self.model.to(DEVICE)
        self.model.eval()
        
        self.threshold = threshold
    
    def process_audio_chunk(self, audio_chunk):
        """Process a chunk of audio data and return drone detection result"""
        # Extract features
        try:
            # Convert to mono if needed
            if len(audio_chunk.shape) > 1:
                audio_chunk = np.mean(audio_chunk, axis=1)
            
            # Ensure fixed length
            if len(audio_chunk) < DURATION * SAMPLE_RATE:
                audio_chunk = np.pad(audio_chunk, (0, DURATION * SAMPLE_RATE - len(audio_chunk)))
            else:
                audio_chunk = audio_chunk[:DURATION * SAMPLE_RATE]
                
            # Extract mel spectrogram
            mel_spec = librosa.feature.melspectrogram(
                y=audio_chunk, 
                sr=SAMPLE_RATE, 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                n_mels=N_MELS
            )
            
            # Convert to log scale (dB)
            log_mel_spec = librosa.power_to_db(mel_spec)
            
            # Normalize
            log_mel_spec = (log_mel_spec - log_mel_spec.mean()) / (log_mel_spec.std() + 1e-9)
            
            # Convert to tensor
            features = torch.FloatTensor(log_mel_spec).unsqueeze(0).to(DEVICE)  # Add batch dimension
            
            # Make prediction
            with torch.no_grad():
                outputs = self.model(features)
                probabilities = torch.softmax(outputs, dim=1)
                drone_probability = probabilities[0][1].item()  # Probability of drone class
                
            return {
                'drone_detected': drone_probability > self.threshold,
                'confidence': drone_probability,
                'timestamp': np.datetime64('now')
            }
            
        except Exception as e:
            print(f"Error processing audio chunk: {e}")
            return {
                'drone_detected': False,
                'confidence': 0.0,
                'error': str(e),
                'timestamp': np.datetime64('now')
            }

# Example usage
def main():
    """Example of using the drone detection system"""
    # Paths to your drone and non-drone audio samples
    drone_audio_dir = "path/to/drone_audio_samples"
    ambient_audio_dir = "path/to/ambient_audio_samples"  # Non-drone audio
    
    # Collecting file paths and labels
    audio_files = []
    labels = []
    
    # Drone audio files (label 1)
    if os.path.exists(drone_audio_dir):
        for filename in os.listdir(drone_audio_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg')):
                audio_files.append(os.path.join(drone_audio_dir, filename))
                labels.append(1)  # 1 for drone
                
    # Ambient (non-drone) audio files (label 0)
    if os.path.exists(ambient_audio_dir):
        for filename in os.listdir(ambient_audio_dir):
            if filename.endswith(('.wav', '.mp3', '.ogg')):
                audio_files.append(os.path.join(ambient_audio_dir, filename))
                labels.append(0)  # 0 for non-drone
    
    # Check if we have any files
    if not audio_files:
        print("No audio files found. Please check the directories.")
        print("Creating simulated data for demonstration purposes...")
        
        # Create dummy data directory
        os.makedirs("dummy_data", exist_ok=True)
        
        # Generate some synthetic data for demonstration
        for i in range(20):
            # Drone-like sounds (higher frequency content)
            duration = 3.0
            sample_rate = SAMPLE_RATE
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            
            # Drone sound - mixture of frequencies with some modulation
            if i < 10:
                # Base frequency around 100-200 Hz (motor hum)
                f1 = np.random.uniform(100, 200)
                # Higher frequencies for propeller noise
                f2 = np.random.uniform(2000, 6000)
                # Modulation frequency
                f_mod = np.random.uniform(5, 15)
                
                # Create drone sound with amplitude modulation
                drone_sound = 0.5 * np.sin(2 * np.pi * f1 * t) + \
                              0.3 * np.sin(2 * np.pi * f2 * t) * \
                              (1 + 0.2 * np.sin(2 * np.pi * f_mod * t))
                
                # Add some noise
                drone_sound += 0.1 * np.random.normal(0, 1, len(t))
                
                # Normalize
                drone_sound = 0.9 * drone_sound / np.max(np.abs(drone_sound))
                
                # Save as WAV file
                filename = os.path.join("dummy_data", f"drone_{i}.wav")
                librosa.output.write_wav(filename, drone_sound, sample_rate)
                
                audio_files.append(filename)
                labels.append(1)  # Drone
                
            # Ambient sounds (lower frequency, less structured)
            else:
                # Create ambient sound with pink noise
                ambient_sound = np.random.normal(0, 1, int(sample_rate * duration))
                
                # Filter to make it more like pink noise
                ambient_sound = 0.8 * librosa.effects.preemphasis(ambient_sound, coef=0.95)
                
                # Add some occasional low frequency rumble
                if np.random.random() > 0.5:
                    f_rumble = np.random.uniform(30, 80)
                    ambient_sound += 0.3 * np.sin(2 * np.pi * f_rumble * t)
                
                # Normalize
                ambient_sound = 0.9 * ambient_sound / np.max(np.abs(ambient_sound))
                
                # Save as WAV file
                filename = os.path.join("dummy_data", f"ambient_{i-10}.wav")
                librosa.output.write_wav(filename, ambient_sound, sample_rate)
                
                audio_files.append(filename)
                labels.append(0)  # Non-drone
    
    # Split the data into training, validation, and test sets
    train_files, test_files, train_labels, test_labels = train_test_split(
        audio_files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    train_files, val_files, train_labels, val_labels = train_test_split(
        train_files, train_labels, test_size=0.25, random_state=42, stratify=train_labels
    )
    
    print(f"Training samples: {len(train_files)}")
    print(f"Validation samples: {len(val_files)}")
    print(f"Test samples: {len(test_files)}")
    
    # Create datasets
    train_dataset = DroneAudioDataset(train_files, train_labels)
    val_dataset = DroneAudioDataset(val_files, val_labels)
    test_dataset = DroneAudioDataset(test_files, test_labels)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)
    
    # Visualize some spectrograms if files exist
    if len(audio_files) > 0:
        drone_files = [f for f, l in zip(audio_files, labels) if l == 1]
        non_drone_files = [f for f, l in zip(audio_files, labels) if l == 0]
        
        if drone_files:
            visualize_spectrogram(drone_files[0], 'Drone Sound Spectrogram')
        if non_drone_files:
            visualize_spectrogram(non_drone_files[0], 'Ambient Sound Spectrogram')
    
    # Initialize the model
    model = DroneDetectionCNN()
    model.to(DEVICE)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    train_losses, val_losses, train_accs, val_accs = train_model(
        model, train_loader, val_loader, criterion, optimizer, EPOCHS
    )
    
    # Plot training and validation metrics
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig('training_metrics.png')
    plt.close()
    
    # Evaluate the model on the test set
    test_loss, true_labels, pred_labels = evaluate_model(model, test_loader, criterion)
    
    # Save the model
    torch.save(model.state_dict(), 'drone_detection_model.pth')
    print("Model saved as 'drone_detection_model.pth'")
    
    # Example of real-time detector usage
    print("\nInitializing real-time drone detector...")
    detector = RealtimeDroneDetector('drone_detection_model.pth')
    
    # Simulate processing of an audio chunk
    # In a real application, this would come from a microphone or other audio source
    if test_files:
        print("Simulating real-time detection with test file...")
        test_audio, _ = librosa.load(test_files[0], sr=SAMPLE_RATE, duration=DURATION)
        result = detector.process_audio_chunk(test_audio)
        
        print(f"Drone detected: {result['drone_detected']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Timestamp: {result['timestamp']}")

if __name__ == "__main__":
    main()
