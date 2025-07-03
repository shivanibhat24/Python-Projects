"""
Model Trainer for AI Exam Monitor
Train custom models for suspicious activity detection
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import os
import json
import logging
from datetime import datetime
from typing import Tuple, List, Dict
import matplotlib.pyplot as plt
import seaborn as sns

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SuspiciousActivityClassifier:
    """
    Custom classifier for detecting suspicious activities in exam monitoring
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (64, 64, 3)):
        self.input_shape = input_shape
        self.model = None
        self.history = None
        self.class_names = [
            'normal',
            'looking_away',
            'multiple_faces',
            'phone_usage',
            'paper_notes',
            'suspicious_movement'
        ]
        
    def create_model(self, num_classes: int = 6) -> Model:
        """Create CNN model architecture"""
        model = Sequential([
            # First Convolutional Block
            Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Second Convolutional Block
            Conv2D(64, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Third Convolutional Block
            Conv2D(128, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Fourth Convolutional Block
            Conv2D(256, (3, 3), activation='relu'),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            
            # Flatten and Dense layers
            Flatten(),
            Dropout(0.5),
            Dense(512, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def prepare_data(self, data_dir: str) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data from directory structure"""
        images = []
        labels = []
        
        for class_idx, class_name in enumerate(self.class_names):
            class_dir = os.path.join(data_dir, class_name)
            if not os.path.exists(class_dir):
                logger.warning(f"Class directory {class_dir} not found, skipping")
                continue
                
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                
                # Load and preprocess image
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, self.input_shape[:2])
                img = img.astype(np.float32) / 255.0
                
                images.append(img)
                labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def create_synthetic_data(self, num_samples: int = 1000) -> Tuple[np.ndarray, np.ndarray]:
        """Create synthetic training data for demonstration"""
        logger.info("Creating synthetic training data...")
        
        images = []
        labels = []
        
        for class_idx in range(len(self.class_names)):
            for _ in range(num_samples // len(self.class_names)):
                # Create synthetic image with class-specific patterns
                img = np.random.rand(*self.input_shape).astype(np.float32)
                
                # Add class-specific patterns
                if class_idx == 0:  # normal
                    img[:, :, 1] *= 0.8  # Slightly green tint
                elif class_idx == 1:  # looking_away
                    img[:, :, 0] *= 1.2  # Red tint
                elif class_idx == 2:  # multiple_faces
                    img[:, :, 2] *= 1.2  # Blue tint
                elif class_idx == 3:  # phone_usage
                    img[20:40, 20:40, :] = 0.9  # Bright rectangle
                elif class_idx == 4:  # paper_notes
                    img[:, :, :] *= 0.6  # Darker overall
                elif class_idx == 5:  # suspicious_movement
                    img[:, :, 0] *= 1.3  # Very red
                
                images.append(img)
                labels.append(class_idx)
        
        return np.array(images), np.array(labels)
    
    def train(self, X: np.ndarray, y: np.ndarray, validation_split: float = 0.2,
              epochs: int = 50, batch_size: int = 32) -> Dict:
        """Train the model"""
        logger.info("Starting model training...")
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y
        )
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            zoom_range=0.1,
            fill_mode='nearest'
        )
        
        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_loss',
                patience=10,
                restore_best_weights=True
            ),
            ModelCheckpoint(
                'best_model.h5',
                monitor='val_loss',
                save_best_only=True
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            )
        ]
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate model
        val_loss, val_accuracy = self.model.evaluate(X_val, y_val, verbose=0)
        logger.info(f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
        
        # Generate predictions for detailed metrics
        y_pred = self.model.predict(X_val)
        y_pred_classes = np.argmax(y_pred, axis=1)
        
        # Classification report
        report = classification_report(y_val, y_pred_classes, 
                                     target_names=self.class_names, 
                                     output_dict=True)
        
        # Confusion matrix
        cm = confusion_matrix(y_val, y_pred_classes)
        
        return {
            'history': self.history.history,
            'validation_loss': val_loss,
            'validation_accuracy': val_accuracy,
            'classification_report': report,
            'confusion_matrix': cm.tolist()
        }
    
    def plot_training_history(self):
        """Plot training history"""
        if self.history is None:
            logger.warning("No training history available")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrix(self, cm: np.ndarray):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.class_names,
                    yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if self.model is None:
            logger.error("No model to save")
            return
        
        self.model.save(filepath)
        logger.info(f"Model saved to {filepath}")
        
        # Save class names and metadata
        metadata = {
            'class_names': self.class_names,
            'input_shape': self.input_shape,
            'timestamp': datetime.now().isoformat()
        }
        
        with open(filepath.replace('.h5', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def load_model(self, filepath: str):
        """Load trained model"""
        try:
            self.model = tf.keras.models.load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            
            # Load metadata if available
            metadata_path = filepath.replace('.h5', '_metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.class_names = metadata.get('class_names', self.class_names)
                    self.input_shape = tuple(metadata.get('input_shape', self.input_shape))
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
    
    def predict(self, image: np.ndarray) -> Tuple[str, float]:
        """Predict class for a single image"""
        if self.model is None:
            logger.error("No model loaded")
            return "unknown", 0.0
        
        # Preprocess image
        if image.shape != self.input_shape:
            image = cv2.resize(image, self.input_shape[:2])
        
        if image.dtype != np.float32:
            image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        # Predict
        predictions = self.model.predict(image, verbose=0)
        predicted_class_idx = np.argmax(predictions[0])
        confidence = predictions[0][predicted_class_idx]
        
        return self.class_names[predicted_class_idx], float(confidence)
    
    def predict_batch(self, images: np.ndarray) -> List[Tuple[str, float]]:
        """Predict classes for multiple images"""
        if self.model is None:
            logger.error("No model loaded")
            return []
        
        # Preprocess images
        processed_images = []
        for img in images:
            if img.shape != self.input_shape:
                img = cv2.resize(img, self.input_shape[:2])
            if img.dtype != np.float32:
                img = img.astype(np.float32) / 255.0
            processed_images.append(img)
        
        processed_images = np.array(processed_images)
        
        # Predict
        predictions = self.model.predict(processed_images, verbose=0)
        
        results = []
        for pred in predictions:
            predicted_class_idx = np.argmax(pred)
            confidence = pred[predicted_class_idx]
            results.append((self.class_names[predicted_class_idx], float(confidence)))
        
        return results

def main():
    """Main training function"""
    # Initialize classifier
    classifier = SuspiciousActivityClassifier(input_shape=(64, 64, 3))
    
    # Create model
    model = classifier.create_model()
    logger.info(f"Model created with {model.count_params()} parameters")
    
    # Option 1: Load real data from directory
    # X, y = classifier.prepare_data('path/to/your/data')
    
    # Option 2: Create synthetic data for demonstration
    X, y = classifier.create_synthetic_data(num_samples=1200)
    logger.info(f"Dataset created: {X.shape[0]} samples, {len(np.unique(y))} classes")
    
    # Train model
    results = classifier.train(X, y, epochs=30, batch_size=32)
    
    # Plot results
    classifier.plot_training_history()
    classifier.plot_confusion_matrix(np.array(results['confusion_matrix']))
    
    # Print classification report
    print("\nClassification Report:")
    for class_name, metrics in results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"{class_name}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    # Save model
    classifier.save_model('suspicious_activity_classifier.h5')
    
    # Test prediction
    test_image = X[0]  # Use first image for testing
    predicted_class, confidence = classifier.predict(test_image)
    logger.info(f"Test prediction: {predicted_class} (confidence: {confidence:.3f})")

if __name__ == "__main__":
    main()
