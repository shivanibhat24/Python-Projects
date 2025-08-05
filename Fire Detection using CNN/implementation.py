import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
import warnings
warnings.filterwarnings('ignore')

class FireDetectionSystem:
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.model = None
        self.history = None
        
    def create_cnn_model(self):
        """
        Create CNN model based on the paper's architecture
        Combines feature extraction with classification
        """
        model = tf.keras.models.Sequential([
            # First Convolutional Block
            tf.keras.layers.Conv2D(92, (10, 12), strides=(3, 3), 
                                 activation='relu', 
                                 input_shape=(self.img_height, self.img_width, 3)),
            tf.keras.layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2)),
            
            # Second Convolutional Block
            tf.keras.layers.Conv2D(246, (5, 5), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 3), strides=(2, 2)),
            
            # Third Convolutional Block
            tf.keras.layers.Conv2D(384, (5, 4), activation='relu'),
            tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
            
            # Flatten and Dense Layers
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.24),
            
            # First Dense Layer
            tf.keras.layers.Dense(2048, activation='relu'),
            tf.keras.layers.Dropout(0.245),
            
            # Second Dense Layer
            tf.keras.layers.Dense(1024, activation='relu'),
            tf.keras.layers.Dropout(0.45),
            
            # Output Layer (2 classes: fire/no-fire)
            tf.keras.layers.Dense(2, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            loss='categorical_crossentropy',
            optimizer=Adam(learning_rate=0.0001),
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def preprocess_image(self, image_path):
        """
        Preprocess image for prediction
        Includes smoke color and direction analysis
        """
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_width, self.img_height))
        
        # Smoke color analysis (as mentioned in paper)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Define smoke color ranges (dark colors: black, brownish shades)
        # Light smoke colors: bluish white shades
        dark_smoke_lower = np.array([0, 0, 0])
        dark_smoke_upper = np.array([180, 255, 100])
        
        light_smoke_lower = np.array([100, 0, 200])
        light_smoke_upper = np.array([130, 50, 255])
        
        # Create masks for smoke detection
        dark_smoke_mask = cv2.inRange(hsv, dark_smoke_lower, dark_smoke_upper)
        light_smoke_mask = cv2.inRange(hsv, light_smoke_lower, light_smoke_upper)
        
        # Combine masks
        smoke_mask = cv2.bitwise_or(dark_smoke_mask, light_smoke_mask)
        
        # Direction vector analysis (simplified)
        smoke_ratio = np.sum(smoke_mask > 0) / (self.img_height * self.img_width)
        
        # Normalize image
        img = img.astype('float32') / 255.0
        
        return img, smoke_ratio
    
    def create_sample_data(self, num_samples=1000):
        """
        Create sample training data for demonstration
        In practice, you would load real fire/smoke images
        """
        X = []
        y = []
        
        print("Generating sample training data...")
        
        for i in range(num_samples):
            # Generate synthetic images
            if i < num_samples // 2:
                # Fire/smoke images (class 1)
                img = np.random.rand(self.img_height, self.img_width, 3)
                # Add some "fire-like" patterns (reddish/orange colors)
                img[:, :, 0] = np.random.uniform(0.7, 1.0, (self.img_height, self.img_width))  # Red
                img[:, :, 1] = np.random.uniform(0.3, 0.7, (self.img_height, self.img_width))  # Green
                img[:, :, 2] = np.random.uniform(0.0, 0.3, (self.img_height, self.img_width))  # Blue
                label = 1
            else:
                # Normal images (class 0)
                img = np.random.rand(self.img_height, self.img_width, 3)
                label = 0
            
            X.append(img)
            y.append(label)
        
        X = np.array(X)
        y = np.array(y)
        
        # Convert labels to categorical
        y = to_categorical(y, num_classes=2)
        
        return X, y
    
    def train_model(self, X, y, epochs=10, validation_split=0.2):
        """
        Train the CNN model
        """
        if self.model is None:
            self.create_cnn_model()
        
        print("Training model...")
        print(f"Training data shape: {X.shape}")
        print(f"Labels shape: {y.shape}")
        
        # Data augmentation
        datagen = ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2
        )
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42
        )
        
        # Train model
        self.history = self.model.fit(
            datagen.flow(X_train, y_train, batch_size=32),
            steps_per_epoch=len(X_train) // 32,
            epochs=epochs,
            validation_data=(X_val, y_val),
            verbose=1
        )
        
        return self.history
    
    def predict_fire(self, image_path):
        """
        Predict if there's fire/smoke in the image
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        img, smoke_ratio = self.preprocess_image(image_path)
        img = np.expand_dims(img, axis=0)
        
        # Get prediction
        prediction = self.model.predict(img)
        probabilities = prediction[0]
        
        # Class 0: No fire, Class 1: Fire detected
        fire_probability = probabilities[1]
        predicted_class = np.argmax(probabilities)
        
        result = {
            'fire_detected': predicted_class == 1,
            'fire_probability': float(fire_probability),
            'smoke_ratio': float(smoke_ratio),
            'confidence': float(np.max(probabilities))
        }
        
        return result
    
    def real_time_detection(self, camera_index=0):
        """
        Real-time fire detection using webcam
        """
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        cap = cv2.VideoCapture(camera_index)
        
        print("Starting real-time fire detection...")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize frame
            resized_frame = cv2.resize(frame, (self.img_width, self.img_height))
            
            # Preprocess
            img = resized_frame.astype('float32') / 255.0
            img = np.expand_dims(img, axis=0)
            
            # Predict
            prediction = self.model.predict(img, verbose=0)
            fire_prob = prediction[0][1]
            
            # Display results
            color = (0, 0, 255) if fire_prob > 0.5 else (0, 255, 0)
            status = "FIRE DETECTED!" if fire_prob > 0.5 else "Normal"
            
            cv2.putText(frame, f"Status: {status}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Fire Prob: {fire_prob:.2f}", (10, 70), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
            cv2.imshow('Fire Detection', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def plot_training_history(self):
        """
        Plot training history
        """
        if self.history is None:
            print("No training history available!")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Plot accuracy
        ax1.plot(self.history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        
        # Plot loss
        ax2.plot(self.history.history['loss'], label='Training Loss')
        ax2.plot(self.history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
    
    def save_model(self, filepath):
        """
        Save trained model
        """
        if self.model is None:
            raise ValueError("No model to save!")
        
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath):
        """
        Load pre-trained model
        """
        self.model = tf.keras.models.load_model(filepath)
        print(f"Model loaded from {filepath}")

# Usage example and demonstration
def main():
    """
    Main function demonstrating the fire detection system
    """
    print("Fire Detection System using CNN")
    print("=" * 40)
    
    # Initialize system
    fire_detector = FireDetectionSystem()
    
    # Create and display model architecture
    model = fire_detector.create_cnn_model()
    print("\nModel Architecture:")
    model.summary()
    
    # Generate sample data for demonstration
    X, y = fire_detector.create_sample_data(num_samples=500)
    
    # Train model
    print("\nTraining model...")
    history = fire_detector.train_model(X, y, epochs=5)
    
    # Plot training history
    fire_detector.plot_training_history()
    
    # Example prediction (you would use real image paths)
    print("\nFire Detection System trained successfully!")
    print("\nTo use the system:")
    print("1. fire_detector.predict_fire('path/to/image.jpg')")
    print("2. fire_detector.real_time_detection()  # For webcam")
    print("3. fire_detector.save_model('fire_model.h5')")
    
    return fire_detector

# Advanced smoke analysis functions based on the paper
class SmokeAnalyzer:
    """
    Advanced smoke analysis based on direction vectors and color features
    """
    
    @staticmethod
    def analyze_smoke_direction(image):
        """
        Analyze smoke direction vector as mentioned in the paper
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        direction_vectors = []
        
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                # Calculate moments
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    # Simple direction analysis (upward movement indicates fire smoke)
                    rect = cv2.minAreaRect(contour)
                    angle = rect[2]
                    
                    direction_vectors.append({
                        'center': (cx, cy),
                        'angle': angle,
                        'area': cv2.contourArea(contour)
                    })
        
        return direction_vectors
    
    @staticmethod
    def analyze_smoke_color(image):
        """
        Analyze smoke color characteristics
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Color analysis for different smoke types
        color_features = {
            'dark_smoke': 0,
            'light_smoke': 0,
            'white_smoke': 0
        }
        
        # Dark smoke (black/brownish)
        dark_lower = np.array([0, 0, 0])
        dark_upper = np.array([180, 255, 100])
        dark_mask = cv2.inRange(hsv, dark_lower, dark_upper)
        color_features['dark_smoke'] = np.sum(dark_mask > 0)
        
        # Light smoke (bluish-white)
        light_lower = np.array([100, 0, 200])
        light_upper = np.array([130, 50, 255])
        light_mask = cv2.inRange(hsv, light_lower, light_upper)
        color_features['light_smoke'] = np.sum(light_mask > 0)
        
        # White smoke
        white_lower = np.array([0, 0, 200])
        white_upper = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, white_lower, white_upper)
        color_features['white_smoke'] = np.sum(white_mask > 0)
        
        return color_features

if __name__ == "__main__":
    # Run the main demonstration
    fire_detector = main()
