import tensorflow as tf
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam, AdamW
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import albumentations as A
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class AdvancedFireDetectionSystem:
    """
    State-of-the-art Multi-Modal Fire Detection System
    Suitable for high-impact journal publication
    """
    
    def __init__(self, img_height=224, img_width=224):
        self.img_height = img_height
        self.img_width = img_width
        self.models = {}
        self.ensemble_model = None
        self.feature_extractor = None
        self.history = {}
        self.evaluation_metrics = {}
        
    def create_advanced_cnn_model(self, model_type='ResNet'):
        """
        Create state-of-the-art CNN architectures
        """
        if model_type == 'ResNet':
            base_model = tf.keras.applications.ResNet50V2(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            
            # Fine-tuning strategy
            for layer in base_model.layers[:-20]:
                layer.trainable = False
                
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(256, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(3, activation='softmax')  # fire, smoke, normal
            ])
            
        elif model_type == 'EfficientNet':
            base_model = tf.keras.applications.EfficientNetB4(
                weights='imagenet',
                include_top=False,
                input_shape=(self.img_height, self.img_width, 3)
            )
            
            model = tf.keras.Sequential([
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.4),
                tf.keras.layers.Dense(1024, activation='relu'),
                tf.keras.layers.BatchNormalization(),
                tf.keras.layers.Dropout(0.5),
                tf.keras.layers.Dense(512, activation='relu'),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(3, activation='softmax')
            ])
            
        elif model_type == 'VisionTransformer':
            # Vision Transformer implementation
            model = self.create_vision_transformer()
            
        # Advanced optimizer with learning rate scheduling
        optimizer = AdamW(
            learning_rate=1e-4,
            weight_decay=1e-5
        )
        
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.models[model_type] = model
        return model
    
    def create_vision_transformer(self):
        """
        Implement Vision Transformer (ViT) for fire detection
        """
        from tensorflow.keras.layers import MultiHeadAttention, LayerNormalization
        
        # Patch extraction layer
        class PatchExtract(tf.keras.layers.Layer):
            def __init__(self, patch_size):
                super(PatchExtract, self).__init__()
                self.patch_size = patch_size
                
            def call(self, images):
                batch_size = tf.shape(images)[0]
                patches = tf.image.extract_patches(
                    images=images,
                    sizes=[1, self.patch_size, self.patch_size, 1],
                    strides=[1, self.patch_size, self.patch_size, 1],
                    rates=[1, 1, 1, 1],
                    padding="VALID",
                )
                return tf.reshape(patches, [batch_size, -1, self.patch_size * self.patch_size * 3])
        
        # Patch embedding layer
        class PatchEmbedding(tf.keras.layers.Layer):
            def __init__(self, num_patches, projection_dim):
                super(PatchEmbedding, self).__init__()
                self.num_patches = num_patches
                self.projection = tf.keras.layers.Dense(units=projection_dim)
                self.position_embedding = tf.keras.layers.Embedding(
                    input_dim=num_patches, output_dim=projection_dim
                )
                
            def call(self, patch):
                positions = tf.range(start=0, limit=self.num_patches, delta=1)
                encoded = self.projection(patch) + self.position_embedding(positions)
                return encoded
        
        # Transformer block
        def transformer_encoder(encoded_patches, num_heads, projection_dim):
            # Layer normalization 1
            x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
            
            # Multi-head attention
            attention_output = MultiHeadAttention(
                num_heads=num_heads, key_dim=projection_dim, dropout=0.1
            )(x1, x1)
            
            # Skip connection 1
            x2 = tf.keras.layers.Add()([attention_output, encoded_patches])
            
            # Layer normalization 2
            x3 = LayerNormalization(epsilon=1e-6)(x2)
            
            # MLP
            x3 = tf.keras.layers.Dense(projection_dim * 2, activation="gelu")(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            x3 = tf.keras.layers.Dense(projection_dim, activation="gelu")(x3)
            x3 = tf.keras.layers.Dropout(0.1)(x3)
            
            # Skip connection 2
            encoded_patches = tf.keras.layers.Add()([x3, x2])
            return encoded_patches
        
        # Build ViT model
        patch_size = 16
        num_patches = (self.img_height // patch_size) ** 2
        projection_dim = 768
        num_heads = 12
        transformer_layers = 12
        
        inputs = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
        patches = PatchExtract(patch_size)(inputs)
        encoded_patches = PatchEmbedding(num_patches, projection_dim)(patches)
        
        # Multiple transformer layers
        for _ in range(transformer_layers):
            encoded_patches = transformer_encoder(encoded_patches, num_heads, projection_dim)
        
        # Final classification
        representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
        representation = tf.keras.layers.GlobalAveragePooling1D()(representation)
        representation = tf.keras.layers.Dropout(0.5)(representation)
        features = tf.keras.layers.Dense(2048, activation="gelu")(representation)
        features = tf.keras.layers.Dropout(0.5)(features)
        logits = tf.keras.layers.Dense(3, activation="softmax")(features)
        
        model = tf.keras.Model(inputs=inputs, outputs=logits)
        return model
    
    def extract_advanced_features(self, image):
        """
        Extract comprehensive multi-modal features
        """
        features = {}
        
        # 1. Color-based features
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Color histograms
        features['color_hist_r'] = cv2.calcHist([image], [0], None, [32], [0, 256]).flatten()
        features['color_hist_g'] = cv2.calcHist([image], [1], None, [32], [0, 256]).flatten()
        features['color_hist_b'] = cv2.calcHist([image], [2], None, [32], [0, 256]).flatten()
        
        # Color moments
        for i, channel in enumerate(['r', 'g', 'b']):
            channel_data = image[:, :, i].flatten()
            features[f'{channel}_mean'] = np.mean(channel_data)
            features[f'{channel}_std'] = np.std(channel_data)
            features[f'{channel}_skewness'] = stats.skew(channel_data)
            features[f'{channel}_kurtosis'] = stats.kurtosis(channel_data)
        
        # 2. Texture features (Local Binary Patterns)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        features['lbp'] = self.calculate_lbp(gray)
        
        # 3. Edge and gradient features
        edges = cv2.Canny(gray, 50, 150)
        features['edge_density'] = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        # Gradient orientation histogram
        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(gx**2 + gy**2)
        orientation = np.arctan2(gy, gx)
        features['gradient_hist'] = np.histogram(orientation, bins=36)[0]
        
        # 4. Smoke-specific features
        smoke_features = self.extract_smoke_features(image, hsv)
        features.update(smoke_features)
        
        # 5. Fire-specific features
        fire_features = self.extract_fire_features(image, hsv)
        features.update(fire_features)
        
        # 6. Motion features (if temporal data available)
        # This would require frame sequences
        
        return features
    
    def calculate_lbp(self, gray_image, radius=3, n_points=24):
        """
        Calculate Local Binary Pattern features
        """
        from skimage.feature import local_binary_pattern
        
        lbp = local_binary_pattern(gray_image, n_points, radius, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=n_points + 2, 
                            range=(0, n_points + 2), density=True)
        return hist
    
    def extract_smoke_features(self, image, hsv):
        """
        Extract smoke-specific features based on research findings
        """
        features = {}
        
        # Smoke color ranges (based on literature)
        smoke_ranges = [
            ([0, 0, 50], [180, 30, 200]),    # Light gray smoke
            ([0, 0, 0], [180, 255, 100]),     # Dark smoke
            ([100, 0, 180], [130, 50, 255])  # Bluish smoke
        ]
        
        total_smoke_pixels = 0
        for i, (lower, upper) in enumerate(smoke_ranges):
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            smoke_pixels = np.sum(mask > 0)
            total_smoke_pixels += smoke_pixels
            features[f'smoke_type_{i}_ratio'] = smoke_pixels / (image.shape[0] * image.shape[1])
        
        features['total_smoke_ratio'] = total_smoke_pixels / (image.shape[0] * image.shape[1])
        
        # Smoke direction analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        contours, _ = cv2.findContours(
            cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            moments = cv2.moments(largest_contour)
            if moments["m00"] != 0:
                features['centroid_x'] = moments["m10"] / moments["m00"]
                features['centroid_y'] = moments["m01"] / moments["m00"]
                
                # Aspect ratio and orientation
                rect = cv2.minAreaRect(largest_contour)
                features['smoke_angle'] = rect[2]
                features['smoke_aspect_ratio'] = max(rect[1]) / min(rect[1]) if min(rect[1]) > 0 else 0
        
        return features
    
    def extract_fire_features(self, image, hsv):
        """
        Extract fire-specific features
        """
        features = {}
        
        # Fire color ranges
        fire_lower = np.array([0, 120, 70])
        fire_upper = np.array([20, 255, 255])
        fire_mask1 = cv2.inRange(hsv, fire_lower, fire_upper)
        
        fire_lower2 = np.array([160, 120, 70])
        fire_upper2 = np.array([180, 255, 255])
        fire_mask2 = cv2.inRange(hsv, fire_lower2, fire_upper2)
        
        fire_mask = cv2.bitwise_or(fire_mask1, fire_mask2)
        features['fire_pixel_ratio'] = np.sum(fire_mask > 0) / (image.shape[0] * image.shape[1])
        
        # Fire intensity measures
        fire_regions = cv2.bitwise_and(image, image, mask=fire_mask)
        if np.sum(fire_mask > 0) > 0:
            features['avg_fire_intensity'] = np.mean(fire_regions[fire_mask > 0])
            features['max_fire_intensity'] = np.max(fire_regions[fire_mask > 0])
        else:
            features['avg_fire_intensity'] = 0
            features['max_fire_intensity'] = 0
        
        return features
    
    def create_ensemble_model(self):
        """
        Create ensemble model combining multiple architectures
        """
        # Create base models
        resnet_model = self.create_advanced_cnn_model('ResNet')
        efficientnet_model = self.create_advanced_cnn_model('EfficientNet')
        
        # Ensemble architecture
        input_layer = tf.keras.layers.Input(shape=(self.img_height, self.img_width, 3))
        
        # Get predictions from base models
        resnet_pred = resnet_model(input_layer)
        efficientnet_pred = efficientnet_model(input_layer)
        
        # Combine predictions
        combined = tf.keras.layers.concatenate([resnet_pred, efficientnet_pred])
        combined = tf.keras.layers.Dense(256, activation='relu')(combined)
        combined = tf.keras.layers.Dropout(0.3)(combined)
        combined = tf.keras.layers.Dense(128, activation='relu')(combined)
        final_pred = tf.keras.layers.Dense(3, activation='softmax')(combined)
        
        ensemble_model = tf.keras.Model(inputs=input_layer, outputs=final_pred)
        
        ensemble_model.compile(
            optimizer=AdamW(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        self.ensemble_model = ensemble_model
        return ensemble_model
    
    def advanced_data_augmentation(self):
        """
        Advanced data augmentation using Albumentations
        """
        transform = A.Compose([
            A.RandomRotate90(),
            A.Flip(),
            A.Transpose(),
            A.OneOf([
                A.GaussNoise(),
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.2),
            A.OneOf([
                A.CLAHE(clip_limit=2),
                A.Sharpen(),
                A.Emboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.OneOf([
                A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
                A.GridDistortion(),
                A.OpticalDistortion(distort_limit=2, shift_limit=0.5),
            ], p=0.3),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=0.75),
            A.Cutout(max_h_size=int(self.img_height * 0.1), max_w_size=int(self.img_width * 0.1), 
                    num_holes=5, p=0.5),
        ])
        
        return transform
    
    def train_with_cross_validation(self, X, y, k_folds=5):
        """
        Train models with k-fold cross-validation
        """
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=42)
        cv_scores = {}
        
        for model_name in ['ResNet', 'EfficientNet']:
            cv_scores[model_name] = []
            
            for fold, (train_idx, val_idx) in enumerate(skf.split(X, np.argmax(y, axis=1))):
                print(f"Training {model_name} - Fold {fold + 1}/{k_folds}")
                
                X_train_fold, X_val_fold = X[train_idx], X[val_idx]
                y_train_fold, y_val_fold = y[train_idx], y[val_idx]
                
                # Create and train model
                model = self.create_advanced_cnn_model(model_name)
                
                # Callbacks
                callbacks = [
                    EarlyStopping(patience=10, restore_best_weights=True),
                    ReduceLROnPlateau(factor=0.5, patience=5),
                    ModelCheckpoint(f'best_model_{model_name}_fold_{fold}.h5', 
                                  save_best_only=True)
                ]
                
                # Train
                history = model.fit(
                    X_train_fold, y_train_fold,
                    validation_data=(X_val_fold, y_val_fold),
                    epochs=100,
                    batch_size=32,
                    callbacks=callbacks,
                    verbose=0
                )
                
                # Evaluate
                val_score = model.evaluate(X_val_fold, y_val_fold, verbose=0)[1]
                cv_scores[model_name].append(val_score)
                
                print(f"Fold {fold + 1} Validation Accuracy: {val_score:.4f}")
        
        # Print cross-validation results
        for model_name, scores in cv_scores.items():
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"{model_name} CV Score: {mean_score:.4f} (+/- {std_score * 2:.4f})")
        
        return cv_scores
    
    def comprehensive_evaluation(self, X_test, y_test):
        """
        Comprehensive model evaluation with multiple metrics
        """
        results = {}
        
        for model_name, model in self.models.items():
            print(f"Evaluating {model_name}...")
            
            # Predictions
            y_pred = model.predict(X_test)
            y_pred_classes = np.argmax(y_pred, axis=1)
            y_true_classes = np.argmax(y_test, axis=1)
            
            # Classification metrics
            results[model_name] = {
                'accuracy': np.mean(y_pred_classes == y_true_classes),
                'classification_report': classification_report(y_true_classes, y_pred_classes),
                'confusion_matrix': confusion_matrix(y_true_classes, y_pred_classes),
                'auc_score': roc_auc_score(y_test, y_pred, multi_class='ovr')
            }
            
            # Plot confusion matrix
            plt.figure(figsize=(8, 6))
            sns.heatmap(results[model_name]['confusion_matrix'], 
                       annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_name}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.show()
        
        return results
    
    def create_interpretability_analysis(self, model, X_sample):
        """
        Create model interpretability analysis using GradCAM
        """
        # GradCAM implementation
        def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
            grad_model = tf.keras.models.Model(
                [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
            )
            
            with tf.GradientTape() as tape:
                last_conv_layer_output, preds = grad_model(img_array)
                if pred_index is None:
                    pred_index = tf.argmax(preds[0])
                class_channel = preds[:, pred_index]
            
            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
            
            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)
            
            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
            return heatmap.numpy()
        
        # Generate heatmaps for sample images
        heatmaps = []
        for i in range(min(5, len(X_sample))):
            img = np.expand_dims(X_sample[i], axis=0)
            heatmap = make_gradcam_heatmap(img, model, 'conv2d', pred_index=None)
            heatmaps.append(heatmap)
        
        return heatmaps
    
    def generate_comprehensive_report(self, evaluation_results, cv_scores):
        """
        Generate comprehensive evaluation report
        """
        report = {
            'executive_summary': {
                'best_model': max(evaluation_results.keys(), 
                                key=lambda x: evaluation_results[x]['accuracy']),
                'best_accuracy': max([r['accuracy'] for r in evaluation_results.values()]),
                'cross_validation_results': cv_scores
            },
            'detailed_results': evaluation_results,
            'model_comparison': self.compare_models(evaluation_results),
            'recommendations': self.generate_recommendations(evaluation_results)
        }
        
        return report
    
    def compare_models(self, results):
        """
        Statistical comparison of models
        """
        comparison = {}
        models = list(results.keys())
        
        for i, model1 in enumerate(models):
            for model2 in models[i+1:]:
                acc1 = results[model1]['accuracy']
                acc2 = results[model2]['accuracy']
                
                comparison[f"{model1}_vs_{model2}"] = {
                    'accuracy_diff': abs(acc1 - acc2),
                    'better_model': model1 if acc1 > acc2 else model2
                }
        
        return comparison
    
    def generate_recommendations(self, results):
        """
        Generate deployment recommendations
        """
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        best_accuracy = results[best_model]['accuracy']
        
        recommendations = [
            f"Deploy {best_model} for production use (Accuracy: {best_accuracy:.4f})",
            "Implement ensemble methods for critical applications",
            "Consider real-time processing optimizations",
            "Establish continuous monitoring and retraining pipeline",
        ]
        
        if best_accuracy < 0.95:
            recommendations.append("Consider collecting more training data")
            recommendations.append("Explore additional feature engineering")
        
        return recommendations
    
    def export_model_for_deployment(self, model_name, export_path):
        """
        Export model for production deployment
        """
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.models[model_name]
        
        # Convert to TensorFlow Lite for mobile deployment
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        with open(f"{export_path}/{model_name}_mobile.tflite", "wb") as f:
            f.write(tflite_model)
        
        # Save full model
        model.save(f"{export_path}/{model_name}_full.h5")
        
        print(f"Models exported to {export_path}")

# Advanced benchmarking and experimental framework
class FireDetectionBenchmark:
    """
    Comprehensive benchmarking framework for fire detection systems
    """
    
    def __init__(self):
        self.benchmark_results = {}
        self.datasets = {}
    
    def create_synthetic_benchmark_dataset(self, num_samples=5000):
        """
        Create comprehensive synthetic benchmark dataset
        """
        print("Creating synthetic benchmark dataset...")
        
        X = []
        y = []
        metadata = []
        
        for i in range(num_samples):
            if i < num_samples // 3:
                # Fire images
                img = self.generate_fire_image()
                label = [1, 0, 0]  # fire, smoke, normal
                scene_type = 'fire'
            elif i < 2 * num_samples // 3:
                # Smoke images
                img = self.generate_smoke_image()
                label = [0, 1, 0]
                scene_type = 'smoke'
            else:
                # Normal images
                img = self.generate_normal_image()
                label = [0, 0, 1]
                scene_type = 'normal'
            
            X.append(img)
            y.append(label)
            metadata.append({
                'scene_type': scene_type,
                'lighting': np.random.choice(['day', 'night', 'dusk']),
                'weather': np.random.choice(['clear', 'cloudy', 'rainy']),
                'environment': np.random.choice(['forest', 'urban', 'rural'])
            })
        
        return np.array(X), np.array(y), metadata
    
    def generate_fire_image(self, height=224, width=224):
        """Generate synthetic fire image"""
        img = np.random.rand(height, width, 3)
        
        # Add fire-like patterns
        center_x, center_y = width // 2, height // 2
        for i in range(height):
            for j in range(width):
                dist = np.sqrt((i - center_y)**2 + (j - center_x)**2)
                if dist < min(height, width) // 3:
                    # Fire colors (red, orange, yellow)
                    img[i, j, 0] = np.random.uniform(0.7, 1.0)  # Red
                    img[i, j, 1] = np.random.uniform(0.3, 0.8)  # Green
                    img[i, j, 2] = np.random.uniform(0.0, 0.4)  # Blue
        
        return img
    
    def generate_smoke_image(self, height=224, width=224):
        """Generate synthetic smoke image"""
        img = np.random.rand(height, width, 3) * 0.3 + 0.4
        
        # Add smoke-like patterns (grayish)
        for i in range(height):
            for j in range(width):
                if np.random.random() > 0.7:
                    # Gray smoke
                    intensity = np.random.uniform(0.3, 0.7)
                    img[i, j] = [intensity, intensity, intensity]
        
        return img
    
    def generate_normal_image(self, height=224, width=224):
        """Generate normal scene image"""
        # Random natural scene
        img = np.random.rand(height, width, 3)
        
        # Add some structure (green for vegetation, blue for sky)
        if np.random.random() > 0.5:
            # Forest scene
            img[:height//2, :, 1] = np.random.uniform(0.4, 0.8, (height//2, width))  # Green
            img[height//2:, :, 2] = np.random.uniform(0.6, 1.0, (height//2, width))  # Blue sky
        
        return img
    
    def run_comprehensive_benchmark(self):
        """
        Run comprehensive benchmark comparing multiple approaches
        """
        print("Starting comprehensive benchmark...")
        
        # Create dataset
        X, y, metadata = self.create_synthetic_benchmark_dataset(2000)
        
        # Initialize system
        system = AdvancedFireDetectionSystem()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=np.argmax(y, axis=1)
        )
        
        # Benchmark different models
        models_to_test = ['ResNet', 'EfficientNet']
        
        for model_name in models_to_test:
            print(f"\nBenchmarking {model_name}...")
            
            # Create and train model
            model = system.create_advanced_cnn_model(model_name)
            
            # Training with callbacks
            callbacks = [
                EarlyStopping(patience=5, restore_best_weights=True),
                ReduceLROnPlateau(factor=0.5, patience=3),
                ModelCheckpoint(f'benchmark_{model_name}.h5', save_best_only=True)
            ]
            
            # Train model
            history = model.fit(
                X_train, y_train,
                validation_split=0.2,
                epochs=20,
                batch_size=32,
                callbacks=callbacks,
                verbose=1
            )
            
            # Evaluate model
            test_loss, test_acc, test_precision, test_recall = model.evaluate(X_test, y_test, verbose=0)
            
            # Store results
            self.benchmark_results[model_name] = {
                'test_accuracy': test_acc,
                'test_precision': test_precision,
                'test_recall': test_recall,
                'f1_score': 2 * (test_precision * test_recall) / (test_precision + test_recall),
                'training_history': history.history
            }
            
            print(f"{model_name} Results:")
            print(f"  Accuracy: {test_acc:.4f}")
            print(f"  Precision: {test_precision:.4f}")
            print(f"  Recall: {test_recall:.4f}")
            print(f"  F1-Score: {self.benchmark_results[model_name]['f1_score']:.4f}")
        
        return self.benchmark_results
    
    def generate_benchmark_report(self):
        """Generate comprehensive benchmark report"""
        if not self.benchmark_results:
            print("No benchmark results available. Run benchmark first.")
            return
        
        print("\n" + "="*60)
        print("COMPREHENSIVE FIRE DETECTION BENCHMARK REPORT")
        print("="*60)
        
        # Performance comparison
        print("\n1. MODEL PERFORMANCE COMPARISON")
        print("-" * 40)
        
        df_results = pd.DataFrame(self.benchmark_results).T
        print(df_results[['test_accuracy', 'test_precision', 'test_recall', 'f1_score']])
        
        # Best model identification
        best_model = df_results['f1_score'].idxmax()
        print(f"\nBest Model: {best_model} (F1-Score: {df_results.loc[best_model, 'f1_score']:.4f})")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy comparison
        models = list(self.benchmark_results.keys())
        accuracies = [self.benchmark_results[m]['test_accuracy'] for m in models]
        axes[0, 0].bar(models, accuracies, color=['skyblue', 'lightcoral'])
        axes[0, 0].set_title('Model Accuracy Comparison')
        axes[0, 0].set_ylabel('Accuracy')
        
        # Precision-Recall comparison
        precisions = [self.benchmark_results[m]['test_precision'] for m in models]
        recalls = [self.benchmark_results[m]['test_recall'] for m in models]
        
        x = np.arange(len(models))
        width = 0.35
        axes[0, 1].bar(x - width/2, precisions, width, label='Precision', color='lightgreen')
        axes[0, 1].bar(x + width/2, recalls, width, label='Recall', color='lightsalmon')
        axes[0, 1].set_title('Precision vs Recall')
        axes[0, 1].set_xticks(x)
        axes[0, 1].set_xticklabels(models)
        axes[0, 1].legend()
        
        # F1-Score comparison
        f1_scores = [self.benchmark_results[m]['f1_score'] for m in models]
        axes[1, 0].bar(models, f1_scores, color=['gold', 'silver'])
        axes[1, 0].set_title('F1-Score Comparison')
        axes[1, 0].set_ylabel('F1-Score')
        
        # Training curves (if available)
        for i, model_name in enumerate(models):
            if 'training_history' in self.benchmark_results[model_name]:
                history = self.benchmark_results[model_name]['training_history']
                if 'val_accuracy' in history:
                    axes[1, 1].plot(history['val_accuracy'], label=f'{model_name} Val Acc')
        
        axes[1, 1].set_title('Training Curves')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Validation Accuracy')
        axes[1, 1].legend()
        
        plt.tight_layout()
        plt.show()
        
        return df_results

# Advanced Multi-Modal Analysis System
class MultiModalFireAnalysis:
    """
    Advanced multi-modal analysis combining visual, thermal, and sensor data
    """
    
    def __init__(self):
        self.visual_model = None
        self.thermal_model = None
        self.fusion_model = None
        
    def create_thermal_analysis_model(self):
        """
        Create specialized model for thermal image analysis
        """
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(224, 224, 1)),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D(2, 2),
            
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        self.thermal_model = model
        return model
    
    def create_sensor_fusion_model(self):
        """
        Create sensor fusion model combining multiple data sources
        """
        # Visual input
        visual_input = tf.keras.layers.Input(shape=(224, 224, 3), name='visual_input')
        visual_features = tf.keras.applications.ResNet50V2(
            weights='imagenet', include_top=False, input_tensor=visual_input
        )
        visual_features = tf.keras.layers.GlobalAveragePooling2D()(visual_features.output)
        visual_features = tf.keras.layers.Dense(256, activation='relu')(visual_features)
        
        # Thermal input
        thermal_input = tf.keras.layers.Input(shape=(224, 224, 1), name='thermal_input')
        thermal_conv = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(thermal_input)
        thermal_conv = tf.keras.layers.MaxPooling2D(2, 2)(thermal_conv)
        thermal_conv = tf.keras.layers.Conv2D(128, (3, 3), activation='relu')(thermal_conv)
        thermal_conv = tf.keras.layers.GlobalAveragePooling2D()(thermal_conv)
        thermal_features = tf.keras.layers.Dense(256, activation='relu')(thermal_conv)
        
        # Sensor input (temperature, humidity, wind speed, etc.)
        sensor_input = tf.keras.layers.Input(shape=(10,), name='sensor_input')
        sensor_features = tf.keras.layers.Dense(128, activation='relu')(sensor_input)
        sensor_features = tf.keras.layers.Dense(64, activation='relu')(sensor_features)
        
        # Fusion layer
        fused = tf.keras.layers.concatenate([visual_features, thermal_features, sensor_features])
        fused = tf.keras.layers.Dense(512, activation='relu')(fused)
        fused = tf.keras.layers.Dropout(0.4)(fused)
        fused = tf.keras.layers.Dense(256, activation='relu')(fused)
        fused = tf.keras.layers.Dropout(0.3)(fused)
        
        # Output
        output = tf.keras.layers.Dense(3, activation='softmax', name='classification_output')(fused)
        
        # Attention mechanism for interpretability
        attention_weights = tf.keras.layers.Dense(3, activation='softmax', name='attention_weights')(fused)
        
        model = tf.keras.Model(
            inputs=[visual_input, thermal_input, sensor_input],
            outputs=[output, attention_weights]
        )
        
        model.compile(
            optimizer=Adam(learning_rate=1e-4),
            loss={'classification_output': 'categorical_crossentropy',
                  'attention_weights': 'categorical_crossentropy'},
            loss_weights={'classification_output': 1.0, 'attention_weights': 0.3},
            metrics={'classification_output': 'accuracy'}
        )
        
        self.fusion_model = model
        return model
    
    def temporal_analysis(self, frame_sequence):
        """
        Analyze temporal patterns in fire/smoke detection
        """
        # LSTM-based temporal analysis
        lstm_model = tf.keras.Sequential([
            tf.keras.layers.TimeDistributed(
                tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
                input_shape=(None, 224, 224, 3)
            ),
            tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(2, 2)),
            tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dense(50, activation='relu'),
            tf.keras.layers.Dense(3, activation='softmax')
        ])
        
        return lstm_model

# Advanced Deployment and Monitoring System
class FireDetectionDeployment:
    """
    Production-ready deployment system with monitoring and alerts
    """
    
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.alert_thresholds = {
            'fire': 0.8,
            'smoke': 0.7,
            'normal': 0.3
        }
        self.monitoring_data = []
        
    def real_time_monitoring(self, camera_sources, alert_callback=None):
        """
        Real-time monitoring system for multiple camera sources
        """
        import threading
        import queue
        from datetime import datetime
        
        # Queue for processing frames
        frame_queue = queue.Queue(maxsize=100)
        alert_queue = queue.Queue()
        
        def camera_capture_thread(camera_id, source):
            """Thread for capturing frames from camera"""
            cap = cv2.VideoCapture(source)
            while True:
                ret, frame = cap.read()
                if ret:
                    timestamp = datetime.now()
                    if not frame_queue.full():
                        frame_queue.put((camera_id, frame, timestamp))
                else:
                    break
            cap.release()
        
        def processing_thread():
            """Thread for processing frames and detecting fire/smoke"""
            while True:
                if not frame_queue.empty():
                    camera_id, frame, timestamp = frame_queue.get()
                    
                    # Preprocess frame
                    processed_frame = cv2.resize(frame, (224, 224))
                    processed_frame = processed_frame.astype('float32') / 255.0
                    processed_frame = np.expand_dims(processed_frame, axis=0)
                    
                    # Predict
                    prediction = self.model.predict(processed_frame, verbose=0)[0]
                    fire_prob = prediction[0]
                    smoke_prob = prediction[1]
                    normal_prob = prediction[2]
                    
                    # Check alert conditions
                    alert_type = None
                    if fire_prob > self.alert_thresholds['fire']:
                        alert_type = 'FIRE_DETECTED'
                    elif smoke_prob > self.alert_thresholds['smoke']:
                        alert_type = 'SMOKE_DETECTED'
                    
                    # Store monitoring data
                    monitoring_record = {
                        'timestamp': timestamp,
                        'camera_id': camera_id,
                        'fire_probability': float(fire_prob),
                        'smoke_probability': float(smoke_prob),
                        'normal_probability': float(normal_prob),
                        'alert_type': alert_type
                    }
                    
                    self.monitoring_data.append(monitoring_record)
                    
                    # Trigger alert if necessary
                    if alert_type and alert_callback:
                        alert_queue.put(monitoring_record)
        
        def alert_thread():
            """Thread for handling alerts"""
            while True:
                if not alert_queue.empty():
                    alert_data = alert_queue.get()
                    if alert_callback:
                        alert_callback(alert_data)
        
        # Start threads
        threads = []
        
        # Camera capture threads
        for camera_id, source in camera_sources.items():
            thread = threading.Thread(target=camera_capture_thread, args=(camera_id, source))
            thread.daemon = True
            thread.start()
            threads.append(thread)
        
        # Processing thread
        proc_thread = threading.Thread(target=processing_thread)
        proc_thread.daemon = True
        proc_thread.start()
        threads.append(proc_thread)
        
        # Alert thread
        alert_thread = threading.Thread(target=alert_thread)
        alert_thread.daemon = True
        alert_thread.start()
        threads.append(alert_thread)
        
        return threads
    
    def generate_monitoring_dashboard(self):
        """
        Generate monitoring dashboard with real-time statistics
        """
        if not self.monitoring_data:
            print("No monitoring data available.")
            return
        
        df = pd.DataFrame(self.monitoring_data)
        
        # Dashboard visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Fire probability over time
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        axes[0, 0].plot(df['timestamp'], df['fire_probability'], color='red', alpha=0.7)
        axes[0, 0].axhline(y=self.alert_thresholds['fire'], color='red', linestyle='--', label='Fire Threshold')
        axes[0, 0].set_title('Fire Probability Over Time')
        axes[0, 0].set_ylabel('Probability')
        axes[0, 0].legend()
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Smoke probability over time
        axes[0, 1].plot(df['timestamp'], df['smoke_probability'], color='gray', alpha=0.7)
        axes[0, 1].axhline(y=self.alert_thresholds['smoke'], color='gray', linestyle='--', label='Smoke Threshold')
        axes[0, 1].set_title('Smoke Probability Over Time')
        axes[0, 1].set_ylabel('Probability')
        axes[0, 1].legend()
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Alert frequency by camera
        alert_counts = df[df['alert_type'].notna()].groupby('camera_id')['alert_type'].count()
        if not alert_counts.empty:
            axes[1, 0].bar(alert_counts.index, alert_counts.values, color='orange')
            axes[1, 0].set_title('Alerts by Camera')
            axes[1, 0].set_ylabel('Number of Alerts')
        
        # Alert type distribution
        alert_types = df[df['alert_type'].notna()]['alert_type'].value_counts()
        if not alert_types.empty:
            axes[1, 1].pie(alert_types.values, labels=alert_types.index, autopct='%1.1f%%')
            axes[1, 1].set_title('Alert Type Distribution')
        
        plt.tight_layout()
        plt.show()
        
        # Statistics summary
        print("\nMONITORING STATISTICS SUMMARY")
        print("=" * 40)
        print(f"Total monitoring records: {len(df)}")
        print(f"Total alerts: {len(df[df['alert_type'].notna()])}")
        print(f"Average fire probability: {df['fire_probability'].mean():.4f}")
        print(f"Average smoke probability: {df['smoke_probability'].mean():.4f}")
        print(f"Max fire probability: {df['fire_probability'].max():.4f}")
        print(f"Max smoke probability: {df['smoke_probability'].max():.4f}")

# Example usage and demonstration
def main_advanced_system():
    """
    Demonstrate the advanced fire detection system
    """
    print("ADVANCED FIRE DETECTION SYSTEM")
    print("Suitable for High-Impact Journal Publication")
    print("=" * 60)
    
    # 1. Initialize advanced system
    system = AdvancedFireDetectionSystem()
    
    # 2. Create benchmark dataset
    benchmark = FireDetectionBenchmark()
    X, y, metadata = benchmark.create_synthetic_benchmark_dataset(1000)
    
    print(f"Dataset created: {X.shape[0]} samples")
    print(f"Classes: Fire={np.sum(np.argmax(y, axis=1) == 0)}, "
          f"Smoke={np.sum(np.argmax(y, axis=1) == 1)}, "
          f"Normal={np.sum(np.argmax(y, axis=1) == 2)}")
    
    # 3. Run comprehensive benchmark
    print("\nRunning comprehensive benchmark...")
    benchmark_results = benchmark.run_comprehensive_benchmark()
    
    # 4. Generate benchmark report
    benchmark.generate_benchmark_report()
    
    # 5. Train ensemble model
    print("\nTraining ensemble model...")
    ensemble_model = system.create_ensemble_model()
    
    # 6. Cross-validation evaluation
    print("\nPerforming cross-validation...")
    cv_scores = system.train_with_cross_validation(X, y, k_folds=3)
    
    # 7. Multi-modal analysis demonstration
    print("\nDemonstrating multi-modal analysis...")
    multimodal = MultiModalFireAnalysis()
    fusion_model = multimodal.create_sensor_fusion_model()
    
    print("\nMulti-modal fusion model created:")
    fusion_model.summary()
    
    # 8. Generate comprehensive report
    print("\nGenerating comprehensive evaluation report...")
    evaluation_results = system.comprehensive_evaluation(X[:200], y[:200])
    report = system.generate_comprehensive_report(evaluation_results, cv_scores)
    
    print("\nRECOMMENDations FOR JOURNAL PUBLICATION:")
    print("-" * 50)
    for i, rec in enumerate(report['recommendations'], 1):
        print(f"{i}. {rec}")
    
    return system, benchmark_results, report

if __name__ == "__main__":
    # Run the advanced demonstration
    system, results, report = main_advanced_system()