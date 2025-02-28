import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, auc, roc_curve, confusion_matrix
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, Lambda, Layer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow.keras.backend as K
import os
import joblib
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetectionSystem:
    """
    Advanced Anomaly Detection & Pattern Recognition System
    
    A comprehensive framework for detecting subtle anomalies in high-dimensional, 
    noisy data using deep learning techniques.
    """
    
    def __init__(self, model_type='vae', random_state=42):
        """
        Initialize the anomaly detection system.
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ('vae', 'autoencoder', 'robust_autoencoder')
        random_state : int
            Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.threshold = None
        self.feature_importance = None
        self.history = None
        self.n_features = None
        self.is_fitted = False
        self.normal_reconstruction_errors = None
        
        # Set random seeds for reproducibility
        np.random.seed(self.random_state)
        tf.random.set_seed(self.random_state)
    
    def load_data(self, filepath, target_col=None, normalize=True):
        """
        Load and preprocess data from a CSV file or pandas DataFrame.
        
        Parameters:
        -----------
        filepath : str or pandas.DataFrame
            Path to the CSV file or pandas DataFrame
        target_col : str, optional
            Name of the target column (anomaly labels)
        normalize : bool
            Whether to normalize the features
            
        Returns:
        --------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray or None
            Target vector (if target_col is provided)
        """
        # Load data
        if isinstance(filepath, pd.DataFrame):
            df = filepath.copy()
        else:
            df = pd.read_csv(filepath)
        
        # Handle missing values
        print(f"Missing values before imputation: {df.isnull().sum().sum()}")
        df = df.fillna(df.median())
        print(f"Missing values after imputation: {df.isnull().sum().sum()}")
        
        # Extract features and target
        if target_col is not None and target_col in df.columns:
            X = df.drop(columns=[target_col])
            y = df[target_col].values
        else:
            X = df
            y = None
        
        # Convert categorical features to numerical
        X = pd.get_dummies(X)
        self.feature_names = X.columns.tolist()
        self.n_features = X.shape[1]
        
        # Normalize features
        if normalize:
            if self.scaler is None:
                self.scaler = StandardScaler()
                X_scaled = self.scaler.fit_transform(X)
            else:
                X_scaled = self.scaler.transform(X)
        else:
            X_scaled = X.values
        
        print(f"Data loaded: {X_scaled.shape[0]} samples, {X_scaled.shape[1]} features")
        return X_scaled, y
    
    def build_model(self, input_dim, encoding_dim=32, latent_dim=16, dropout_rate=0.2):
        """
        Build the anomaly detection model based on the specified model_type.
        
        Parameters:
        -----------
        input_dim : int
            Dimensionality of the input data
        encoding_dim : int
            Dimensionality of the encoder's hidden layer
        latent_dim : int
            Dimensionality of the latent space
        dropout_rate : float
            Dropout rate for regularization
            
        Returns:
        --------
        model : tensorflow.keras.Model
            Compiled Keras model
        """
        if self.model_type == 'vae':
            return self._build_variational_autoencoder(input_dim, encoding_dim, latent_dim, dropout_rate)
        elif self.model_type == 'autoencoder':
            return self._build_deep_autoencoder(input_dim, encoding_dim, dropout_rate)
        elif self.model_type == 'robust_autoencoder':
            return self._build_robust_autoencoder(input_dim, encoding_dim, latent_dim, dropout_rate)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _build_deep_autoencoder(self, input_dim, encoding_dim, dropout_rate):
        """
        Build a deep autoencoder model for anomaly detection.
        """
        # Encoder
        input_layer = Input(shape=(input_dim,))
        encoder = Dense(encoding_dim * 2, activation='relu')(input_layer)
        encoder = Dropout(dropout_rate)(encoder)
        encoder = Dense(encoding_dim, activation='relu')(encoder)
        
        # Decoder
        decoder = Dense(encoding_dim * 2, activation='relu')(encoder)
        decoder = Dropout(dropout_rate)(decoder)
        decoder = Dense(input_dim, activation='sigmoid')(decoder)
        
        # Autoencoder model
        autoencoder = Model(inputs=input_layer, outputs=decoder)
        encoder_model = Model(inputs=input_layer, outputs=encoder)
        
        # Compile the model
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                           loss='mse')
        
        return {
            'model': autoencoder,
            'encoder': encoder_model
        }
    
    def _build_variational_autoencoder(self, input_dim, encoding_dim, latent_dim, dropout_rate):
        """
        Build a variational autoencoder (VAE) model for anomaly detection.
        """
        # Encoder
        inputs = Input(shape=(input_dim,))
        x = Dense(encoding_dim * 2, activation='relu')(inputs)
        x = Dropout(dropout_rate)(x)
        x = Dense(encoding_dim, activation='relu')(x)
        
        # VAE latent space parameters
        z_mean = Dense(latent_dim)(x)
        z_log_var = Dense(latent_dim)(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = K.shape(z_mean)[0]
            dim = K.int_shape(z_mean)[1]
            epsilon = K.random_normal(shape=(batch, dim))
            return z_mean + K.exp(0.5 * z_log_var) * epsilon
        
        # Sample from latent space
        z = Lambda(sampling)([z_mean, z_log_var])
        
        # Decoder
        decoder_hidden = Dense(encoding_dim, activation='relu')
        decoder_hidden2 = Dense(encoding_dim * 2, activation='relu')
        decoder_output = Dense(input_dim, activation='sigmoid')
        
        x = decoder_hidden(z)
        x = Dropout(dropout_rate)(x)
        x = decoder_hidden2(x)
        outputs = decoder_output(x)
        
        # VAE model
        vae = Model(inputs, outputs)
        
        # VAE loss function
        def vae_loss(inputs, outputs):
            reconstruction_loss = K.sum(K.binary_crossentropy(inputs, outputs), axis=1)
            kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=1)
            return K.mean(reconstruction_loss + kl_loss)
        
        vae.compile(optimizer=Adam(learning_rate=0.001), loss=vae_loss)
        
        # Encoder model for extracting latent representations
        encoder = Model(inputs, z_mean)
        
        return {
            'model': vae,
            'encoder': encoder,
            'z_mean': z_mean,
            'z_log_var': z_log_var
        }
    
    def _build_robust_autoencoder(self, input_dim, encoding_dim, latent_dim, dropout_rate):
        """
        Build a robust autoencoder with noise injection for better anomaly detection.
        """
        # Define custom noise layer
        class GaussianNoiseLayer(Layer):
            def __init__(self, stddev=0.1, **kwargs):
                self.stddev = stddev
                super(GaussianNoiseLayer, self).__init__(**kwargs)
                
            def call(self, x, training=None):
                if training:
                    noise = K.random_normal(shape=K.shape(x), mean=0.0, stddev=self.stddev)
                    return x + noise
                return x
                
            def get_config(self):
                config = {'stddev': self.stddev}
                base_config = super(GaussianNoiseLayer, self).get_config()
                return dict(list(base_config.items()) + list(config.items()))
        
        # Encoder with noise
        inputs = Input(shape=(input_dim,))
        noise_layer = GaussianNoiseLayer(stddev=0.1)(inputs)
        
        x = Dense(encoding_dim * 2, activation='elu')(noise_layer)
        x = Dropout(dropout_rate)(x)
        x = Dense(encoding_dim, activation='elu')(x)
        encoded = Dense(latent_dim, activation='elu')(x)
        
        # Decoder
        x = Dense(encoding_dim, activation='elu')(encoded)
        x = Dropout(dropout_rate)(x)
        x = Dense(encoding_dim * 2, activation='elu')(x)
        outputs = Dense(input_dim, activation='sigmoid')(x)
        
        # Autoencoder model
        autoencoder = Model(inputs, outputs)
        encoder_model = Model(inputs, encoded)
        
        # Use Huber loss for robustness against outliers
        autoencoder.compile(optimizer=Adam(learning_rate=0.001), 
                           loss=tf.keras.losses.Huber(delta=1.0))
        
        return {
            'model': autoencoder,
            'encoder': encoder_model
        }
    
    def fit(self, X, y=None, validation_split=0.2, epochs=100, batch_size=64, patience=10):
        """
        Fit the anomaly detection model to the data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Training data
        y : numpy.ndarray, optional
            Labels for supervised fine-tuning (if available)
        validation_split : float
            Fraction of data to use for validation
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
        patience : int
            Number of epochs with no improvement after which training will be stopped
            
        Returns:
        --------
        self : object
            Returns self
        """
        if y is not None and np.sum(y) > 0:
            # If labels are available, use only normal samples for training
            X_train = X[y == 0]
            print(f"Training on {X_train.shape[0]} normal samples")
        else:
            X_train = X
            print(f"Training on all {X_train.shape[0]} samples (unsupervised)")
        
        # Split into train and validation sets
        if validation_split > 0:
            X_train, X_val = train_test_split(X_train, test_size=validation_split, 
                                              random_state=self.random_state)
        else:
            X_val = X_train
        
        # Build the model if not already built
        if self.model is None:
            input_dim = X_train.shape[1]
            self.n_features = input_dim
            model_dict = self.build_model(input_dim)
            self.model = model_dict['model']
            self.encoder = model_dict['encoder']
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        
        # Model checkpoint
        checkpoint = ModelCheckpoint(
            'best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=0
        )
        
        # Train the model
        print(f"Training {self.model_type} model...")
        self.history = self.model.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, X_val),
            callbacks=[early_stopping, checkpoint],
            verbose=1
        )
        
        # Compute reconstruction errors on normal data to set threshold
        if y is not None:
            normal_data = X[y == 0]
        else:
            normal_data = X
        
        reconstructions = self.model.predict(normal_data)
        mse = np.mean(np.square(normal_data - reconstructions), axis=1)
        
        # Store reconstruction errors of normal data for future analysis
        self.normal_reconstruction_errors = mse
        
        # Set threshold as a percentile of reconstruction errors
        self.threshold = np.percentile(mse, 95)  # 95th percentile as default
        
        print(f"Model trained. Anomaly threshold set to: {self.threshold:.6f}")
        self.is_fitted = True
        
        # Compute feature importance
        self._compute_feature_importance(X)
        
        return self
    
    def _compute_feature_importance(self, X):
        """
        Compute feature importance by measuring the impact of each feature
        on reconstruction error.
        """
        if not hasattr(self, 'feature_names') or len(self.feature_names) != X.shape[1]:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        baseline_error = self._get_reconstruction_errors(X)
        feature_importance = {}
        
        print("Computing feature importance...")
        for i in tqdm(range(X.shape[1])):
            # Create a perturbed version of X with feature i shuffled
            X_perturbed = X.copy()
            X_perturbed[:, i] = np.random.permutation(X_perturbed[:, i])
            
            # Compute reconstruction error with the perturbed feature
            perturbed_error = self._get_reconstruction_errors(X_perturbed)
            
            # Feature importance is the increase in reconstruction error
            importance = np.mean(perturbed_error - baseline_error)
            feature_importance[self.feature_names[i]] = importance
        
        # Normalize feature importance
        total = sum(max(0, v) for v in feature_importance.values())
        if total > 0:
            self.feature_importance = {k: max(0, v) / total for k, v in feature_importance.items()}
        else:
            self.feature_importance = feature_importance
            
        return self.feature_importance
    
    def _get_reconstruction_errors(self, X):
        """
        Compute reconstruction errors for input data.
        """
        reconstructions = self.model.predict(X)
        return np.mean(np.square(X - reconstructions), axis=1)
    
    def predict(self, X, return_scores=False):
        """
        Predict anomalies in new data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to predict anomalies for
        return_scores : bool
            Whether to return anomaly scores along with predictions
            
        Returns:
        --------
        predictions : numpy.ndarray
            Binary anomaly predictions (1 for anomaly, 0 for normal)
        scores : numpy.ndarray, optional
            Anomaly scores if return_scores is True
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Preprocess data if needed
        if self.scaler is not None and X.shape[1] == self.n_features:
            X = self.scaler.transform(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Compute mean squared error for each sample
        mse = np.mean(np.square(X - reconstructions), axis=1)
        
        # Classify as anomaly if mse > threshold
        predictions = (mse > self.threshold).astype(int)
        
        if return_scores:
            return predictions, mse
        else:
            return predictions
    
    def score_samples(self, X):
        """
        Compute anomaly scores for samples in X.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to compute anomaly scores for
            
        Returns:
        --------
        scores : numpy.ndarray
            Anomaly scores (higher means more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Preprocess data if needed
        if self.scaler is not None and X.shape[1] == self.n_features:
            X = self.scaler.transform(X)
        
        # Get reconstructions
        reconstructions = self.model.predict(X)
        
        # Compute mean squared error for each sample
        mse = np.mean(np.square(X - reconstructions), axis=1)
        
        return mse
    
    def set_threshold(self, threshold):
        """
        Set a custom threshold for anomaly detection.
        
        Parameters:
        -----------
        threshold : float
            New threshold value
        """
        self.threshold = threshold
        print(f"Anomaly threshold updated to: {self.threshold:.6f}")
        
    def auto_tune_threshold(self, X, y, target_precision=0.95):
        """
        Automatically tune the anomaly threshold based on a validation set.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Validation data
        y : numpy.ndarray
            True anomaly labels (1 for anomaly, 0 for normal)
        target_precision : float
            Target precision level
            
        Returns:
        --------
        threshold : float
            Optimal threshold
        """
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Calculate precision and recall for different thresholds
        precision, recall, thresholds = precision_recall_curve(y, scores)
        
        # Find the threshold that achieves the target precision
        idx = np.argmin(np.abs(precision - target_precision))
        optimal_threshold = thresholds[idx]
        
        self.threshold = optimal_threshold
        print(f"Threshold tuned to {optimal_threshold:.6f} (precision: {precision[idx]:.4f}, recall: {recall[idx]:.4f})")
        
        return optimal_threshold
    
    def get_feature_importance(self, top_n=10):
        """
        Get the most important features for anomaly detection.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        feature_importance : dict
            Dictionary of feature importance scores
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance is not computed. Train the model first.")
        
        # Sort features by importance
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        
        # Return top N features
        return dict(sorted_features[:top_n])
    
    def explain_anomaly(self, x, top_n=5):
        """
        Explain why a sample was flagged as an anomaly.
        
        Parameters:
        -----------
        x : numpy.ndarray
            Sample to explain (must be a single sample)
        top_n : int
            Number of top contributing features to return
            
        Returns:
        --------
        explanation : dict
            Dictionary with feature contributions
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Reshape to ensure we have a 2D array with a single sample
        if len(x.shape) == 1:
            x = x.reshape(1, -1)
        
        # Ensure we're working with processed data
        if self.scaler is not None:
            x_processed = self.scaler.transform(x)
        else:
            x_processed = x
        
        # Get reconstruction
        reconstruction = self.model.predict(x_processed)
        
        # Compute squared error for each feature
        squared_errors = np.square(x_processed - reconstruction)
        
        # Create a dictionary of feature contributions
        contributions = {}
        for i, error in enumerate(squared_errors[0]):
            if i < len(self.feature_names):
                feature_name = self.feature_names[i]
                contributions[feature_name] = float(error)
        
        # Sort by contribution
        sorted_contributions = sorted(contributions.items(), key=lambda x: x[1], reverse=True)
        
        # Get the reconstruction error
        total_error = np.sum(squared_errors)
        
        # Prepare the explanation
        anomaly_score = float(total_error)
        is_anomaly = anomaly_score > self.threshold
        top_features = dict(sorted_contributions[:top_n])
        
        # Context-aware analysis
        context = self._provide_context(x_processed, anomaly_score, top_features)
        
        explanation = {
            "is_anomaly": bool(is_anomaly),
            "anomaly_score": anomaly_score,
            "threshold": self.threshold,
            "top_contributing_features": top_features,
            "context": context
        }
        
        return explanation
    
    def _provide_context(self, x, anomaly_score, top_features):
        """
        Provide contextual information for an anomaly.
        
        This is a placeholder for domain-specific context. In a real-world scenario,
        this would be customized based on the specific domain and data.
        """
        # Calculate the percentile of this anomaly score compared to normal data
        if self.normal_reconstruction_errors is not None:
            percentile = 100 * (np.sum(self.normal_reconstruction_errors < anomaly_score) / 
                               len(self.normal_reconstruction_errors))
            
            if percentile > 99.9:
                severity = "Critical"
            elif percentile > 99:
                severity = "High"
            elif percentile > 95:
                severity = "Medium"
            else:
                severity = "Low"
        else:
            severity = "Unknown"
            percentile = None
        
        # Get the distance from the decision boundary
        boundary_distance = anomaly_score / self.threshold - 1
        
        context = {
            "severity": severity,
            "percentile": percentile,
            "boundary_distance": boundary_distance
        }
        
        return context
    
    def plot_anomaly_scores(self, X, y=None, figsize=(12, 6)):
        """
        Plot the distribution of anomaly scores.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Data to compute anomaly scores for
        y : numpy.ndarray, optional
            True anomaly labels for colored visualization
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        scores = self.score_samples(X)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        if y is not None:
            # Plot scores for normal and anomalous points separately
            normal_scores = scores[y == 0]
            anomaly_scores = scores[y == 1]
            
            sns.histplot(normal_scores, color='blue', alpha=0.5, bins=50, 
                        label='Normal', ax=ax)
            sns.histplot(anomaly_scores, color='red', alpha=0.5, bins=50, 
                        label='Anomaly', ax=ax)
        else:
            # Plot all scores together
            sns.histplot(scores, bins=50, ax=ax)
        
        # Add threshold line
        plt.axvline(x=self.threshold, color='red', linestyle='--', 
                   label=f'Threshold: {self.threshold:.4f}')
        
        plt.title('Distribution of Anomaly Scores')
        plt.xlabel('Anomaly Score')
        plt.ylabel('Count')
        plt.legend()
        
        return fig
    
    def plot_feature_importance(self, top_n=10, figsize=(10, 8)):
        """
        Plot feature importance for anomaly detection.
        
        Parameters:
        -----------
        top_n : int
            Number of top features to display
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        if self.feature_importance is None:
            raise ValueError("Feature importance is not computed. Train the model first.")
        
        # Sort features by importance and get top N
        sorted_features = sorted(self.feature_importance.items(), key=lambda x: x[1], reverse=True)
        top_features = dict(sorted_features[:top_n])
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create a horizontal bar plot
        features = list(top_features.keys())
        importance = list(top_features.values())
        
        # Sort by importance (ascending for better visualization)
        indices = np.argsort(importance)
        features = [features[i] for i in indices]
        importance = [importance[i] for i in indices]
        
        # Plot
        ax.barh(features, importance, color='skyblue')
        
        plt.title('Feature Importance for Anomaly Detection')
        plt.xlabel('Importance Score')
        
        # Format y-tick labels to handle long feature names
        max_length = 30
        plt.yticks(range(len(features)), 
                  [f[:max_length] + '...' if len(f) > max_length else f for f in features])
        
        plt.tight_layout()
        
        return fig
    
    def plot_training_history(self, figsize=(10, 6)):
        """
        Plot the training history of the model.
        
        Parameters:
        -----------
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        if self.history is None:
            raise ValueError("No training history available. Train the model first.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot training & validation loss
        ax.plot(self.history.history['loss'], label='Training Loss')
        
        if 'val_loss' in self.history.history:
            ax.plot(self.history.history['val_loss'], label='Validation Loss')
        
        plt.title('Model Training History')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        return fig
    
    def evaluate(self, X, y):
        """
        Evaluate the anomaly detection model on labeled data.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test data
        y : numpy.ndarray
            True anomaly labels
            
        Returns:
        --------
        metrics : dict
            Dictionary of evaluation metrics
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
        
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Make predictions using the current threshold
        predictions = (scores > self.threshold).astype(int)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        
        # Calculate precision-recall curve
        precision, recall, _ = precision_recall_curve(y, scores)
        pr_auc = auc(recall, precision)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = confusion_matrix(y, predictions).ravel()
        
        # Calculate metrics
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision_score = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall_score = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision_score * recall_score / (precision_score + recall_score) if (precision_score + recall_score) > 0 else 0
        
        # Compile metrics
        metrics = {
            'accuracy': accuracy,
            'precision': precision_score,
            'recall': recall_score,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'confusion_matrix': {
                'true_negative': tn,
                'false_positive': fp,
                'false_negative': fn,
                'true_positive': tp
            }
        }
        
        return metrics
    
    def plot_roc_curve(self, X, y, figsize=(8, 8)):
        """
        Plot the ROC curve for the anomaly detection model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test data
        y : numpy.ndarray
            True anomaly labels
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure.Figure
            Figure object
        """
        # Get anomaly scores
        scores = self.score_samples(X)
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, scores)
        roc_auc = auc(fpr, tpr)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot ROC curve
        ax.plot(fpr, tpr, color='blue', lw=2, 
               label=f'ROC curve (AUC = {roc_auc:.4f})')
        
        # Plot diagonal line (random classifier)
        ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        
        return fig
    
    def plot_precision_recall_curve(self, X, y, figsize=(8, 8)):
        """
        Plot the precision-recall curve for the anomaly detection model.
        
        Parameters:
        -----------
        X : numpy.ndarray
            Test data
        y : numpy.ndarray
            True anomaly labels
        figsize : tuple
            Figure size
            
        Returns:
        --------
        fig : matplotlib.figure
