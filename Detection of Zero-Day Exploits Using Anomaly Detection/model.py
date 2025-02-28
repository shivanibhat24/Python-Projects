import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import joblib

class ZeroDayExploitDetector:
    def __init__(self, latent_dim=10, threshold_percentile=99):
        """
        Initialize the Zero-Day Exploit Detector.
        
        Args:
            latent_dim: Dimension of the latent space for the VAE.
            threshold_percentile: Percentile for setting anomaly threshold.
        """
        self.latent_dim = latent_dim
        self.threshold_percentile = threshold_percentile
        self.vae = None
        self.encoder = None
        self.decoder = None
        self.preprocessor = None
        self.clusterer = None
        self.threshold = None
        self.cluster_centers = None
        self.cluster_distances_threshold = None
    
    def _build_vae(self, input_dim, encoder_layers=[128, 64], decoder_layers=[64, 128]):
        """
        Build the Variational Autoencoder model.
        
        Args:
            input_dim: Input dimension of the data.
            encoder_layers: List of hidden units for encoder layers.
            decoder_layers: List of hidden units for decoder layers.
        """
        # Encoder
        encoder_inputs = keras.Input(shape=(input_dim,))
        x = encoder_inputs
        for units in encoder_layers:
            x = layers.Dense(units, activation='relu')(x)
        
        # Mean and variance for latent space
        z_mean = layers.Dense(self.latent_dim, name='z_mean')(x)
        z_log_var = layers.Dense(self.latent_dim, name='z_log_var')(x)
        
        # Sampling function
        def sampling(args):
            z_mean, z_log_var = args
            batch = tf.shape(z_mean)[0]
            dim = tf.shape(z_mean)[1]
            epsilon = tf.random.normal(shape=(batch, dim))
            return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
        z = layers.Lambda(sampling, output_shape=(self.latent_dim,), name='z')([z_mean, z_log_var])
        
        # Instantiate encoder model
        self.encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')
        
        # Decoder
        latent_inputs = keras.Input(shape=(self.latent_dim,), name='z_sampling')
        x = latent_inputs
        for units in decoder_layers:
            x = layers.Dense(units, activation='relu')(x)
        decoder_outputs = layers.Dense(input_dim)(x)
        
        # Instantiate decoder model
        self.decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')
        
        # Instantiate VAE model
        outputs = self.decoder(self.encoder(encoder_inputs)[2])
        self.vae = keras.Model(encoder_inputs, outputs, name='vae')
        
        # VAE loss
        reconstruction_loss = keras.losses.mse(encoder_inputs, outputs)
        reconstruction_loss *= input_dim
        kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        vae_loss = reconstruction_loss + kl_loss
        
        self.vae.add_loss(vae_loss)
        self.vae.compile(optimizer='adam')
        
        return self.vae
    
    def preprocess_data(self, data, categorical_cols=None, numerical_cols=None, fit=True):
        """
        Preprocess data with standardization for numerical features and one-hot encoding for categorical features.
        
        Args:
            data: Input data as DataFrame.
            categorical_cols: List of categorical column names.
            numerical_cols: List of numerical column names.
            fit: Whether to fit the preprocessor or just transform.
            
        Returns:
            Preprocessed data as numpy array.
        """
        if categorical_cols is None:
            categorical_cols = []
        
        if numerical_cols is None:
            numerical_cols = [col for col in data.columns if col not in categorical_cols]
        
        # Create preprocessing pipeline
        if fit:
            numeric_transformer = StandardScaler()
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')
            
            self.preprocessor = ColumnTransformer(
                transformers=[
                    ('num', numeric_transformer, numerical_cols),
                    ('cat', categorical_transformer, categorical_cols)
                ]
            )
            
            processed_data = self.preprocessor.fit_transform(data)
        else:
            processed_data = self.preprocessor.transform(data)
        
        return processed_data
    
    def train(self, data, categorical_cols=None, numerical_cols=None, 
              epochs=50, batch_size=128, validation_split=0.2):
        """
        Train the VAE model on the given data.
        
        Args:
            data: Input data as DataFrame.
            categorical_cols: List of categorical column names.
            numerical_cols: List of numerical column names.
            epochs: Number of training epochs.
            batch_size: Batch size for training.
            validation_split: Fraction of data to use as validation.
            
        Returns:
            Training history.
        """
        # Preprocess data
        print("Preprocessing data...")
        processed_data = self.preprocess_data(data, categorical_cols, numerical_cols)
        
        print(f"Processed data shape: {processed_data.shape}")
        
        # Build VAE
        print("Building VAE model...")
        input_dim = processed_data.shape[1]
        self._build_vae(input_dim)
        
        # Train VAE
        print("Training VAE model...")
        history = self.vae.fit(
            processed_data, processed_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            verbose=1
        )
        
        # Compute reconstruction errors for setting threshold
        print("Computing reconstruction errors and setting threshold...")
        reconstructions = self.vae.predict(processed_data)
        mse = np.mean(np.power(processed_data - reconstructions, 2), axis=1)
        self.threshold = np.percentile(mse, self.threshold_percentile)
        
        # Encode data to latent space for clustering
        print("Performing clustering on latent space...")
        _, _, encoded_data = self.encoder.predict(processed_data)
        
        # Determine optimal number of clusters using silhouette score
        best_score = -1
        best_k = 2
        
        for k in range(2, min(10, len(encoded_data) // 10)):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(encoded_data)
            score = silhouette_score(encoded_data, cluster_labels)
            if score > best_score:
                best_score = score
                best_k = k
        
        # Fit KMeans with best number of clusters
        self.clusterer = KMeans(n_clusters=best_k, random_state=42)
        self.clusterer.fit(encoded_data)
        
        # Store cluster centers
        self.cluster_centers = self.clusterer.cluster_centers_
        
        # Compute distances to nearest cluster center
        cluster_labels = self.clusterer.predict(encoded_data)
        distances = []
        
        for i, point in enumerate(encoded_data):
            center = self.cluster_centers[cluster_labels[i]]
            distance = np.linalg.norm(point - center)
            distances.append(distance)
        
        # Set cluster distance threshold
        self.cluster_distances_threshold = np.percentile(distances, self.threshold_percentile)
        
        print(f"Training complete. Anomaly threshold: {self.threshold:.4f}")
        print(f"Cluster distance threshold: {self.cluster_distances_threshold:.4f}")
        
        return history
    
    def detect_anomalies(self, data, return_scores=False):
        """
        Detect anomalies in new data.
        
        Args:
            data: Input data as DataFrame.
            return_scores: Whether to return anomaly scores.
            
        Returns:
            Binary array of anomaly flags or anomaly scores if return_scores=True.
        """
        # Preprocess data
        processed_data = self.preprocess_data(data, fit=False)
        
        # Compute reconstruction error
        reconstructions = self.vae.predict(processed_data)
        mse = np.mean(np.power(processed_data - reconstructions, 2), axis=1)
        
        # Get latent representations
        _, _, encoded_data = self.encoder.predict(processed_data)
        
        # Predict clusters and compute distances
        cluster_labels = self.clusterer.predict(encoded_data)
        distances = []
        
        for i, point in enumerate(encoded_data):
            center = self.cluster_centers[cluster_labels[i]]
            distance = np.linalg.norm(point - center)
            distances.append(distance)
        
        # Combine both metrics for anomaly detection
        reconstruction_anomalies = mse > self.threshold
        cluster_anomalies = np.array(distances) > self.cluster_distances_threshold
        
        # An instance is an anomaly if either metric flags it
        anomalies = np.logical_or(reconstruction_anomalies, cluster_anomalies)
        
        if return_scores:
            # Normalize scores between 0 and 1
            rec_scores = mse / (np.max(mse) + 1e-10)
            dist_scores = np.array(distances) / (np.max(distances) + 1e-10)
            
            # Combined score as average
            combined_scores = (rec_scores + dist_scores) / 2
            return anomalies, combined_scores
        
        return anomalies
    
    def save_model(self, path):
        """Save the model components to files."""
        os.makedirs(path, exist_ok=True)
        self.vae.save(os.path.join(path, 'vae_model'))
        self.encoder.save(os.path.join(path, 'encoder_model'))
        self.decoder.save(os.path.join(path, 'decoder_model'))
        joblib.dump(self.preprocessor, os.path.join(path, 'preprocessor.pkl'))
        joblib.dump(self.clusterer, os.path.join(path, 'clusterer.pkl'))
        joblib.dump({
            'threshold': self.threshold,
            'cluster_distances_threshold': self.cluster_distances_threshold,
            'latent_dim': self.latent_dim,
            'threshold_percentile': self.threshold_percentile
        }, os.path.join(path, 'params.pkl'))
    
    @classmethod
    def load_model(cls, path):
        """Load a trained model from files."""
        params = joblib.load(os.path.join(path, 'params.pkl'))
        detector = cls(
            latent_dim=params['latent_dim'],
            threshold_percentile=params['threshold_percentile']
        )
        detector.vae = keras.models.load_model(os.path.join(path, 'vae_model'), compile=False)
        detector.encoder = keras.models.load_model(os.path.join(path, 'encoder_model'), compile=False)
        detector.decoder = keras.models.load_model(os.path.join(path, 'decoder_model'), compile=False)
        detector.preprocessor = joblib.load(os.path.join(path, 'preprocessor.pkl'))
        detector.clusterer = joblib.load(os.path.join(path, 'clusterer.pkl'))
        detector.threshold = params['threshold']
        detector.cluster_distances_threshold = params['cluster_distances_threshold']
        detector.cluster_centers = detector.clusterer.cluster_centers_
        return detector


class UNSWNB15DataLoader:
    """Utility class for loading and preprocessing the UNSW-NB15 dataset."""
    
    def __init__(self, data_path):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to the UNSW-NB15 dataset files.
        """
        self.data_path = data_path
    
    def load_data(self, sample_size=None):
        """
        Load the UNSW-NB15 dataset.
        
        Args:
            sample_size: Number of samples to load (for testing purposes).
            
        Returns:
            DataFrame containing the dataset.
        """
        # Column names based on UNSW-NB15 dataset description
        columns = [
            'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 
            'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 'service', 'sload', 'dload', 
            'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 
            'trans_depth', 'res_bdy_len', 'sjit', 'djit', 'stime', 'ltime', 'sintpkt', 
            'dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports', 'ct_state_ttl', 
            'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src', 'ct_srv_dst', 
            'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm', 
            'ct_dst_src_ltm', 'attack_cat', 'label'
        ]
        
        try:
            # Load UNSW-NB15 dataset from CSV
            print(f"Loading data from {self.data_path}...")
            
            # For simplicity, assuming the data is in a single CSV file
            # In reality, you might need to concatenate multiple files
            df = pd.read_csv(self.data_path, names=columns, low_memory=False)
            
            if sample_size:
                df = df.sample(n=min(sample_size, len(df)), random_state=42)
            
            print(f"Loaded {len(df)} records.")
            return df
            
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Using synthetic data for demonstration...")
            
            # Create synthetic data for demonstration purposes
            np.random.seed(42)
            n_samples = sample_size if sample_size else 10000
            
            # Create numerical features
            numerical_features = [
                'dur', 'sbytes', 'dbytes', 'sttl', 'dttl', 'sloss', 'dloss', 
                'sload', 'dload', 'spkts', 'dpkts', 'swin', 'dwin', 'stcpb', 
                'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 
                'sjit', 'djit', 'sintpkt', 'dintpkt', 'tcprtt', 'synack', 'ackdat',
                'ct_state_ttl', 'ct_flw_http_mthd', 'ct_ftp_cmd', 'ct_srv_src', 
                'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 
                'ct_dst_sport_ltm', 'ct_dst_src_ltm'
            ]
            
            # Create categorical features
            categorical_features = [
                'proto', 'state', 'service', 'is_sm_ips_ports', 'is_ftp_login'
            ]
            
            # Generate synthetic data
            synthetic_data = {}
            
            # Add numerical features with random values
            for feature in numerical_features:
                synthetic_data[feature] = np.random.exponential(scale=10, size=n_samples)
            
            # Add categorical features
            synthetic_data['proto'] = np.random.choice(['tcp', 'udp', 'icmp'], size=n_samples)
            synthetic_data['state'] = np.random.choice(['FIN', 'CON', 'REQ', 'INT'], size=n_samples)
            synthetic_data['service'] = np.random.choice(['http', 'ftp', 'smtp', 'dns', '-'], size=n_samples)
            synthetic_data['is_sm_ips_ports'] = np.random.choice([0, 1], size=n_samples)
            synthetic_data['is_ftp_login'] = np.random.choice([0, 1], size=n_samples)
            
            # Add labels (mostly normal with a few attacks)
            attack_ratio = 0.1  # 10% attacks
            synthetic_data['label'] = np.random.choice([0, 1], size=n_samples, p=[1-attack_ratio, attack_ratio])
            synthetic_data['attack_cat'] = ['normal' if label == 0 else 
                                         np.random.choice(['DoS', 'Exploits', 'Reconnaissance', 'Fuzzers', 'Analysis', 'Backdoor', 'Shellcode', 'Worms', 'Generic']) 
                                         for label in synthetic_data['label']]
            
            df = pd.DataFrame(synthetic_data)
            print(f"Created synthetic dataset with {len(df)} records.")
            return df
    
    def preprocess(self, df, drop_attacks=False):
        """
        Preprocess the UNSW-NB15 dataset for anomaly detection.
        
        Args:
            df: DataFrame containing the UNSW-NB15 data.
            drop_attacks: Whether to drop known attacks for training.
            
        Returns:
            Preprocessed DataFrame.
        """
        # Handle missing values
        print("Preprocessing data...")
        df = df.copy()
        
        # Convert IP addresses to numerical features (simplified)
        if 'srcip' in df.columns and 'dstip' in df.columns:
            # For simplicity, we'll just drop them, but in practice you'd extract useful features
            df = df.drop(['srcip', 'dstip'], axis=1, errors='ignore')
        
        # Handle categorical features
        categorical_cols = ['proto', 'state', 'service', 'attack_cat']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Handle timestamps if present
        timestamp_cols = ['stime', 'ltime']
        for col in timestamp_cols:
            if col in df.columns:
                # Convert to timestamp features or drop
                df = df.drop(col, axis=1, errors='ignore')
        
        # Fill missing values
        for col in df.select_dtypes(include=['number']).columns:
            df[col] = df[col].fillna(df[col].median())
        
        for col in df.select_dtypes(include=['category']).columns:
            df[col] = df[col].fillna(df[col].mode()[0])
        
        # For training, keep only normal traffic if requested
        if drop_attacks and 'label' in df.columns:
            print("Keeping only normal traffic for training...")
            df = df[df['label'] == 0]
        
        # Drop the label and attack_cat columns for unsupervised learning
        if 'label' in df.columns:
            labels = df['label'].copy()
            df = df.drop(['label'], axis=1)
        else:
            labels = None
            
        if 'attack_cat' in df.columns:
            attack_cats = df['attack_cat'].copy()
            df = df.drop(['attack_cat'], axis=1)
        else:
            attack_cats = None
        
        print(f"Preprocessing complete. Shape: {df.shape}")
        return df, labels, attack_cats


def visualize_results(detector, data, labels, attack_cats=None):
    """
    Visualize the results of anomaly detection.
    
    Args:
        detector: Trained ZeroDayExploitDetector instance.
        data: DataFrame containing the test data.
        labels: Ground truth labels (0 for normal, 1 for attack).
        attack_cats: Attack categories (if available).
    """
    # Get anomaly predictions and scores
    anomalies, scores = detector.detect_anomalies(data, return_scores=True)
    
    # Encode data to latent space
    processed_data = detector.preprocess_data(data, fit=False)
    _, _, encoded_data = detector.encoder.predict(processed_data)
    
    # Create a results DataFrame
    results = pd.DataFrame({
        'Anomaly_Score': scores,
        'Is_Anomaly': anomalies,
        'True_Label': labels if labels is not None else np.zeros(len(data)),
        'Attack_Category': attack_cats if attack_cats is not None else ['Unknown'] * len(data),
        'Latent_1': encoded_data[:, 0],
        'Latent_2': encoded_data[:, 1]
    })
    
    # Plot anomaly score distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(data=results, x='Anomaly_Score', hue='True_Label', kde=True, bins=50)
    plt.axvline(x=np.percentile(results['Anomaly_Score'], detector.threshold_percentile), 
                color='red', linestyle='--', label='Threshold')
    plt.title('Anomaly Score Distribution')
    plt.legend(['Threshold', 'Normal', 'Attack'])
    plt.tight_layout()
    plt.savefig('anomaly_score_distribution.png')
    plt.close()
    
    # Plot latent space visualization
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red']
    for i, label in enumerate([0, 1]):
        mask = results['True_Label'] == label
        plt.scatter(
            results.loc[mask, 'Latent_1'], 
            results.loc[mask, 'Latent_2'],
            c=colors[i], 
            label='Normal' if label == 0 else 'Attack',
            alpha=0.6
        )
    
    # Plot cluster centers
    for i, center in enumerate(detector.cluster_centers):
        plt.scatter(
            center[0], 
            center[1], 
            c='black', 
            marker='X', 
            s=100, 
            label='Cluster Center' if i == 0 else ""
        )
    
    plt.title('Latent Space Visualization')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig('latent_space_visualization.png')
    plt.close()
    
    # Confusion matrix for anomaly detection
    if labels is not None:
        from sklearn.metrics import confusion_matrix, classification_report
        
        cm = confusion_matrix(labels, anomalies)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Normal', 'Anomaly'],
                    yticklabels=['Normal', 'Attack'])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(labels, anomalies, target_names=['Normal', 'Attack']))
        
        # Plot ROC curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig('roc_curve.png')
        plt.close()


def main():
    """Main function to demonstrate the Zero-Day Exploit Detection system."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Zero-Day Exploit Detection System')
    parser.add_argument('--data_path', type=str, default='UNSW-NB15.csv',
                        help='Path to the UNSW-NB15 dataset')
    parser.add_argument('--sample_size', type=int, default=None,
                        help='Number of samples to use (for testing purposes)')
    parser.add_argument('--latent_dim', type=int, default=10,
                        help='Dimension of the latent space for the VAE')
    parser.add_argument('--threshold_percentile', type=int, default=99,
                        help='Percentile for setting anomaly threshold (e.g., 99 means 1% anomalies)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='Batch size for training')
    parser.add_argument('--save_path', type=str, default='./models/zero_day_detector',
                        help='Path to save the trained model')
    parser.add_argument('--load_model', action='store_true',
                        help='Load a previously trained model')
    
    args = parser.parse_args()
    
    print("Zero-Day Exploit Detection System")
    print("=================================")
    
    # Load and preprocess the data
    data_loader = UNSWNB15DataLoader(args.data_path)
    df = data_loader.load_data(sample_size=args.sample_size)
    
    # Split data into training and testing sets
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # For training, use only normal traffic
    train_df, _, _ = data_loader.preprocess(train_df, drop_attacks=True)
    
    # For testing, keep all traffic including attacks
    test_df, test_labels, test_attack_cats = data_loader.preprocess(test_df, drop_attacks=False)
    
    # Initialize the detector
    if args.load_model and os.path.exists(args.save_path):
        print(f"Loading model from {args.save_path}...")
        detector = ZeroDayExploitDetector.load_model(args.save_path)
    else:
        print("Training new model...")
        detector = ZeroDayExploitDetector(
            latent_dim=args.latent_dim,
            threshold_percentile=args.threshold_percentile
        )
        
        # Train the model
        detector.train(
            train_df,
            epochs=args.epochs,
            batch_size=args.batch_size
        )
        
        # Save the model
        print(f"Saving model to {args.save_path}...")
        detector.save_model(args.save_path)
    
    # Detect anomalies on test data
    print("Detecting anomalies in test data...")
    anomalies = detector.detect_anomalies(test_df)
    
    # Calculate accuracy if true labels are available
    if test_labels is not None:
        accuracy = np.mean(anomalies == test_labels)
        print(f"Anomaly detection accuracy: {accuracy:.4f}")
    
    # Visualize results
    print("Visualizing results...")
    visualize_results(detector, test_df, test_labels, test_attack_cats)
    
    print("All done! Check the output directory for visualizations.")


if __name__ == "__main__":
    main()
