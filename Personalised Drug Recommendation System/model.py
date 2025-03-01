import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import os

class PersonalizedDrugResponsePredictor:
    def __init__(self, model_type="gan"):
        """
        Initialize the drug response prediction model
        
        Parameters:
        -----------
        model_type : str
            Type of model to use ("gan" or "rl")
        """
        self.model_type = model_type
        self.preprocessor = None
        self.model = None
        self.drug_features = None
        
    def download_gdsc_data(self, save_dir="data"):
        """
        Download and extract GDSC dataset
        Note: In a real implementation, you would need to register and follow proper data access protocols
        """
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        print("In a real implementation, this would download the GDSC data")
        print("For this demo, we'll generate synthetic data that mimics GDSC structure")
        
        # Generate synthetic data that mimics GDSC structure
        # 1. Drug response data
        n_samples = 1000
        n_drugs = 20
        n_cell_lines = 50
        
        # Drug response data
        drug_ids = [f"Drug_{i}" for i in range(n_drugs)]
        cell_line_ids = [f"CellLine_{i}" for i in range(n_cell_lines)]
        
        # Random drug responses (IC50 values)
        responses = []
        for _ in range(n_samples):
            drug_id = np.random.choice(drug_ids)
            cell_line_id = np.random.choice(cell_line_ids)
            ic50 = np.random.lognormal(mean=2, sigma=1)  # Log-normal distribution for IC50 values
            auc = np.random.uniform(0, 1)  # Area under curve
            responses.append({
                'drug_id': drug_id,
                'cell_line_id': cell_line_id,
                'ic50': ic50,
                'auc': auc
            })
        
        response_df = pd.DataFrame(responses)
        
        # 2. Drug features data
        drug_features = []
        for drug_id in drug_ids:
            # Generate random molecular descriptors
            descriptors = np.random.normal(size=10)
            drug_features.append({
                'drug_id': drug_id,
                **{f'descriptor_{i}': val for i, val in enumerate(descriptors)}
            })
        
        drug_df = pd.DataFrame(drug_features)
        
        # 3. Cell line (patient) genomic features
        genomic_features = []
        for cell_id in cell_line_ids:
            # Generate random genomic features (gene expression, mutations, etc.)
            gene_expr = np.random.normal(size=15)
            mutations = np.random.binomial(1, 0.1, size=5)
            
            genomic_features.append({
                'cell_line_id': cell_id,
                **{f'gene_expr_{i}': val for i, val in enumerate(gene_expr)},
                **{f'mutation_{i}': val for i, val in enumerate(mutations)}
            })
        
        genomic_df = pd.DataFrame(genomic_features)
        
        # Save synthetic datasets
        response_df.to_csv(f"{save_dir}/drug_responses.csv", index=False)
        drug_df.to_csv(f"{save_dir}/drug_features.csv", index=False)
        genomic_df.to_csv(f"{save_dir}/genomic_features.csv", index=False)
        
        return response_df, drug_df, genomic_df
    
    def load_and_preprocess_data(self, response_df, drug_df, genomic_df):
        """
        Load and preprocess the GDSC datasets for model training
        
        Parameters:
        -----------
        response_df : pandas.DataFrame
            Drug response data
        drug_df : pandas.DataFrame
            Drug molecular features
        genomic_df : pandas.DataFrame
            Patient/cell line genomic features
        
        Returns:
        --------
        X : numpy.ndarray
            Preprocessed feature matrix
        y : numpy.ndarray
            Target values (drug responses)
        """
        # Merge datasets
        merged_df = response_df.merge(drug_df, on='drug_id')
        merged_df = merged_df.merge(genomic_df, on='cell_line_id')
        
        # Drop IDs as they won't be used for prediction
        X_df = merged_df.drop(['drug_id', 'cell_line_id', 'ic50', 'auc'], axis=1)
        y_df = merged_df['ic50']  # Predict IC50 values
        
        # Store drug features for later use in recommendations
        self.drug_features = drug_df.set_index('drug_id').drop('drug_id', axis=1, errors='ignore')
        
        # Create a preprocessing pipeline
        # Identify categorical and numerical columns
        categorical_cols = X_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X_df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # Create preprocessor
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
            ])
        
        # Fit and transform the data
        X = preprocessor.fit_transform(X_df)
        y = np.log(y_df.values)  # Log-transform IC50 values for better prediction
        
        self.preprocessor = preprocessor
        
        return X, y
    
    def build_gan_model(self, input_dim, latent_dim=64):
        """
        Build a GAN model for drug response prediction
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
        latent_dim : int
            Dimension of latent space for generator
            
        Returns:
        --------
        tuple
            (generator, discriminator, gan)
        """
        # Generator model
        generator = keras.Sequential([
            keras.layers.Input(shape=(latent_dim + input_dim,)),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dense(1)  # Predict IC50 value
        ])
        
        # Discriminator model
        discriminator = keras.Sequential([
            keras.layers.Input(shape=(input_dim + 1,)),  # Features + IC50
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(1, activation='sigmoid')  # Real or fake prediction
        ])
        
        # Compile discriminator
        discriminator.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy',
            metrics=['accuracy']
        )
        
        # Create GAN
        discriminator.trainable = False
        gan_input = keras.layers.Input(shape=(latent_dim + input_dim,))
        fake_response = generator(gan_input)
        
        # Combine fake response with original input features
        features_input = keras.layers.Lambda(lambda x: x[:, latent_dim:])(gan_input)
        gan_output = discriminator(keras.layers.Concatenate()([features_input, fake_response]))
        
        gan = keras.Model(gan_input, gan_output)
        gan.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5),
            loss='binary_crossentropy'
        )
        
        return generator, discriminator, gan
    
    def build_rl_model(self, input_dim):
        """
        Build a reinforcement learning model for drug response prediction
        
        Parameters:
        -----------
        input_dim : int
            Dimension of input features
            
        Returns:
        --------
        keras.Model
            RL agent model
        """
        # Actor-Critic architecture for RL
        # Shared layers
        input_layer = keras.layers.Input(shape=(input_dim,))
        shared = keras.layers.Dense(256, activation='relu')(input_layer)
        shared = keras.layers.Dense(128, activation='relu')(shared)
        
        # Actor head (policy network)
        actor = keras.layers.Dense(64, activation='relu')(shared)
        actor = keras.layers.Dense(1, activation='linear')(actor)  # Predicted IC50
        
        # Critic head (value network)
        critic = keras.layers.Dense(64, activation='relu')(shared)
        critic = keras.layers.Dense(1, activation='linear')(critic)  # Estimated value
        
        model = keras.Model(inputs=input_layer, outputs=[actor, critic])
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss=['mse', 'mse']
        )
        
        return model
    
    def train_gan(self, X, y, epochs=100, batch_size=32, latent_dim=64):
        """
        Train the GAN model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        latent_dim : int
            Dimension of latent space
        """
        generator, discriminator, gan = self.build_gan_model(X.shape[1], latent_dim)
        
        # Training loop
        d_losses = []
        g_losses = []
        
        for epoch in range(epochs):
            # Train discriminator
            idx = np.random.randint(0, X.shape[0], batch_size)
            real_features = X[idx]
            real_responses = y[idx].reshape(-1, 1)
            
            # Generate fake responses
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_input = np.hstack([noise, real_features])
            fake_responses = generator.predict(gen_input)
            
            # Train discriminator
            real_data = np.hstack([real_features, real_responses])
            fake_data = np.hstack([real_features, fake_responses])
            
            d_loss_real = discriminator.train_on_batch(real_data, np.ones((batch_size, 1)))
            d_loss_fake = discriminator.train_on_batch(fake_data, np.zeros((batch_size, 1)))
            d_loss = 0.5 * np.add(d_loss_real[0], d_loss_fake[0])
            
            # Train generator
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            gen_input = np.hstack([noise, real_features])
            g_loss = gan.train_on_batch(gen_input, np.ones((batch_size, 1)))
            
            d_losses.append(d_loss)
            g_losses.append(g_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Discriminator Loss: {d_loss}, Generator Loss: {g_loss}")
        
        # Store models
        self.model = generator
        
        # Plot training losses
        plt.figure(figsize=(12, 6))
        plt.plot(d_losses, label='Discriminator Loss')
        plt.plot(g_losses, label='Generator Loss')
        plt.legend()
        plt.title('GAN Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('gan_training_losses.png')
        plt.close()
        
        return generator, discriminator
    
    def train_rl(self, X, y, epochs=100, batch_size=32):
        """
        Train the reinforcement learning model
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target values
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size
        """
        rl_model = self.build_rl_model(X.shape[1])
        
        # Define reward function: negative MSE
        def compute_reward(y_true, y_pred):
            return -np.mean((y_true - y_pred) ** 2)
        
        # Training loop
        losses = []
        
        for epoch in range(epochs):
            # Sample batch
            idx = np.random.randint(0, X.shape[0], batch_size)
            batch_X = X[idx]
            batch_y = y[idx].reshape(-1, 1)
            
            # Get predictions
            pred_y, values = rl_model.predict(batch_X)
            
            # Compute rewards
            rewards = np.array([compute_reward(batch_y[i], pred_y[i]) for i in range(batch_size)])
            rewards = rewards.reshape(-1, 1)
            
            # Compute advantage (difference between actual reward and predicted value)
            advantages = rewards - values
            
            # Train model with actor-critic loss
            loss = rl_model.train_on_batch(batch_X, [batch_y, rewards])
            losses.append(loss[0])  # Total loss
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}, Loss: {loss[0]}, MSE: {loss[1]}, Value Loss: {loss[2]}")
        
        # Store the model
        self.model = rl_model
        
        # Plot training loss
        plt.figure(figsize=(10, 6))
        plt.plot(losses)
        plt.title('RL Model Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.savefig('rl_training_loss.png')
        plt.close()
        
        return rl_model
    
    def train(self, X, y):
        """
        Train the model based on the selected model type
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target values
        """
        if self.model_type == "gan":
            return self.train_gan(X, y)
        elif self.model_type == "rl":
            return self.train_rl(X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def predict_drug_response(self, patient_features, drug_features):
        """
        Predict the response for a specific patient to a specific drug
        
        Parameters:
        -----------
        patient_features : pandas.DataFrame
            Patient-specific features
        drug_features : pandas.DataFrame
            Drug molecular features
            
        Returns:
        --------
        float
            Predicted IC50 value
        """
        # Combine features
        combined_features = pd.concat([patient_features, drug_features], axis=1)
        
        # Preprocess features
        X = self.preprocessor.transform(combined_features)
        
        # Make prediction based on model type
        if self.model_type == "gan":
            # For GAN, we need to add noise vector
            latent_dim = 64  # Same as in training
            noise = np.random.normal(0, 1, (1, latent_dim))
            gen_input = np.hstack([noise, X])
            pred = self.model.predict(gen_input)
            ic50 = np.exp(pred[0][0])  # Transform back from log scale
        else:  # RL model
            pred, _ = self.model.predict(X)
            ic50 = np.exp(pred[0][0])  # Transform back from log scale
            
        return ic50
    
    def recommend_optimal_drug(self, patient_features, available_drugs=None, top_n=3):
        """
        Recommend the optimal drug(s) for a given patient
        
        Parameters:
        -----------
        patient_features : pandas.DataFrame
            Patient-specific features
        available_drugs : list, optional
            List of available drug IDs to consider
        top_n : int
            Number of top recommendations to return
            
        Returns:
        --------
        list
            List of tuples (drug_id, predicted_ic50)
        """
        if available_drugs is None:
            # Use all drugs in the dataset
            available_drugs = self.drug_features.index.tolist()
        
        results = []
        for drug_id in available_drugs:
            drug_feat = self.drug_features.loc[[drug_id]]
            # Replicate patient features for each drug
            patient_feat_repeated = pd.DataFrame([patient_features.iloc[0].to_dict()])
            
            # Predict response
            ic50 = self.predict_drug_response(patient_feat_repeated, drug_feat)
            results.append((drug_id, ic50))
        
        # Sort by increasing IC50 (lower is better)
        results.sort(key=lambda x: x[1])
        
        return results[:top_n]
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model on test data
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test feature matrix
        y_test : numpy.ndarray
            Test target values
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model_type == "gan":
            # For GAN, we need to add noise vector
            latent_dim = 64  # Same as in training
            noise = np.random.normal(0, 1, (X_test.shape[0], latent_dim))
            gen_input = np.hstack([noise, X_test])
            y_pred = self.model.predict(gen_input).flatten()
        else:  # RL model
            y_pred, _ = self.model.predict(X_test)
            y_pred = y_pred.flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Transform back to original scale for interpretability
        y_test_orig = np.exp(y_test)
        y_pred_orig = np.exp(y_pred)
        
        # Scatter plot of predicted vs actual
        plt.figure(figsize=(10, 8))
        plt.scatter(y_test, y_pred, alpha=0.5)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.xlabel('Actual log(IC50)')
        plt.ylabel('Predicted log(IC50)')
        plt.title('Predicted vs Actual Drug Response')
        plt.savefig('prediction_evaluation.png')
        plt.close()
        
        # Distribution of errors
        errors = y_pred - y_test
        plt.figure(figsize=(10, 6))
        plt.hist(errors, bins=50)
        plt.xlabel('Prediction Error')
        plt.ylabel('Frequency')
        plt.title('Distribution of Prediction Errors')
        plt.savefig('error_distribution.png')
        plt.close()
        
        return {
            'mse': mse,
            'r2': r2,
            'log_scale': {
                'mse': mse,
                'r2': r2
            },
            'original_scale': {
                'mse': mean_squared_error(y_test_orig, y_pred_orig),
                'r2': r2_score(y_test_orig, y_pred_orig)
            }
        }

def main():
    """
    Main function to demonstrate the personalized drug response prediction system
    """
    # Initialize model
    model_type = "gan"  # or "rl"
    predictor = PersonalizedDrugResponsePredictor(model_type=model_type)
    
    # Download and load data
    print("Downloading and preparing GDSC data...")
    response_df, drug_df, genomic_df = predictor.download_gdsc_data()
    
    # Preprocess data
    print("Preprocessing data...")
    X, y = predictor.load_and_preprocess_data(response_df, drug_df, genomic_df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    print(f"Training {model_type.upper()} model...")
    predictor.train(X_train, y_train)
    
    # Evaluate model
    print("Evaluating model...")
    metrics = predictor.evaluate(X_test, y_test)
    print(f"Test MSE: {metrics['mse']:.4f}")
    print(f"Test R²: {metrics['r2']:.4f}")
    
    # Demo: Patient-specific drug recommendation
    print("\nDemonstrating drug recommendation for a sample patient...")
    
    # Get a sample patient from test set (in real scenario, this would be a new patient)
    sample_idx = np.random.randint(0, len(genomic_df))
    sample_patient = genomic_df.iloc[[sample_idx]]
    
    # Recommend drugs
    recommendations = predictor.recommend_optimal_drug(sample_patient, top_n=5)
    
    print(f"Patient ID: {sample_patient.iloc[0]['cell_line_id']}")
    print("Top drug recommendations:")
    for i, (drug_id, ic50) in enumerate(recommendations, 1):
        print(f"{i}. Drug: {drug_id}, Predicted IC50: {ic50:.2f}")
    
    print("\nNote: Lower IC50 values indicate higher drug sensitivity")
    
    # Ethical and regulatory considerations
    print("\nEthical and Regulatory Considerations:")
    print("1. This model should be used as a decision support tool, not as a replacement for clinical judgment")
    print("2. Model predictions should be validated in clinical trials before use in patient care")
    print("3. Patient data privacy and security must be ensured according to regulations (e.g., HIPAA)")
    print("4. The model should be regularly updated as new data becomes available")
    print("5. Transparency in the AI decision-making process is essential for clinical adoption")

if __name__ == "__main__":
    main()