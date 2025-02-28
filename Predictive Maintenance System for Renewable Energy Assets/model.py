import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, f1_score, precision_score, recall_score
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping
import warnings
warnings.filterwarnings('ignore')

class RenewableEnergyPredictiveMaintenance:
    """
    A system for predicting maintenance needs in renewable energy assets
    using deep learning-based anomaly detection on SCADA data.
    """
    
    def __init__(self, scada_data_path, failure_logs_path=None):
        """
        Initialize the predictive maintenance system.
        
        Args:
            scada_data_path: Path to the SCADA data CSV file
            failure_logs_path: Path to historical failure logs (optional)
        """
        self.scada_data_path = scada_data_path
        self.failure_logs_path = failure_logs_path
        self.scada_data = None
        self.failure_logs = None
        self.autoencoder = None
        self.rul_model = None
        self.anomaly_threshold = None
        self.scaler = None
        self.sequence_length = 24  # Default: use 24 hours of data for prediction
        
    def load_and_preprocess_data(self):
        """
        Load and preprocess SCADA data and failure logs.
        Handle missing values, outliers, and feature scaling.
        """
        print("Loading and preprocessing data...")
        
        # Load SCADA data
        self.scada_data = pd.read_csv(self.scada_data_path)
        
        # Load failure logs if available
        if self.failure_logs_path:
            self.failure_logs = pd.read_csv(self.failure_logs_path)
        
        # Convert timestamp to datetime
        if 'timestamp' in self.scada_data.columns:
            self.scada_data['timestamp'] = pd.to_datetime(self.scada_data['timestamp'])
            self.scada_data.set_index('timestamp', inplace=True)
        
        # Handle missing values
        self.scada_data = self._handle_missing_values(self.scada_data)
        
        # Detect and handle outliers
        self.scada_data = self._handle_outliers(self.scada_data)
        
        # Create relevant features
        self.scada_data = self._engineer_features(self.scada_data)
        
        # Scale the features
        self._scale_features()
        
        print("Data preprocessing completed.")
        return self.scada_data
    
    def _handle_missing_values(self, data):
        """Handle missing values in the dataset."""
        # Check percentage of missing values in each column
        missing_percentage = data.isnull().mean() * 100
        
        # For columns with less than 10% missing values, use interpolation
        for col in data.columns:
            if missing_percentage[col] < 10:
                data[col] = data[col].interpolate(method='time')
            # For columns with 10-30% missing values, use forward and backward fill
            elif 10 <= missing_percentage[col] < 30:
                data[col] = data[col].fillna(method='ffill').fillna(method='bfill')
            # For columns with more than 30% missing values, drop the column
            elif missing_percentage[col] >= 30:
                print(f"Warning: Dropping column {col} due to high percentage of missing values ({missing_percentage[col]:.2f}%)")
                data.drop(col, axis=1, inplace=True)
        
        return data
    
    def _handle_outliers(self, data):
        """Detect and handle outliers in the dataset."""
        for col in data.select_dtypes(include=np.number).columns:
            # Calculate the IQR
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            
            # Define bounds for outliers
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            # Replace outliers with bounds
            outliers = ((data[col] < lower_bound) | (data[col] > upper_bound))
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                print(f"Detected {outlier_count} outliers in column {col}")
                data.loc[data[col] < lower_bound, col] = lower_bound
                data.loc[data[col] > upper_bound, col] = upper_bound
        
        return data
    
    def _engineer_features(self, data):
        """Create additional features from the data."""
        # Add time-based features
        if isinstance(data.index, pd.DatetimeIndex):
            data['hour'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['month'] = data.index.month
            data['season'] = (data['month'] % 12 + 3) // 3
        
        # Add lagged features for time series analysis
        for col in data.select_dtypes(include=np.number).columns:
            if col not in ['hour', 'day_of_week', 'month', 'season']:
                data[f'{col}_lag_1'] = data[col].shift(1)
                data[f'{col}_lag_12'] = data[col].shift(12)
                data[f'{col}_lag_24'] = data[col].shift(24)
        
        # Calculate rolling statistics
        for col in data.select_dtypes(include=np.number).columns:
            if col not in ['hour', 'day_of_week', 'month', 'season']:
                data[f'{col}_rolling_mean_6'] = data[col].rolling(window=6).mean()
                data[f'{col}_rolling_std_6'] = data[col].rolling(window=6).std()
                data[f'{col}_rolling_mean_24'] = data[col].rolling(window=24).mean()
                data[f'{col}_rolling_std_24'] = data[col].rolling(window=24).std()
        
        # Drop rows with NaN values introduced by lag and rolling operations
        data = data.dropna()
        
        return data
    
    def _scale_features(self):
        """Scale features using Min-Max scaling."""
        self.scaler = MinMaxScaler()
        numerical_cols = self.scada_data.select_dtypes(include=np.number).columns
        self.scada_data[numerical_cols] = self.scaler.fit_transform(self.scada_data[numerical_cols])
    
    def _create_sequences(self, data, sequence_length):
        """Create sequences for time series prediction."""
        xs, ys = [], []
        for i in range(len(data) - sequence_length):
            x = data.iloc[i:(i + sequence_length)].values
            y = data.iloc[i + sequence_length].values
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)
    
    def build_autoencoder(self, encoding_dim=10):
        """
        Build an autoencoder model for anomaly detection.
        
        Args:
            encoding_dim: Dimension of the encoded representation
        """
        # Get input dimension from data
        input_dim = self.scada_data.shape[1]
        
        # Define the architecture
        input_layer = Input(shape=(input_dim,))
        
        # Encoder layers
        encoded = Dense(128, activation='relu')(input_layer)
        encoded = Dense(64, activation='relu')(encoded)
        encoded = Dense(encoding_dim, activation='relu')(encoded)
        
        # Decoder layers
        decoded = Dense(64, activation='relu')(encoded)
        decoded = Dense(128, activation='relu')(decoded)
        decoded = Dense(input_dim, activation='sigmoid')(decoded)
        
        # Define the autoencoder model
        self.autoencoder = Model(input_layer, decoded)
        
        # Compile the model
        self.autoencoder.compile(optimizer='adam', loss='mse')
        
        print("Autoencoder model built with architecture:")
        self.autoencoder.summary()
        
        return self.autoencoder
    
    def build_rul_model(self):
        """
        Build a Recurrent Neural Network model for
        Remaining Useful Life (RUL) prediction.
        """
        # Get input shape from sequence length and number of features
        input_dim = self.scada_data.shape[1]
        
        # Define the RUL model
        self.rul_model = Sequential([
            Bidirectional(LSTM(64, return_sequences=True), input_shape=(self.sequence_length, input_dim)),
            Dropout(0.2),
            Bidirectional(LSTM(32)),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)  # Single output for RUL prediction
        ])
        
        # Compile the model
        self.rul_model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        print("RUL model built with architecture:")
        self.rul_model.summary()
        
        return self.rul_model
    
    def train_autoencoder(self, train_data=None, validation_split=0.2, epochs=50, batch_size=32):
        """
        Train the autoencoder model for anomaly detection.
        
        Args:
            train_data: Training data (if None, uses preprocessed SCADA data)
            validation_split: Portion of data to use for validation
            epochs: Number of training epochs
            batch_size: Batch size for training
        """
        if self.autoencoder is None:
            self.build_autoencoder()
        
        if train_data is None:
            train_data = self.scada_data
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the model
        print("Training autoencoder model...")
        history = self.autoencoder.fit(
            train_data,
            train_data,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Calculate reconstruction error on training data
        train_predictions = self.autoencoder.predict(train_data)
        train_mse = np.mean(np.power(train_data - train_predictions, 2), axis=1)
        
        # Set anomaly threshold as the 95th percentile of errors
        self.anomaly_threshold = np.percentile(train_mse, 95)
        print(f"Anomaly threshold set at: {self.anomaly_threshold:.6f}")
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Autoencoder Training History')
        plt.legend()
        plt.show()
        
        return history
    
    def train_rul_model(self, train_data=None, epochs=50, batch_size=32, validation_split=0.2):
        """
        Train the RUL prediction model.
        
        Args:
            train_data: Training data (if None, uses preprocessed SCADA data)
            epochs: Number of training epochs
            batch_size: Batch size for training
            validation_split: Portion of data to use for validation
        """
        if self.rul_model is None:
            self.build_rul_model()
        
        if train_data is None:
            train_data = self.scada_data
        
        # Create sequences for time series prediction
        X, y = self._create_sequences(train_data, self.sequence_length)
        
        # Split into training and validation sets
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=validation_split, shuffle=False)
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Train the model
        print("Training RUL prediction model...")
        history = self.rul_model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=1
        )
        
        # Plot training history
        plt.figure(figsize=(10, 6))
        plt.plot(history.history['loss'], label='Training Loss')
        plt.plot(history.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('RUL Model Training History')
        plt.legend()
        plt.show()
        
        return history
    
    def detect_anomalies(self, data=None):
        """
        Detect anomalies in the SCADA data using the trained autoencoder.
        
        Args:
            data: Data to analyze for anomalies (if None, uses preprocessed SCADA data)
            
        Returns:
            DataFrame with original data and anomaly scores
        """
        if self.autoencoder is None:
            raise ValueError("Autoencoder model has not been trained. Call train_autoencoder() first.")
        
        if data is None:
            data = self.scada_data
        
        # Make predictions with the autoencoder
        predictions = self.autoencoder.predict(data)
        
        # Calculate mean squared error for each sample
        mse = np.mean(np.power(data - predictions, 2), axis=1)
        
        # Create a copy of the original data
        result_df = pd.DataFrame(data.copy())
        
        # Add anomaly score column
        result_df['anomaly_score'] = mse
        
        # Add anomaly flag column (1 if anomaly, 0 if normal)
        result_df['is_anomaly'] = (mse > self.anomaly_threshold).astype(int)
        
        # Add severity score based on how far above threshold
        result_df['severity'] = (mse - self.anomaly_threshold) / self.anomaly_threshold
        result_df.loc[result_df['severity'] < 0, 'severity'] = 0
        
        print(f"Detected {result_df['is_anomaly'].sum()} anomalies out of {len(result_df)} samples")
        
        return result_df
    
    def predict_rul(self, data=None, days_ahead=30):
        """
        Predict the Remaining Useful Life (RUL) for components.
        
        Args:
            data: Data to use for prediction (if None, uses preprocessed SCADA data)
            days_ahead: Number of days to forecast ahead
            
        Returns:
            DataFrame with RUL predictions
        """
        if self.rul_model is None:
            raise ValueError("RUL model has not been trained. Call train_rul_model() first.")
        
        if data is None:
            data = self.scada_data
        
        # Create sequences for prediction
        X, _ = self._create_sequences(data, self.sequence_length)
        
        # Predict RUL values
        rul_predictions = self.rul_model.predict(X)
        
        # Create a DataFrame with the results
        prediction_dates = data.index[self.sequence_length:len(X) + self.sequence_length]
        rul_df = pd.DataFrame({
            'timestamp': prediction_dates,
            'predicted_rul': rul_predictions.flatten()
        })
        rul_df.set_index('timestamp', inplace=True)
        
        # Generate future dates for forecasting
        last_date = rul_df.index[-1]
        future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=days_ahead)
        
        # Use last sequence for forecasting
        last_sequence = X[-1]
        forecasted_rul = []
        current_sequence = last_sequence.copy()
        
        for _ in range(days_ahead):
            # Predict the next day's values
            predicted = self.rul_model.predict(current_sequence.reshape(1, self.sequence_length, -1))
            forecasted_rul.append(predicted[0][0])
            
            # Update the sequence for next prediction (rolling window approach)
            current_sequence = np.roll(current_sequence, -1, axis=0)
            current_sequence[-1] = predicted
        
        # Add forecasts to the DataFrame
        forecast_df = pd.DataFrame({
            'timestamp': future_dates,
            'predicted_rul': forecasted_rul
        })
        forecast_df.set_index('timestamp', inplace=True)
        forecast_df['is_forecast'] = 1
        
        # Add forecast flag to historical data
        rul_df['is_forecast'] = 0
        
        # Combine historical and forecasted data
        combined_df = pd.concat([rul_df, forecast_df])
        
        return combined_df
    
    def visualize_anomalies(self, anomaly_df, columns_to_plot):
        """
        Visualize detected anomalies in the data.
        
        Args:
            anomaly_df: DataFrame with anomaly scores and flags
            columns_to_plot: List of column names to visualize
        """
        for col in columns_to_plot:
            if col in anomaly_df.columns:
                plt.figure(figsize=(15, 6))
                
                # Plot the original values
                plt.plot(anomaly_df.index, anomaly_df[col], label=col, color='blue')
                
                # Highlight anomalies
                anomalies = anomaly_df[anomaly_df['is_anomaly'] == 1]
                plt.scatter(anomalies.index, anomalies[col], color='red', label='Anomalies', s=50)
                
                plt.title(f'Anomaly Detection for {col}')
                plt.xlabel('Timestamp')
                plt.ylabel('Value')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
    
    def visualize_rul_predictions(self, rul_df, component_name="Component"):
        """
        Visualize RUL predictions over time.
        
        Args:
            rul_df: DataFrame with RUL predictions
            component_name: Name of the component for plotting
        """
        plt.figure(figsize=(15, 6))
        
        # Plot historical RUL values
        historical = rul_df[rul_df['is_forecast'] == 0]
        plt.plot(historical.index, historical['predicted_rul'], label='Historical RUL', color='blue')
        
        # Plot forecasted RUL values
        forecast = rul_df[rul_df['is_forecast'] == 1]
        plt.plot(forecast.index, forecast['predicted_rul'], label='Forecasted RUL', color='red', linestyle='--')
        
        # Add maintenance threshold line
        maintenance_threshold = 30  # Example threshold
        plt.axhline(y=maintenance_threshold, color='green', linestyle='-', label=f'Maintenance Threshold ({maintenance_threshold} days)')
        
        # Highlight when RUL crosses below the threshold
        below_threshold = rul_df[rul_df['predicted_rul'] < maintenance_threshold]
        if len(below_threshold) > 0:
            first_alarm = below_threshold.iloc[0]
            plt.scatter(first_alarm.name, first_alarm['predicted_rul'], color='orange', s=100, zorder=5, label='Maintenance Alert')
            plt.annotate(f'Maintenance needed by: {first_alarm.name.date()}', 
                        xy=(first_alarm.name, first_alarm['predicted_rul']),
                        xytext=(first_alarm.name, first_alarm['predicted_rul'] + 10),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5))
        
        plt.title(f'Remaining Useful Life Prediction for {component_name}')
        plt.xlabel('Date')
        plt.ylabel('Predicted RUL (Days)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def generate_maintenance_plan(self, anomaly_df, rul_df, maintenance_threshold=30):
        """
        Generate a maintenance plan based on anomaly detection and RUL prediction.
        
        Args:
            anomaly_df: DataFrame with anomaly scores and flags
            rul_df: DataFrame with RUL predictions
            maintenance_threshold: RUL threshold for scheduling maintenance
            
        Returns:
            DataFrame with maintenance recommendations
        """
        # Merge anomaly and RUL data
        # First reset index to make timestamp a column
        anomaly_reset = anomaly_df.reset_index()
        rul_reset = rul_df.reset_index()
        
        # Merge the DataFrames on timestamp
        merged_df = pd.merge_asof(
            anomaly_reset.sort_values('timestamp'),
            rul_reset.sort_values('timestamp'),
            on='timestamp'
        )
        
        # Set timestamp back as index
        merged_df.set_index('timestamp', inplace=True)
        
        # Identify components/sensors requiring maintenance
        merged_df['needs_maintenance'] = (merged_df['predicted_rul'] < maintenance_threshold) | (merged_df['is_anomaly'] == 1)
        
        # Get list of components requiring maintenance
        components_needing_maintenance = []
        
        # Generate maintenance recommendations
        maintenance_plan = []
        
        for component in components_needing_maintenance:
            # Find the earliest date when maintenance is needed
            component_data = merged_df[merged_df['component'] == component]
            if len(component_data[component_data['needs_maintenance']]) > 0:
                earliest_date = component_data[component_data['needs_maintenance']].index.min()
                severity = component_data.loc[earliest_date, 'severity']
                rul = component_data.loc[earliest_date, 'predicted_rul']
                
                # Determine maintenance priority based on severity and RUL
                if rul < 7 or severity > 5:
                    priority = "Critical"
                elif rul < 14 or severity > 3:
                    priority = "High"
                else:
                    priority = "Medium"
                
                # Add to maintenance plan
                maintenance_plan.append({
                    'component': component,
                    'maintenance_date': earliest_date,
                    'estimated_rul': rul,
                    'anomaly_severity': severity,
                    'priority': priority
                })
        
        # Create maintenance plan DataFrame
        maintenance_df = pd.DataFrame(maintenance_plan)
        
        if len(maintenance_df) > 0:
            # Sort by priority and date
            priority_order = {"Critical": 0, "High": 1, "Medium": 2}
            maintenance_df['priority_order'] = maintenance_df['priority'].map(priority_order)
            maintenance_df.sort_values(['priority_order', 'maintenance_date'], inplace=True)
            maintenance_df.drop('priority_order', axis=1, inplace=True)
        
        return maintenance_df
    
    def run_full_analysis(self, columns_to_plot=None, maintenance_threshold=30, days_to_forecast=30):
        """
        Run a complete analysis pipeline including anomaly detection,
        RUL prediction, and maintenance planning.
        
        Args:
            columns_to_plot: List of columns to visualize
            maintenance_threshold: RUL threshold for maintenance scheduling
            days_to_forecast: Number of days to forecast RUL
            
        Returns:
            Tuple of (anomaly_df, rul_df, maintenance_plan)
        """
        # Detect anomalies
        anomaly_df = self.detect_anomalies()
        
        # Predict RUL
        rul_df = self.predict_rul(days_ahead=days_to_forecast)
        
        # Generate maintenance plan
        maintenance_plan = self.generate_maintenance_plan(anomaly_df, rul_df, maintenance_threshold)
        
        # Visualize results
        if columns_to_plot:
            self.visualize_anomalies(anomaly_df, columns_to_plot)
        
        self.visualize_rul_predictions(rul_df)
        
        # Print maintenance plan
        if len(maintenance_plan) > 0:
            print("\nMaintenance Plan:")
            print(maintenance_plan)
        else:
            print("\nNo immediate maintenance required.")
        
        return anomaly_df, rul_df, maintenance_plan


# Example usage
if __name__ == "__main__":
    # Sample execution with Wind Turbine SCADA data
    # Note: Replace with actual file path to your SCADA data
    scada_data_path = "wind_turbine_scada_data.csv"
    
    try:
        # Initialize the predictive maintenance system
        pm_system = RenewableEnergyPredictiveMaintenance(scada_data_path)
        
        # Load and preprocess data
        pm_system.load_and_preprocess_data()
        
        # Build and train models
        pm_system.build_autoencoder()
        pm_system.train_autoencoder(epochs=30)
        
        pm_system.build_rul_model()
        pm_system.train_rul_model(epochs=50)
        
        # Run full analysis
        columns_to_plot = ['power_output', 'rotor_speed', 'wind_speed', 'temperature']
        anomaly_df, rul_df, maintenance_plan = pm_system.run_full_analysis(
            columns_to_plot=columns_to_plot,
            maintenance_threshold=30,
            days_to_forecast=60
        )
        
        print("Analysis completed successfully!")
        
    except FileNotFoundError:
        print(f"Error: SCADA data file not found at {scada_data_path}")
        print("Please update the file path or use the following code as a template for your data.")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
