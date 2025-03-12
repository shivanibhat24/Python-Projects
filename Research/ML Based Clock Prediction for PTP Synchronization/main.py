import argparse
import os
import pandas as pd
import numpy as np
import time
from data_processing import PTPDataProcessor
from models import DriftPredictionModel
from clock_servo import AdaptiveClockServo
from ptp_integration import PTPIntegration

def train_model(args):
    """Train the ML model using historical PTP data."""
    print(f"Loading data from {args.data_file}")
    processor = PTPDataProcessor(sequence_length=args.sequence_length, prediction_horizon=args.prediction_horizon)
    data = processor.load_data(args.data_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    print("Preparing training data...")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = processor.prepare_training_data(
        data, 
        target_col=args.target_column,
        test_size=0.2,
        validation_size=0.1
    )
    
    print("Building model...")
    model = DriftPredictionModel(
        sequence_length=args.sequence_length,
        n_features=X_train.shape[2],
        prediction_horizon=args.prediction_horizon
    )
    
    print("Training model...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        batch_size=args.batch_size,
        epochs=args.epochs
    )
    
    print("Evaluating model...")
    y_pred = model.predict(X_test, apply_kalman=False)
    
    # Calculate metrics
    mse = np.mean((y_test - y_pred) ** 2)
    mae = np.mean(np.abs(y_test - y_pred))
    
    print(f"Test MSE: {mse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    
    # Save model
    model.save(args.model_file)
    print(f"Model saved to {args.model_file}")

def run_inference(args):
    """Run the model in inference mode with PTP integration."""
    print(f"Loading model from {args.model_file}")
    
    # Load sample data to get feature dimensions
    processor = PTPDataProcessor(sequence_length=args.sequence_length, prediction_horizon=args.prediction_horizon)
    data = processor.load_data(args.data_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract feature dimensions
    feature_columns = [col for col in data.columns if col not in ['timestamp', args.target_column]]
    n_features = len(feature_columns)
    
    # Load model
    model = DriftPredictionModel.load(
        args.model_file,
        sequence_length=args.sequence_length,
        n_features=n_features,
        prediction_horizon=args.prediction_horizon
    )
    
    # Create servo controller
    servo = AdaptiveClockServo(model, update_interval_ms=args.update_interval)
    
    # Create PTP integration
    ptp = PTPIntegration(servo)
    
    print(f"Starting PTPd on interface {args.interface}...")
    if ptp.start_ptpd(interface=args.interface, master=args.master):
        print("Starting monitoring...")
        ptp.start_monitoring()
        
        try:
            # Run until interrupted
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Stopping...")
        finally:
            ptp.stop()
    else:
        print("Failed to start PTPd. Exiting.")

def main():
    """Main entry point for ML-PTP sync system."""
    parser = argparse.ArgumentParser(description='ML-based PTP Clock Synchronization')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train ML model')
    train_parser.add_argument('--data-file', type=str, required=True, help='Path to training data CSV')
    train_parser.add_argument('--model-file', type=str, default='ml_ptp_model.h5', help='Path to save model')
    train_parser.add_argument('--target-column', type=str, default='offset', help='Target column to predict')
    train_parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length for LSTM')
    train_parser.add_argument('--prediction-horizon', type=int, default=20, help='Prediction horizon')
    train_parser.add_argument('--batch-size', type=int, default=32, help='Training batch size')
    train_parser.add_argument('--epochs', type=int, default=100, help='Training epochs')
    
    # Run command
    run_parser = subparsers.add_parser('run', help='Run inference with PTP integration')
    run_parser.add_argument('--data-file', type=str, required=True, help='Path to sample data CSV (for feature dimensions)')
    run_parser.add_argument('--model-file', type=str, required=True, help='Path to trained model')
    run_parser.add_argument('--target-column', type=str, default='offset', help='Target column to predict')
    run_parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length for LSTM')
    run_parser.add_argument('--prediction-horizon', type=int, default=20, help='Prediction horizon')
    run_parser.add_argument('--interface', type=str, default='eth0', help='Network interface for PTP')
    run_parser.add_argument('--update-interval', type=int, default=1000, help='Update interval in milliseconds')
    run_parser.add_argument('--master', action='store_true', help='Run as PTP master instead of slave')
    
    # Simulation command
    sim_parser = subparsers.add_parser('simulate', help='Run simulation without PTP hardware')
    sim_parser.add_argument('--data-file', type=str, required=True, help='Path to test data CSV')
    sim_parser.add_argument('--model-file', type=str, required=True, help='Path to trained model')
    sim_parser.add_argument('--target-column', type=str, default='offset', help='Target column to predict')
    sim_parser.add_argument('--sequence-length', type=int, default=100, help='Sequence length for LSTM')
    sim_parser.add_argument('--prediction-horizon', type=int, default=20, help='Prediction horizon')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args)
    elif args.command == 'run':
        run_inference(args)
    elif args.command == 'simulate':
        simulate(args)
    else:
        parser.print_help()

def simulate(args):
    """Run simulation using pre-recorded data."""
    print(f"Loading model from {args.model_file}")
    
    # Load test data
    processor = PTPDataProcessor(sequence_length=args.sequence_length, prediction_horizon=args.prediction_horizon)
    data = processor.load_data(args.data_file)
    
    if data is None:
        print("Failed to load data. Exiting.")
        return
    
    # Extract feature dimensions
    feature_columns = [col for col in data.columns if col not in ['timestamp', args.target_column]]
    n_features = len(feature_columns)
    
    # Load model
    model = DriftPredictionModel.load(
        args.model_file,
        sequence_length=args.sequence_length,
        n_features=n_features,
        prediction_horizon=args.prediction_horizon
    )
    
    # Create servo controller
    servo = AdaptiveClockServo(model)
    
    # Prepare sequences
    X, y = processor.create_sequences(data, target_col=args.target_column)
    
    # Run simulation
    print("Running simulation...")
    corrections = []
    true_offsets = []
    predicted_offsets = []
    
    for i in range(len(X)):
        # Current offset (using ground truth)
        true_offset = processor.target_scaler.inverse_transform(y[i].reshape(-1, 1))[0, 0]
        
        # Get model prediction
        prediction = model.predict(X[i:i+1])
        predicted_offset = processor.target_scaler.inverse_transform(prediction[0].reshape(-1, 1))[0, 0]
        
        # Calculate correction
        correction = servo.compute_correction(true_offset, prediction[0])
        
        # Store results
        corrections.append(correction)
        true_offsets.append(true_offset)
        predicted_offsets.append(predicted_offset)
        
        # Print progress
        if i % 100 == 0:
            print(f"Processed {i}/{len(X)} samples")
    
    # Calculate performance metrics
    original_rmse = np.sqrt(np.mean(np.array(true_offsets) ** 2))
    corrected_offsets = np.array(true_offsets) - np.array(corrections)
    corrected_rmse = np.sqrt(np.mean(corrected_offsets ** 2))
    
    print("\nSimulation Results:")
    print(f"Original RMSE: {original_rmse:.2f} ns")
    print(f"Corrected RMSE: {corrected_rmse:.2f} ns")
    print(f"Improvement: {100 * (original_rmse - corrected_rmse) / original_rmse:.2f}%")
    
    # Save simulation results
    results = pd.DataFrame({
        'true_offset': true_offsets,
        'predicted_offset': predicted_offsets,
        'correction': corrections,
        'corrected_offset': corrected_offsets
    })
    
    results_file = 'simulation_results.csv'
    results.to_csv(results_file, index=False)
    print(f"Results saved to {results_file}")

if __name__ == "__main__":
    main()
