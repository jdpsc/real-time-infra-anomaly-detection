#!/usr/bin/env python3
"""
Generate synthetic time series data with anomalies for training.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import argparse
from typing import Tuple, List
import sys
from sklearn.model_selection import train_test_split
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.logger_config import get_logger
from config.config_loader import load_config as load_config_new

# Set up logging
logger = get_logger(__name__)



def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load configuration from YAML file with only required keys for data generation."""
    required_keys = [
        'data.training_samples',
        'data.features',
        'data.anomaly_rate', 
        'data.feature_names'
    ]
    return load_config_new(config_path, required_keys)


def generate_normal_data(n_samples: int, n_features: int, random_seed: int = 42) -> np.ndarray:
    """
    Generate normal time series data with realistic patterns.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of features
        random_seed: Random seed for reproducible results
    
    Returns:
        numpy array of shape (n_samples, n_features)
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    data = np.zeros((n_samples, n_features))
    
    # Generate base patterns for each feature
    t = np.arange(n_samples)
    
    # Feature 1: CPU usage (daily pattern + noise)
    daily_pattern = 30 + 20 * np.sin(2 * np.pi * t / 1440)  # 1440 minutes in a day
    weekly_pattern = 10 * np.sin(2 * np.pi * t / 10080)  # 10080 minutes in a week
    data[:, 0] = daily_pattern + weekly_pattern + np.random.normal(0, 5, n_samples)
    data[:, 0] = np.clip(data[:, 0], 0, 100)  # CPU usage between 0-100%
    
    # Feature 2: Memory usage (gradual increase + daily pattern)
    memory_trend = 40 + (t / n_samples) * 20  # Gradual increase
    memory_daily = 10 * np.sin(2 * np.pi * t / 1440)
    data[:, 1] = memory_trend + memory_daily + np.random.normal(0, 3, n_samples)
    data[:, 1] = np.clip(data[:, 1], 0, 100)  # Memory usage between 0-100%
    
    # Feature 3: Network throughput (bursty pattern)
    base_network = 20 + 10 * np.sin(2 * np.pi * t / 720)  # 12-hour pattern
    burst_mask = np.random.random(n_samples) < 0.1  # 10% chance of burst
    burst_values = np.random.uniform(50, 80, n_samples)
    data[:, 2] = base_network + burst_mask * burst_values + np.random.normal(0, 5, n_samples)
    data[:, 2] = np.clip(data[:, 2], 0, 100)
    
    # Feature 4: Disk I/O (correlated with CPU)
    data[:, 3] = 0.6 * data[:, 0] + 0.4 * np.random.uniform(10, 40, n_samples)
    data[:, 3] += np.random.normal(0, 4, n_samples)
    data[:, 3] = np.clip(data[:, 3], 0, 100)
    
    return data


def inject_anomalies(data: np.ndarray, anomaly_rate: float, random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Inject various types of anomalies into the data.
    
    Args:
        data: Normal data array
        anomaly_rate: Fraction of samples to make anomalous
        random_seed: Random seed for reproducible results
    
    Returns:
        Tuple of (anomalous_data, labels) where labels are 0 for normal, 1 for anomaly
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    n_samples = data.shape[0]
    n_anomalies = int(n_samples * anomaly_rate)
    
    # Initialize labels (0 = normal, 1 = anomaly)
    labels = np.zeros(n_samples, dtype=int)
    
    # Randomly select indices for anomalies
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)
    labels[anomaly_indices] = 1
    
    # Create copy of data to modify
    anomalous_data = data.copy()
    
    # Define different types of anomalies
    anomaly_types = ['spike', 'drop', 'shift', 'noise', 'correlation_break']
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(anomaly_types)
        feature_idx = np.random.randint(0, data.shape[1])
        
        if anomaly_type == 'spike':
            # Sudden spike in one feature
            anomalous_data[idx, feature_idx] = np.random.uniform(80, 100)
        
        elif anomaly_type == 'drop':
            # Sudden drop in one feature
            anomalous_data[idx, feature_idx] = np.random.uniform(0, 20)
        
        elif anomaly_type == 'shift':
            # Shift all features up or down
            shift = np.random.uniform(-30, 30)
            anomalous_data[idx, :] += shift
        
        elif anomaly_type == 'noise':
            # Add high noise to all features
            anomalous_data[idx, :] += np.random.normal(0, 20, data.shape[1])
        
        elif anomaly_type == 'correlation_break':
            # Break the correlation between features
            anomalous_data[idx, :] = np.random.uniform(0, 100, data.shape[1])
        
        # Ensure values stay within bounds
        anomalous_data[idx, :] = np.clip(anomalous_data[idx, :], 0, 100)
    
    return anomalous_data, labels


def create_dataframe(data: np.ndarray, labels: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with timestamps and feature names.
    
    Args:
        data: Feature data
        labels: Anomaly labels
        feature_names: Names of features
    
    Returns:
        DataFrame with all data
    """
    n_samples = data.shape[0]
    
    # Create timestamps (1-minute intervals)
    start_time = datetime.now() - timedelta(minutes=n_samples)
    timestamps = [start_time + timedelta(minutes=i) for i in range(n_samples)]
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=feature_names)
    df['timestamp'] = timestamps
    df['is_anomaly'] = labels
    
    # Reorder columns
    columns = ['timestamp'] + feature_names + ['is_anomaly']
    df = df[columns]
    
    return df


def split_data_reproducibly(df: pd.DataFrame, validation_split: float = 0.2, random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and validation sets reproducibly.
    
    Args:
        df: Full dataset
        validation_split: Fraction of data to use for validation
        random_seed: Random seed for reproducible splits
    
    Returns:
        Tuple of (training_df, validation_df)
    """
    # Use train_test_split with stratification on anomaly labels to maintain proportions
    train_df, val_df = train_test_split(
        df,
        test_size=validation_split,
        random_state=random_seed,
        stratify=df['is_anomaly'],  # Maintain anomaly ratio in both sets
        shuffle=True
    )
    
    # Reset indices
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    
    return train_df, val_df


def main():
    """Main function to generate training data."""
    parser = argparse.ArgumentParser(description='Generate synthetic training and validation data')
    parser.add_argument('--config', default='../config/config.yaml', help='Path to config file')
    parser.add_argument('--train-output', default='training_data.csv', help='Training data output file')
    parser.add_argument('--val-output', default='validation_data.csv', help='Validation data output file')
    parser.add_argument('--validation-split', type=float, default=0.2, help='Validation split ratio (default: 0.2)')
    parser.add_argument('--random-seed', type=int, default=42, help='Random seed for reproducibility')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Extract parameters
    n_samples = config['data']['training_samples']
    n_features = config['data']['features']
    anomaly_rate = config['data']['anomaly_rate']
    feature_names = config['data']['feature_names']
    
    logger.info(f"Generating {n_samples} samples with {n_features} features")
    logger.info(f"Anomaly rate: {anomaly_rate * 100:.1f}%")
    
    # Generate normal data
    logger.info("Generating normal data patterns...")
    normal_data = generate_normal_data(n_samples, n_features, args.random_seed)
    
    # Validate normal data
    if np.any(np.isnan(normal_data)) or np.any(np.isinf(normal_data)):
        raise ValueError("Generated normal data contains NaN or infinite values")
    
    # Inject anomalies
    logger.info("Injecting anomalies...")
    anomalous_data, labels = inject_anomalies(normal_data, anomaly_rate, args.random_seed)
    
    # Validate anomalous data
    if np.any(np.isnan(anomalous_data)) or np.any(np.isinf(anomalous_data)):
        raise ValueError("Generated anomalous data contains NaN or infinite values")
    
    # Create DataFrame
    logger.info("Creating DataFrame...")
    df = create_dataframe(anomalous_data, labels, feature_names)
    
    # Split data into training and validation sets
    logger.info(f"Splitting data: {(1-args.validation_split)*100:.0f}% training, {args.validation_split*100:.0f}% validation")
    train_df, val_df = split_data_reproducibly(df, args.validation_split, args.random_seed)
    
    # Prepare data for SageMaker Random Cut Forest
    # Training data: only features, no labels (unsupervised learning)
    train_features = train_df[feature_names]
    
    # Test/validation data: anomaly label in first column, then features
    val_sagemaker = val_df[['is_anomaly'] + feature_names]
    
    # Save training data (SageMaker format: only features, no headers)
    train_output_path = os.path.join(os.path.dirname(__file__), args.train_output)
    train_features.to_csv(train_output_path, index=False, header=False)
    logger.info(f"Training data saved to {train_output_path}")
    
    # Save validation data (SageMaker format: label + features, no headers)
    val_output_path = os.path.join(os.path.dirname(__file__), args.val_output)
    val_sagemaker.to_csv(val_output_path, index=False, header=False)
    logger.info(f"Validation data saved to {val_output_path}")
    
    # Also save full datasets with headers for analysis (optional)
    train_full_path = os.path.join(os.path.dirname(__file__), 'training_data_full.csv')
    val_full_path = os.path.join(os.path.dirname(__file__), 'validation_data_full.csv')
    train_df.to_csv(train_full_path, index=False)
    val_df.to_csv(val_full_path, index=False)
    logger.info(f"Full training data (with headers) saved to {train_full_path}")
    logger.info(f"Full validation data (with headers) saved to {val_full_path}")
    
    # Print summary statistics
    logger.info("\nData Summary:")
    logger.info(f"Total samples: {len(df)}")
    logger.info(f"Training samples: {len(train_df)} ({len(train_df)/len(df)*100:.1f}%)")
    logger.info(f"Validation samples: {len(val_df)} ({len(val_df)/len(df)*100:.1f}%)")
    
    train_anomalies = (train_df['is_anomaly'] == 1).sum()
    val_anomalies = (val_df['is_anomaly'] == 1).sum()
    
    logger.info(f"Training anomalies: {train_anomalies} ({train_anomalies/len(train_df)*100:.2f}%)")
    logger.info(f"Validation anomalies: {val_anomalies} ({val_anomalies/len(val_df)*100:.2f}%)")
    
    # Print feature statistics for training data
    logger.info("\nTraining Data Feature Statistics:")
    for feature in feature_names:
        logger.info(f"{feature}: mean={train_df[feature].mean():.2f}, std={train_df[feature].std():.2f}")
    
    # Print feature statistics for validation data
    logger.info("\nValidation Data Feature Statistics:")
    for feature in feature_names:
        logger.info(f"{feature}: mean={val_df[feature].mean():.2f}, std={val_df[feature].std():.2f}")


if __name__ == "__main__":
    main()