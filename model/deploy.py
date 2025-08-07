#!/usr/bin/env python3
"""
Deploy trained model as SageMaker endpoint.
"""

import boto3
import os
import argparse
import json
from datetime import datetime
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.logger_config import get_logger
from config.config_loader import load_config as load_config_new

# Set up logging
logger = get_logger(__name__)


def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load configuration from YAML file with only required keys for model deployment."""
    required_keys = [
        'sagemaker.endpoint_name',
        'sagemaker.inference_instance_type'
    ]
    return load_config_new(config_path, required_keys)


def get_latest_model_name() -> str:
    """Get the latest model name from saved info."""
    info_path = os.path.join(os.path.dirname(__file__), '.model_info.json')
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            return info['model_name']
    else:
        raise FileNotFoundError("Model info file not found. Please run make train-model first.")


def create_endpoint_config(model_name: str, config: dict) -> str:
    """
    Create endpoint configuration.
    
    Args:
        model_name: Name of the model
        config: Configuration dictionary
    
    Returns:
        Endpoint configuration name
    """
    sm_client = boto3.client('sagemaker')
    
    # Create endpoint config name
    endpoint_config_name = f"{config['sagemaker']['endpoint_name']}-config-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    logger.info(f"Creating endpoint configuration: {endpoint_config_name}")
    
    # Create endpoint configuration
    sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'primary',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': config['sagemaker']['inference_instance_type'],
                'InitialVariantWeight': 1.0
            }
        ],
        Tags=[
            {'Key': 'Project', 'Value': 'anomaly-detection'},
            {'Key': 'Model', 'Value': model_name}
        ]
    )
    
    logger.info(f"Endpoint configuration created: {endpoint_config_name}")
    return endpoint_config_name


def create_or_update_endpoint(endpoint_config_name: str, config: dict):
    """
    Create or update SageMaker endpoint.
    
    Args:
        endpoint_config_name: Name of the endpoint configuration
        config: Configuration dictionary
    """
    sm_client = boto3.client('sagemaker')
    endpoint_name = config['sagemaker']['endpoint_name']
    
    # Check if endpoint exists
    try:
        sm_client.describe_endpoint(EndpointName=endpoint_name)
        endpoint_exists = True
        logger.info(f"Endpoint {endpoint_name} already exists. Updating...")
    except sm_client.exceptions.ClientError:
        endpoint_exists = False
        logger.info(f"Creating new endpoint: {endpoint_name}")
    
    if endpoint_exists:
        # Update existing endpoint
        sm_client.update_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name
        )
    else:
        # Create new endpoint
        sm_client.create_endpoint(
            EndpointName=endpoint_name,
            EndpointConfigName=endpoint_config_name,
            Tags=[
                {'Key': 'Project', 'Value': 'anomaly-detection'}
            ]
        )
    
    # Wait for endpoint to be ready
    logger.info("Waiting for endpoint to be ready...")
    waiter = sm_client.get_waiter('endpoint_in_service')
    waiter.wait(
        EndpointName=endpoint_name,
        WaiterConfig={
            'Delay': 30,
            'MaxAttempts': 30  # Wait up to 15 minutes
        }
    )
    
    logger.info(f"Endpoint {endpoint_name} is ready!")


def test_endpoint(endpoint_name: str):
    """
    Test the deployed endpoint with sample data.
    
    Args:
        endpoint_name: Name of the endpoint
    """
    runtime_client = boto3.client('sagemaker-runtime')
    
    # Create test data (4 features)
    test_data = "50.0,60.0,30.0,40.0\n75.0,80.0,45.0,65.0\n95.0,90.0,85.0,88.0"
    
    logger.info("Testing endpoint with sample data...")
    
    try:
        response = runtime_client.invoke_endpoint(
            EndpointName=endpoint_name,
            ContentType='text/csv',
            Body=test_data
        )
        
        result = json.loads(response['Body'].read().decode())
        logger.info(f"Test successful! Response: {result}")
        
        # Check if response has expected structure
        if 'scores' in result:
            logger.info(f"Anomaly scores: {result['scores']}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
        raise


def save_endpoint_info(endpoint_name: str, endpoint_config_name: str, model_name: str):
    """Save endpoint information for later use."""
    endpoint_info = {
        'endpoint_name': endpoint_name,
        'endpoint_config_name': endpoint_config_name,
        'model_name': model_name,
        'timestamp': datetime.now().isoformat()
    }
    
    info_path = os.path.join(os.path.dirname(__file__), '.endpoint_info.json')
    with open(info_path, 'w') as f:
        json.dump(endpoint_info, f, indent=2)


def main():
    """Main function to deploy the model."""
    parser = argparse.ArgumentParser(description='Deploy anomaly detection model')
    parser.add_argument('--config', default='../config/config.yaml', help='Path to config file')
    parser.add_argument('--model-name', help='Specific model name to deploy (optional)')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get model name
    if args.model_name:
        model_name = args.model_name
    else:
        try:
            model_name = get_latest_model_name()
            logger.info(f"Using latest model: {model_name}")
        except FileNotFoundError as e:
            logger.error(str(e))
            return
    
    # Create endpoint configuration
    endpoint_config_name = create_endpoint_config(model_name, config)
    
    # Create or update endpoint
    create_or_update_endpoint(endpoint_config_name, config)
    
    # Test endpoint
    endpoint_name = config['sagemaker']['endpoint_name']
    test_endpoint(endpoint_name)
    
    # Save endpoint info
    save_endpoint_info(endpoint_name, endpoint_config_name, model_name)
    
    logger.info("\nDeployment complete!")
    logger.info(f"Endpoint name: {endpoint_name}")
    logger.info("The endpoint is ready to receive requests")


if __name__ == "__main__":
    main()