#!/usr/bin/env python3
"""
Train anomaly detection model using SageMaker Random Cut Forest.
"""

import boto3
import sagemaker
from sagemaker import image_uris
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
import os
import argparse
from datetime import datetime
import json
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.logger_config import get_logger
from config.config_loader import load_config as load_config_new

# Set up logging
logger = get_logger(__name__)


def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load configuration from YAML file with only required keys for model training."""
    required_keys = [
        'sagemaker.training_instance_type',
        's3.model_bucket', 
        'model.num_samples_per_tree',
        'model.num_trees',
        'data.features',
        'sagemaker.model_name',
        'cloudformation.stack_name'
    ]
    return load_config_new(config_path, required_keys)


def get_s3_data_locations() -> tuple[str, str]:
    """Get S3 training and validation data locations from saved files."""
    train_s3_location_file = os.path.join(os.path.dirname(__file__), '../data/.s3_train_location')
    val_s3_location_file = os.path.join(os.path.dirname(__file__), '../data/.s3_validation_location')
    
    if not os.path.exists(train_s3_location_file):
        raise FileNotFoundError("Training S3 location file not found. Please run upload_to_s3.py first.")
    
    if not os.path.exists(val_s3_location_file):
        raise FileNotFoundError("Validation S3 location file not found. Please run upload_to_s3.py first.")
    
    with open(train_s3_location_file, 'r') as f:
        train_location = f.read().strip()
    
    with open(val_s3_location_file, 'r') as f:
        val_location = f.read().strip()
    
    return train_location, val_location


def get_sagemaker_role(config: dict) -> str:
    """Get SageMaker execution role from CloudFormation or IAM."""
    cf_client = boto3.client('cloudformation')
    
    try:
        stack_name = config['cloudformation']['stack_name']
        response = cf_client.describe_stacks(StackName=stack_name)
        outputs = response['Stacks'][0]['Outputs']
        
        for output in outputs:
            if output['OutputKey'] == 'SageMakerRoleArn':
                return output['OutputValue']
        
        raise ValueError("SageMakerRoleArn not found in stack outputs")
    
    except Exception as e:
        logger.warning(f"Could not get role from CloudFormation: {e}")
        
        # Try to get from SageMaker session
        session = sagemaker.Session()
        return session.get_execution_role()


def train_random_cut_forest(config: dict, s3_train_path: str, s3_val_path: str, role: str) -> str:
    """
    Train Random Cut Forest model for anomaly detection.
    
    Args:
        config: Configuration dictionary
        s3_train_path: S3 path to training data
        s3_val_path: S3 path to validation data
        role: SageMaker execution role ARN
    
    Returns:
        Name of the trained model
    """
    # Initialize SageMaker session
    session = sagemaker.Session()
    region = session.boto_region_name
    
    # Get container image for Random Cut Forest
    container = image_uris.retrieve(
        framework='randomcutforest',
        region=region,
        version='1'
    )
    
    logger.info(f"Using container: {container}")
    
    # Set up training job name
    job_name = f"anomaly-detection-rcf-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    # Get model bucket name from CloudFormation outputs
    cf_client = boto3.client('cloudformation')
    
    # Use configurable stack name if available
    stack_name = config.get('cloudformation', {}).get('stack_name', 'anomaly-detection-stack')
    response = cf_client.describe_stacks(StackName=stack_name)
    outputs = response['Stacks'][0]['Outputs']
    
    model_bucket_name = None
    for output in outputs:
        if output['OutputKey'] == 'ModelBucketName':
            model_bucket_name = output['OutputValue']
            break
    
    if not model_bucket_name:
        raise ValueError("ModelBucketName not found in stack outputs")
    
    logger.info(f"Using model bucket: {model_bucket_name}")
    
    # Configure the estimator
    rcf = Estimator(
        image_uri=container,
        role=role,
        instance_count=1,
        instance_type=config['sagemaker']['training_instance_type'],
        output_path=f"s3://{model_bucket_name}/models",
        sagemaker_session=session,
        base_job_name='anomaly-detection-rcf',
        tags=[
            {'Key': 'Project', 'Value': 'anomaly-detection'},
            {'Key': 'Algorithm', 'Value': 'RandomCutForest'},
            {'Key': 'Environment', 'Value': 'dev'}
        ]
    )
    
    # Set hyperparameters
    rcf.set_hyperparameters(
        num_samples_per_tree=config['model']['num_samples_per_tree'],
        num_trees=config['model']['num_trees'],
        feature_dim=config['data']['features']
    )
    
    logger.info("Starting training job...")
    logger.info(f"Training data: {s3_train_path}")
    logger.info(f"Validation data: {s3_val_path}")
    logger.info(f"Hyperparameters: {rcf.hyperparameters()}")
    
    # Create TrainingInput objects with correct distribution type for Random Cut Forest
    train_input = TrainingInput(
        s3_data=s3_train_path,
        content_type='text/csv;label_size=0',  # No labels for training (unsupervised)
        s3_data_type='S3Prefix',
        distribution='ShardedByS3Key'
    )
    
    # Random Cut Forest uses 'test' channel for validation data, not 'validation'
    # The test channel uses FullyReplicated distribution and expects labels in first column
    test_input = TrainingInput(
        s3_data=s3_val_path,
        content_type='text/csv;label_size=1',  # First column is anomaly label (0/1)
        s3_data_type='S3Prefix',
        distribution='FullyReplicated'
    )
    
    # Start training with both training and test (validation) data
    rcf.fit({'train': train_input, 'test': test_input}, job_name=job_name, wait=True)
    
    logger.info(f"Training completed! Job name: {job_name}")
    
    # Save training job info
    training_info = {
        'job_name': job_name,
        'model_data': rcf.model_data,
        'timestamp': datetime.now().isoformat(),
        'hyperparameters': rcf.hyperparameters()
    }
    
    info_path = os.path.join(os.path.dirname(__file__), '.training_info.json')
    with open(info_path, 'w') as f:
        json.dump(training_info, f, indent=2)
    
    return job_name


def create_model(job_name: str, config: dict, role: str) -> str:
    """
    Create a SageMaker model from the training job.
    
    Args:
        job_name: Training job name
        config: Configuration dictionary
        role: SageMaker execution role ARN
    
    Returns:
        Model name
    """
    sm_client = boto3.client('sagemaker')
    
    # Get training job details
    response = sm_client.describe_training_job(TrainingJobName=job_name)
    model_data = response['ModelArtifacts']['S3ModelArtifacts']
    container = response['AlgorithmSpecification']['TrainingImage']
    
    # Create model name
    model_name = f"{config['sagemaker']['model_name']}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    logger.info(f"Creating model: {model_name}")
    
    # Create model
    sm_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': container,
            'ModelDataUrl': model_data
        },
        ExecutionRoleArn=role,
        Tags=[
            {'Key': 'Project', 'Value': 'anomaly-detection'},
            {'Key': 'Algorithm', 'Value': 'RandomCutForest'}
        ]
    )
    
    logger.info(f"Model created successfully: {model_name}")
    
    # Save model info
    model_info = {
        'model_name': model_name,
        'model_data': model_data,
        'container': container,
        'timestamp': datetime.now().isoformat()
    }
    
    info_path = os.path.join(os.path.dirname(__file__), '.model_info.json')
    with open(info_path, 'w') as f:
        json.dump(model_info, f, indent=2)
    
    return model_name


def main():
    """Main function to train the model."""
    parser = argparse.ArgumentParser(description='Train anomaly detection model')
    parser.add_argument('--config', default='../config/config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Get S3 data locations
    try:
        s3_train_path, s3_val_path = get_s3_data_locations()
        logger.info(f"Training data location: {s3_train_path}")
        logger.info(f"Validation data location: {s3_val_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return
    
    # Get SageMaker role
    role = get_sagemaker_role(config)
    logger.info(f"Using SageMaker role: {role}")
    
    # Train model
    job_name = train_random_cut_forest(config, s3_train_path, s3_val_path, role)
    
    # Create model
    model_name = create_model(job_name, config, role)
    
    logger.info("\nTraining complete!")
    logger.info(f"Model name: {model_name}")
    logger.info("You can now run make deploy-model to create an endpoint")


if __name__ == "__main__":
    main()