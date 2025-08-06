#!/usr/bin/env python3
"""
Upload training data to S3 bucket.
"""

import boto3
import os
import argparse
from datetime import datetime
from typing import Dict, Any
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.logger_config import get_logger
from config.config_loader import load_config as load_config_new

# Set up logging
logger = get_logger(__name__)




def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load configuration from YAML file with only required keys for S3 upload."""
    required_keys = [
        'aws.region',
        's3.data_bucket'
    ]
    return load_config_new(config_path, required_keys)


def get_bucket_name(config: Dict[str, Any]) -> str:
    """Get the S3 bucket name from CloudFormation stack outputs or config."""
    
    cf_client = boto3.client('cloudformation')
    
    # Use configurable stack name if available
    stack_name = config.get('cloudformation', {}).get('stack_name', 'anomaly-detection-stack')
    response = cf_client.describe_stacks(StackName=stack_name)
    outputs = response['Stacks'][0]['Outputs']
    
    for output in outputs:
        if output['OutputKey'] == 'DataBucketName':
            return output['OutputValue']
    
    raise ValueError("DataBucketName not found in stack outputs")


def upload_file_to_s3(file_path: str, bucket_name: str, s3_key: str):
    """
    Upload a file to S3.
    
    Args:
        file_path: Local path to the file
        bucket_name: S3 bucket name
        s3_key: S3 object key
    """
    s3_client = boto3.client('s3')
    
    try:
        # Check if bucket exists
        try:
            s3_client.head_bucket(Bucket=bucket_name)
        except Exception as e:
            logger.error(f"Bucket {bucket_name} not accessible: {e}")
            raise RuntimeError(f"Cannot access S3 bucket {bucket_name}")
        
        # Get file size for logging
        file_size = os.path.getsize(file_path) / (1024 * 1024)  # Convert to MB
        logger.info(f"Uploading file: {file_path} ({file_size:.2f} MB)")
        
        # Upload file
        with open(file_path, 'rb') as f:
            s3_client.put_object(
                Bucket=bucket_name,
                Key=s3_key,
                Body=f,
                ContentType='text/csv'
            )
        
        logger.info(f"Successfully uploaded {file_path} to s3://{bucket_name}/{s3_key}")
        
    except Exception as e:
        logger.error(f"Failed to upload file: {e}")
        raise


def create_manifest_file(bucket_name: str, train_s3_key: str, val_s3_key: str, output_path: str):
    """
    Create a manifest file for SageMaker training with both training and validation data.
    
    Args:
        bucket_name: S3 bucket name
        train_s3_key: S3 key of the training data
        val_s3_key: S3 key of the validation data
        output_path: Local path to save the manifest
    """
    manifest_content = {
        "train": f"s3://{bucket_name}/{train_s3_key}",
        "validation": f"s3://{bucket_name}/{val_s3_key}",
        "ContentType": "text/csv"
    }
    
    with open(output_path, 'w') as f:
        import json
        json.dump(manifest_content, f, indent=2)
    
    logger.info(f"Created manifest file at {output_path}")


def main():
    """Main function to upload data to S3."""
    parser = argparse.ArgumentParser(description='Upload training and validation data to S3')
    parser.add_argument('--config', default='../config/config.yaml', help='Path to config file')
    parser.add_argument('--train-input', default='training_data.csv', help='Training data file to upload')
    parser.add_argument('--val-input', default='validation_data.csv', help='Validation data file to upload')
    parser.add_argument('--prefix', default='training-data', help='S3 prefix for the files')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    region = config['aws']['region']
    
    # Set AWS region
    os.environ['AWS_DEFAULT_REGION'] = region
    
    # Get bucket name
    bucket_name = get_bucket_name(config)
    logger.info(f"Using S3 bucket: {bucket_name}")
    
    # Check if training file exists
    train_file_path = os.path.join(os.path.dirname(__file__), args.train_input)
    if not os.path.exists(train_file_path):
        logger.error(f"Training file not found: {train_file_path}")
        logger.info("Please run generate_training_data.py first")
        return
    
    # Check if validation file exists
    val_file_path = os.path.join(os.path.dirname(__file__), args.val_input)
    if not os.path.exists(val_file_path):
        logger.error(f"Validation file not found: {val_file_path}")
        logger.info("Please run generate_training_data.py first")
        return
    
    # Create S3 keys with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_s3_key = f"{args.prefix}/{timestamp}/training_data.csv"
    val_s3_key = f"{args.prefix}/{timestamp}/validation_data.csv"
    
    # Upload training file
    logger.info(f"Uploading {args.train_input} to S3...")
    upload_file_to_s3(train_file_path, bucket_name, train_s3_key)
    
    # Upload validation file
    logger.info(f"Uploading {args.val_input} to S3...")
    upload_file_to_s3(val_file_path, bucket_name, val_s3_key)
    
    # Create manifest file for SageMaker
    manifest_path = os.path.join(os.path.dirname(__file__), 'training_manifest.json')
    create_manifest_file(bucket_name, train_s3_key, val_s3_key, manifest_path)
    
    # Save S3 locations for later use
    s3_location_file = os.path.join(os.path.dirname(__file__), '.s3_train_location')
    with open(s3_location_file, 'w') as f:
        f.write(f"s3://{bucket_name}/{train_s3_key}")
    
    # Save validation S3 location
    val_s3_location_file = os.path.join(os.path.dirname(__file__), '.s3_validation_location')
    with open(val_s3_location_file, 'w') as f:
        f.write(f"s3://{bucket_name}/{val_s3_key}")
    
    logger.info("\nUpload complete!")
    logger.info(f"Training data location: s3://{bucket_name}/{train_s3_key}")
    logger.info(f"Validation data location: s3://{bucket_name}/{val_s3_key}")
    logger.info("You can now run make train-model to start training the model.")


if __name__ == "__main__":
    main()