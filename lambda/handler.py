#!/usr/bin/env python3
"""
Lambda function handler for processing Kinesis stream data and detecting anomalies.
"""

import json
import base64
import os
import logging
import boto3
from typing import Dict, List, Any
import statistics
from datetime import datetime

# Set up logging - Lambda environment uses environment variables
logger = logging.getLogger()
log_level = os.environ.get('LOG_LEVEL', 'INFO').upper()
logger.setLevel(getattr(logging, log_level, logging.INFO))

# Initialize clients
sagemaker_runtime = boto3.client('sagemaker-runtime')
cloudwatch = boto3.client('cloudwatch')

# Environment variables
ENDPOINT_NAME = os.environ.get('ENDPOINT_NAME', 'anomaly-detection-endpoint')
REGION = os.environ.get('REGION', 'us-east-1')
ANOMALY_THRESHOLD = float(os.environ.get('ANOMALY_THRESHOLD', '3.0'))


def decode_kinesis_record(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode a Kinesis record from base64.
    
    Args:
        record: Kinesis record from the event
    
    Returns:
        Decoded data as dictionary
    """
    try:
        # Decode base64 data
        payload = base64.b64decode(record['data']).decode('utf-8')
        
        # Parse JSON
        data = json.loads(payload)
        
        # Add metadata
        data['_kinesis_metadata'] = {
            'eventID': record.get('eventID'),
            'eventSourceARN': record.get('eventSourceARN'),
            'approximateArrivalTimestamp': record.get('approximateArrivalTimestamp')
        }
        
        return data
    
    except Exception as e:
        logger.error(f"Failed to decode record: {e}")
        logger.error(f"Record: {record}")
        raise


def prepare_features(data: Dict[str, Any]) -> List[float]:
    """
    Extract and prepare features for the model.
    
    Args:
        data: Decoded data dictionary
    
    Returns:
        List of feature values
    """
    # Expected features in order
    feature_names = ['cpu_usage', 'memory_usage', 'network_throughput', 'disk_io']
    
    features = []
    for feature in feature_names:
        if feature not in data:
            logger.warning(f"Missing feature: {feature}, using default value 0")
            features.append(0.0)
        else:
            features.append(float(data[feature]))
    
    return features


def invoke_sagemaker_endpoint(features_batch: List[List[float]]) -> Dict[str, Any]:
    """
    Invoke SageMaker endpoint for anomaly detection.
    
    Args:
        features_batch: Batch of feature vectors
    
    Returns:
        Endpoint response
    """
    try:
        # Convert to CSV format for Random Cut Forest
        csv_data = '\n'.join([','.join(map(str, features)) for features in features_batch])
        
        response = sagemaker_runtime.invoke_endpoint(
            EndpointName=ENDPOINT_NAME,
            ContentType='text/csv',
            Body=csv_data
        )
        
        # Parse response
        result = json.loads(response['Body'].read().decode())
        return result
    
    except Exception as e:
        logger.error(f"Failed to invoke SageMaker endpoint: {e}")
        raise


def process_anomaly_scores(scores: List[float], records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process anomaly scores and determine which records are anomalies.
    
    Args:
        scores: Anomaly scores from the model
        records: Original records
    
    Returns:
        List of results with anomaly indicators
    """
    results = []
    anomaly_count = 0
    
    for score, record in zip(scores, records):
        is_anomaly = score > ANOMALY_THRESHOLD
        
        if is_anomaly:
            anomaly_count += 1
            logger.warning(f"Anomaly detected! Score: {score:.2f}, Data: {record}")
        
        result = {
            'timestamp': record.get('timestamp', datetime.utcnow().isoformat()),
            'anomaly_score': score,
            'is_anomaly': is_anomaly,
            'data': record,
            'threshold': ANOMALY_THRESHOLD
        }
        
        results.append(result)
    
    # Log summary
    logger.info(f"Processed {len(records)} records, found {anomaly_count} anomalies")
    
    return results


def publish_metrics(results: List[Dict[str, Any]]):
    """
    Publish metrics to CloudWatch.
    
    Args:
        results: Processing results
    """
    try:
        anomaly_count = sum(1 for r in results if r['is_anomaly'])
        total_count = len(results)
        anomaly_rate = anomaly_count / total_count if total_count > 0 else 0
        
        # Average anomaly score
        avg_score = statistics.mean([r['anomaly_score'] for r in results])
        
        metrics = [
            {
                'MetricName': 'RecordsProcessed',
                'Value': total_count,
                'Unit': 'Count'
            },
            {
                'MetricName': 'AnomaliesDetected',
                'Value': anomaly_count,
                'Unit': 'Count'
            },
            {
                'MetricName': 'AnomalyRate',
                'Value': anomaly_rate * 100,  # As percentage
                'Unit': 'Percent'
            },
            {
                'MetricName': 'AverageAnomalyScore',
                'Value': avg_score,
                'Unit': 'None'
            }
        ]
        
        cloudwatch.put_metric_data(
            Namespace='AnomalyDetection',
            MetricData=metrics
        )
        
        logger.info(f"Published metrics to CloudWatch: {anomaly_count}/{total_count} anomalies")
    
    except Exception as e:
        logger.error(f"Failed to publish metrics: {e}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """
    Main Lambda handler function.
    
    Args:
        event: Kinesis event with records
        context: Lambda context
    
    Returns:
        Response dictionary
    """
    logger.info(f"Processing {len(event['Records'])} Kinesis records")
    
    try:
        # Decode all records
        decoded_records = []
        features_batch = []
        
        for record in event['Records']:
            try:
                # Decode Kinesis record
                data = decode_kinesis_record(record['kinesis'])
                decoded_records.append(data)
                
                # Prepare features
                features = prepare_features(data)
                features_batch.append(features)
                
            except Exception as e:
                logger.error(f"Failed to process record: {e}")
                continue
        
        if not features_batch:
            logger.warning("No valid records to process")
            return {
                'statusCode': 200,
                'batchItemFailures': []
            }
        
        # Invoke SageMaker endpoint
        logger.info(f"Invoking SageMaker endpoint with {len(features_batch)} records")
        response = invoke_sagemaker_endpoint(features_batch)
        logger.info(f"SageMaker response: {response}")
        
        # Extract anomaly scores from SageMaker response
        # Random Cut Forest returns: {"scores": [{"score": 0.02}, {"score": 0.25}]}
        scores = []
        if 'scores' in response and isinstance(response['scores'], list):
            for score_obj in response['scores']:
                if isinstance(score_obj, dict) and 'score' in score_obj:
                    scores.append(float(score_obj['score']))
                elif isinstance(score_obj, (int, float)):
                    scores.append(float(score_obj))
                else:
                    logger.warning(f"Unexpected score format: {score_obj}")
                    scores.append(0.0)
        else:
            logger.error(f"Unexpected SageMaker response format: {response}")
            scores = [0.0] * len(decoded_records)  # Default to non-anomalous
        
        # Process results
        results = process_anomaly_scores(scores, decoded_records)
        
        # Publish metrics
        publish_metrics(results)
        
        # Log anomalies for monitoring
        for result in results:
            if result['is_anomaly']:
                logger.warning(json.dumps({
                    'event': 'anomaly_detected',
                    'score': result['anomaly_score'],
                    'data': result['data'],
                    'timestamp': result['timestamp']
                }))
        
        return {
            'statusCode': 200,
            'batchItemFailures': [],
            'summary': {
                'processed': len(results),
                'anomalies': sum(1 for r in results if r['is_anomaly'])
            }
        }
    
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        
        # Return all records as failures for retry
        return {
            'statusCode': 500,
            'batchItemFailures': [
                {'itemIdentifier': record['kinesis']['sequenceNumber']}
                for record in event['Records']
            ]
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'Records': [{
            'kinesis': {
                'data': base64.b64encode(json.dumps({
                    'timestamp': datetime.utcnow().isoformat(),
                    'cpu_usage': 75.5,
                    'memory_usage': 82.3,
                    'network_throughput': 45.7,
                    'disk_io': 60.2
                }).encode()).decode(),
                'sequenceNumber': '12345',
                'approximateArrivalTimestamp': 1234567890
            }
        }]
    }
    
    result = lambda_handler(test_event, None)
    print(json.dumps(result, indent=2))