#!/usr/bin/env python3
"""
Streaming client to send test data to Kinesis and monitor results.
"""

import boto3
import json
import yaml
import time
import random
import logging
import argparse
import numpy as np
import os
from datetime import datetime
from typing import Dict, List
import signal
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from config.logger_config import get_logger
from config.config_loader import load_config as load_config_new

# Set up logging
logger = get_logger(__name__)

# Global flag for graceful shutdown
running = True


def signal_handler(sig, frame):
    """Handle shutdown signal."""
    global running
    logger.info("Received shutdown signal. Stopping...")
    running = False


def load_config(config_path: str = "../config/config.yaml") -> dict:
    """Load configuration from YAML file with only required keys for streaming client."""
    required_keys = [
        'aws.region'
    ]
    return load_config_new(config_path, required_keys)


def get_cloudformation_output(output_key: str, default_value: str = None) -> str:
    """Get output value from CloudFormation stack."""
    cf_client = boto3.client('cloudformation')
    
    try:
        response = cf_client.describe_stacks(StackName='anomaly-detection-stack')
        outputs = response['Stacks'][0]['Outputs']
        
        for output in outputs:
            if output['OutputKey'] == output_key:
                return output['OutputValue']
        
        raise ValueError(f"{output_key} not found in stack outputs")
    
    except Exception as e:
        logger.warning(f"Could not get {output_key} from CloudFormation: {e}")
        return default_value


def get_stream_name() -> str:
    """Get Kinesis stream name from CloudFormation stack."""
    return get_cloudformation_output('StreamName', 'anomaly-detection-dev-stream')


def get_log_group_name() -> str:
    """Get Lambda log group name from CloudFormation stack."""
    function_name = get_cloudformation_output('LambdaFunctionName')
    if function_name:
        return f'/aws/lambda/{function_name}'
    return '/aws/lambda/anomaly-detection-dev-anomaly-detector'


def generate_normal_record() -> Dict[str, float]:
    """Generate a normal data record."""
    # Base values with some correlation
    cpu = np.random.normal(50, 10)
    memory = cpu * 0.8 + np.random.normal(10, 5)
    network = np.random.normal(30, 8)
    disk = cpu * 0.6 + np.random.normal(5, 3)
    
    # Clip values to valid range
    return {
        'cpu_usage': np.clip(cpu, 0, 100),
        'memory_usage': np.clip(memory, 0, 100),
        'network_throughput': np.clip(network, 0, 100),
        'disk_io': np.clip(disk, 0, 100)
    }


def generate_anomaly_record() -> Dict[str, float]:
    """Generate an anomalous data record."""
    anomaly_type = random.choice(['spike', 'drop', 'uncorrelated', 'all_high'])
    
    if anomaly_type == 'spike':
        # Spike in one metric
        record = generate_normal_record()
        metric = random.choice(['cpu_usage', 'memory_usage', 'network_throughput', 'disk_io'])
        record[metric] = random.uniform(90, 100)
        
    elif anomaly_type == 'drop':
        # Drop in one metric
        record = generate_normal_record()
        metric = random.choice(['cpu_usage', 'memory_usage', 'network_throughput', 'disk_io'])
        record[metric] = random.uniform(0, 10)
        
    elif anomaly_type == 'uncorrelated':
        # Break normal correlations
        record = {
            'cpu_usage': random.uniform(0, 100),
            'memory_usage': random.uniform(0, 100),
            'network_throughput': random.uniform(0, 100),
            'disk_io': random.uniform(0, 100)
        }
        
    else:  # all_high
        # All metrics high
        record = {
            'cpu_usage': random.uniform(85, 100),
            'memory_usage': random.uniform(85, 100),
            'network_throughput': random.uniform(85, 100),
            'disk_io': random.uniform(85, 100)
        }
    
    return record


def send_to_kinesis(kinesis_client, stream_name: str, records: List[Dict]):
    """
    Send records to Kinesis stream.
    
    Args:
        kinesis_client: Boto3 Kinesis client
        stream_name: Name of the stream
        records: List of records to send
    """
    # Prepare records for Kinesis
    kinesis_records = []
    
    for record in records:
        # Add timestamp
        record['timestamp'] = datetime.utcnow().isoformat()
        
        # Convert to JSON
        data = json.dumps(record)
        
        # Create Kinesis record
        kinesis_records.append({
            'Data': data,
            'PartitionKey': str(random.randint(1, 1000))
        })
    
    # Send to Kinesis
    try:
        response = kinesis_client.put_records(
            Records=kinesis_records,
            StreamName=stream_name
        )
        
        # Check for failures
        if response['FailedRecordCount'] > 0:
            logger.warning(f"Failed to send {response['FailedRecordCount']} records")
        
        return response['FailedRecordCount'] == 0
    
    except Exception as e:
        logger.error(f"Failed to send records to Kinesis: {e}")
        return False


def monitor_cloudwatch_logs(logs_client, log_group: str = None, 
                          start_time: int = None) -> int:
    """
    Monitor CloudWatch logs for anomaly detections.
    
    Args:
        logs_client: Boto3 CloudWatch Logs client
        log_group: Log group name
        start_time: Start time for log query
    
    Returns:
        End time of last log event
    """
    if log_group is None:
        log_group = get_log_group_name()
    
    if start_time is None:
        start_time = int((time.time() - 60) * 1000)  # Last minute
    
    try:
        # Get log events
        response = logs_client.filter_log_events(
            logGroupName=log_group,
            startTime=start_time,
            filterPattern='anomaly_detected'
        )
        
        # Process events
        for event in response.get('events', []):
            message = event['message']
            if 'anomaly_detected' in message:
                try:
                    # Parse JSON log
                    log_data = json.loads(message.split('\t')[-1])
                    score = log_data.get('score', 'N/A')
                    data = log_data.get('data', {})
                    
                    logger.warning(f"🚨 ANOMALY DETECTED! Score: {score}")
                    logger.warning(f"   Data: CPU={data.get('cpu_usage', 'N/A'):.1f}%, "
                                 f"Memory={data.get('memory_usage', 'N/A'):.1f}%, "
                                 f"Network={data.get('network_throughput', 'N/A'):.1f}%, "
                                 f"Disk={data.get('disk_io', 'N/A'):.1f}%")
                except:
                    pass
        
        # Return last event time
        if response.get('events'):
            return response['events'][-1]['timestamp']
        else:
            return start_time
    
    except Exception as e:
        logger.debug(f"Error monitoring logs: {e}")
        return start_time


def stream_data(config: dict, duration: int = 300, anomaly_rate: float = 0.05):
    """
    Stream data to Kinesis for testing.
    
    Args:
        config: Configuration dictionary
        duration: Duration to stream in seconds
        anomaly_rate: Rate of anomalies to inject
    """
    # Initialize clients
    kinesis_client = boto3.client('kinesis', region_name=config['aws']['region'])
    logs_client = boto3.client('logs', region_name=config['aws']['region'])
    
    # Get stream name
    stream_name = get_stream_name()
    logger.info(f"Streaming to: {stream_name}")
    
    # Statistics
    total_sent = 0
    anomalies_sent = 0
    start_time = time.time()
    last_log_time = int(start_time * 1000)
    
    logger.info(f"Starting stream for {duration} seconds...")
    logger.info(f"Anomaly rate: {anomaly_rate * 100:.1f}%")
    logger.info("Press Ctrl+C to stop\n")
    
    # Main streaming loop
    while running and (time.time() - start_time) < duration:
        # Generate batch of records
        batch = []
        batch_anomalies = 0
        
        for _ in range(10):  # Send 10 records at a time
            if random.random() < anomaly_rate:
                record = generate_anomaly_record()
                batch_anomalies += 1
                logger.info(f"📤 Injecting anomaly: CPU={record['cpu_usage']:.1f}%, "
                           f"Memory={record['memory_usage']:.1f}%")
            else:
                record = generate_normal_record()
            
            batch.append(record)
        
        # Send to Kinesis
        if send_to_kinesis(kinesis_client, stream_name, batch):
            total_sent += len(batch)
            anomalies_sent += batch_anomalies
        
        # Monitor CloudWatch logs for detections
        last_log_time = monitor_cloudwatch_logs(logs_client, start_time=last_log_time)
        
        # Display progress
        elapsed = time.time() - start_time
        logger.info(f"Progress: {elapsed:.0f}s / {duration}s | "
                   f"Sent: {total_sent} records ({anomalies_sent} anomalies)")
        
        # Wait before next batch
        time.sleep(1)
    
    # Final statistics
    logger.info("\n" + "="*50)
    logger.info("Streaming Complete!")
    logger.info("="*50)
    logger.info(f"Total records sent: {total_sent}")
    logger.info(f"Anomalies injected: {anomalies_sent}")
    logger.info(f"Actual anomaly rate: {anomalies_sent/total_sent*100:.2f}%")
    logger.info(f"Duration: {time.time() - start_time:.1f} seconds")
    
    # Wait a bit for final logs
    logger.info("\nWaiting for final log processing... Some logs may still be processing.")
    time.sleep(30)
    monitor_cloudwatch_logs(logs_client, start_time=last_log_time)


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Stream test data to Kinesis')
    parser.add_argument('--config', default='../config/config.yaml', help='Path to config file')
    parser.add_argument('--duration', type=int, default=30, help='Duration in seconds')
    parser.add_argument('--anomaly-rate', type=float, default=0.10, help='Anomaly injection rate')
    args = parser.parse_args()
    
    # Set up signal handler
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load configuration
    config = load_config(args.config)
    
    # Start streaming
    stream_data(config, args.duration, args.anomaly_rate)


if __name__ == "__main__":
    main()