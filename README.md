# Real-Time Anomaly Detection Pipeline

A production-ready serverless system that monitors streaming infrastructure metrics in real-time and automatically detects anomalous behavior using machine learning. Built on AWS, it processes thousands of records per second and provides sub-second anomaly detection for IT operations and monitoring.

## What This System Does

This project demonstrates a complete anomaly detection pipeline for **infrastructure monitoring**. It simulates a real-world scenario where you need to monitor server health metrics and automatically detect when something unusual happens - like performance degradation, security incidents, or hardware failures.

### Key Capabilities
- **Real-time Processing**: Analyzes streaming data as it arrives with sub-second latency
- **Machine Learning**: Uses Amazon SageMaker's Random Cut Forest algorithm to learn normal patterns and detect anomalies
- **Scalable**: Automatically handles varying data loads from hundreds to thousands of records per second
- **Production-Ready**: Includes comprehensive logging, monitoring, error handling, and automated deployment

## Data Use Case: Infrastructure Health Monitoring

The system monitors four critical server metrics that typically correlate in healthy systems:

### Monitored Metrics
1. **CPU Usage (%)** - Processor utilization levels
2. **Memory Usage (%)** - RAM consumption 
3. **Network Throughput (Mbps)** - Data transfer rates
4. **Disk I/O (IOPS)** - Storage read/write operations

### What Makes This Realistic
The synthetic data generator creates realistic infrastructure patterns:
- **Daily cycles**: Higher usage during business hours, lower at night
- **Weekly patterns**: Reduced activity on weekends
- **Natural correlations**: Memory usage typically follows CPU usage patterns
- **Realistic noise**: Normal variations you'd see in real systems

### Anomalies the System Detects

**Performance Issues**:
- CPU spikes to 95%+ (potential runaway processes or attacks)
- Memory consumption exceeding normal patterns (memory leaks)
- Sudden drops in performance metrics (hardware failures)

**Security Incidents**:
- Unusual network throughput patterns (data exfiltration, DDoS attacks)
- Broken correlations between metrics (compromised systems behaving abnormally)

**Operational Problems**:
- Disk I/O anomalies indicating storage issues
- Unusual patterns outside normal operational windows
- System instability causing erratic metric relationships

### Real-World Applications
This same architecture can be applied to:
- **DevOps**: Monitor application performance and infrastructure health
- **Security**: Detect potential breaches through unusual resource usage
- **IoT**: Monitor sensor data from industrial equipment or smart devices  
- **Financial**: Detect fraudulent transaction patterns
- **E-commerce**: Identify unusual user behavior or system performance issues

## Technologies Used

**AWS Services**:
- **Kinesis Data Streams**: Ingests streaming data
- **Lambda**: Processes stream records and calls ML model
- **SageMaker**: Hosts Random Cut Forest anomaly detection model
- **S3**: Stores training data and model artifacts
- **CloudFormation**: Infrastructure as Code deployment

## Prerequisites

- AWS CLI configured with appropriate credentials
- Python 3.8+
- [uv](https://docs.astral.sh/uv/) - Fast Python package installer and resolver
- Docker (for SageMaker local testing, optional)
- Make command available

## Quick Start

```bash
# 1. Install dependencies
make install

# 2. Deploy infrastructure
make deploy-infra

# 3. Generate and upload training data
make generate-data
make upload-data

# 4. Train the model
make train-model

# 5. Deploy the model endpoint
make deploy-model

# 6. Deploy Lambda function
make deploy-lambda

# 7. Test the pipeline
make test-stream
```

## Architecture

```
Local Client → Kinesis Data Stream → Lambda Function → SageMaker Endpoint
                                            ↓
                                     CloudWatch Logs
```


### Data Flow

1. **Training Phase**:
   ```
   generate_training_data.py → CSV file → S3 → SageMaker Training Job
   ```

2. **Inference Phase**:
   ```
   stream_client.py → Kinesis → Lambda → SageMaker Endpoint → CloudWatch
   ```

## Configuration

Edit `config/config.yaml` to customize:
- AWS region
- S3 bucket names
- Model parameters
- Stream settings

## Monitoring

- **Lambda Logs**: Check CloudWatch Logs group
- **Model Metrics**: View in SageMaker console
- **Stream Metrics**: Monitor in Kinesis console

## Cleanup

To remove all resources:

```bash
make cleanup-all
```