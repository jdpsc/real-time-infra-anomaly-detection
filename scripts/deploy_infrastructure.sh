#!/bin/bash
# Deploy infrastructure using CloudFormation

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="anomaly-detection-stack"
REGION=$(aws configure get region || echo "eu-west-1")
TEMPLATE_DIR="infrastructure"

echo -e "${GREEN}Deploying Anomaly Detection Infrastructure${NC}"
echo "Stack Name: $STACK_NAME"
echo "Region: $REGION"
echo ""

# Check AWS CLI is installed
if ! command -v aws &> /dev/null; then
    echo -e "${RED}Error: AWS CLI is not installed${NC}"
    exit 1
fi

# Check if logged in
if ! aws sts get-caller-identity &> /dev/null; then
    echo -e "${RED}Error: Not logged in to AWS. Please run 'aws configure'${NC}"
    exit 1
fi

# Get AWS account ID
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
echo "AWS Account ID: $ACCOUNT_ID"

# Extract log level from config
LOG_LEVEL=$(python3 -c "import yaml; import os; config = yaml.safe_load(open('config/config.yaml')); print(config.get('logging', {}).get('level', 'INFO'))")
echo "Log Level: $LOG_LEVEL"
echo ""

# Package nested templates
echo -e "${YELLOW}Packaging CloudFormation templates...${NC}"
PACKAGE_BUCKET="cf-templates-${ACCOUNT_ID}-${REGION}"

# Create bucket if it doesn't exist
if aws s3 ls "s3://${PACKAGE_BUCKET}" >/dev/null 2>&1; then
    echo "Using existing bucket: ${PACKAGE_BUCKET}"
else
    echo "Creating bucket: ${PACKAGE_BUCKET}"
    aws s3 mb "s3://${PACKAGE_BUCKET}" --region $REGION
fi

# Upload nested templates
for template in s3.yaml kinesis.yaml sagemaker.yaml lambda.yaml; do
    echo "Uploading ${template}..."
    if ! aws s3 cp "${TEMPLATE_DIR}/${template}" "s3://${PACKAGE_BUCKET}/anomaly-detection/${template}"; then
        echo -e "${RED}Failed to upload ${template}${NC}"
        exit 1
    fi
done

# Package main template
echo -e "${YELLOW}Packaging main template...${NC}"
aws cloudformation package \
    --template-file "${TEMPLATE_DIR}/main.yaml" \
    --s3-bucket "${PACKAGE_BUCKET}" \
    --s3-prefix "anomaly-detection" \
    --output-template-file "${TEMPLATE_DIR}/packaged-main.yaml"

# Deploy stack
echo -e "${YELLOW}Deploying CloudFormation stack...${NC}"
if aws cloudformation deploy \
    --template-file "${TEMPLATE_DIR}/packaged-main.yaml" \
    --stack-name "${STACK_NAME}" \
    --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
    --parameter-overrides \
        ProjectName="anomaly-detection" \
        Environment="dev" \
        LogLevel="${LOG_LEVEL}" \
        KinesisShardCount=1 \
    --region "${REGION}"; then
    echo -e "${GREEN}Stack deployment completed successfully!${NC}"
    
    # Get outputs
    echo -e "\n${YELLOW}Stack Outputs:${NC}"
    aws cloudformation describe-stacks \
        --stack-name "${STACK_NAME}" \
        --region "${REGION}" \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
else
    echo -e "${RED}Stack deployment failed!${NC}"
    exit 1
fi

echo -e "\n${GREEN}Infrastructure deployment complete!${NC}"
echo "Next steps:"
echo "1. Run 'make generate-data' to create training data"
echo "2. Run 'make upload-data' to upload to S3"
echo "3. Run 'make train-model' to train the model"