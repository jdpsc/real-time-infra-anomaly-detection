#!/bin/bash
# Clean up all AWS resources

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
STACK_NAME="anomaly-detection-stack"
REGION=$(aws configure get region || echo "eu-west-1")

echo -e "${RED}WARNING: This will delete all resources!${NC}"
echo "This includes:"
echo "- CloudFormation stack and all resources"
echo "- S3 buckets and their contents"
echo "- SageMaker endpoints and models"
echo "- Lambda functions"
echo "- Kinesis streams"
echo ""

# Get confirmation
read -p "Are you absolutely sure? Type 'DELETE' to confirm: " -r
echo
if [[ ! $REPLY =~ ^DELETE$ ]]; then
    echo "Cleanup cancelled."
    exit 0
fi

echo -e "${YELLOW}Starting cleanup process...${NC}"

# Get AWS account ID for template bucket
ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text 2>/dev/null || echo "unknown")
PACKAGE_BUCKET="cf-templates-${ACCOUNT_ID}-${REGION}"

# Get stack outputs before deletion
echo "Getting resource information..."
PROJECT_NAME=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Parameters[?ParameterKey==`ProjectName`].ParameterValue' \
    --output text 2>/dev/null || echo "anomaly-detection")
ENVIRONMENT=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Parameters[?ParameterKey==`Environment`].ParameterValue' \
    --output text 2>/dev/null || echo "dev")

# Get endpoint name from config file
ENDPOINT_NAME=$(python3 -c "import yaml; import os; config = yaml.safe_load(open('config/config.yaml')); print(config['sagemaker']['endpoint_name'])" 2>/dev/null || echo "anomaly-detection-endpoint")

DATA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' \
    --output text 2>/dev/null || echo "")

MODEL_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`ModelBucketName`].OutputValue' \
    --output text 2>/dev/null || echo "")

LAMBDA_BUCKET=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query 'Stacks[0].Outputs[?OutputKey==`LambdaBucketName`].OutputValue' \
    --output text 2>/dev/null || echo "")

# Delete SageMaker endpoint if it exists
echo -e "${YELLOW}Deleting SageMaker endpoint...${NC}"
aws sagemaker delete-endpoint --endpoint-name $ENDPOINT_NAME 2>/dev/null || echo "No endpoint found"

# Delete SageMaker endpoint configurations
echo -e "${YELLOW}Deleting SageMaker endpoint configurations...${NC}"
if [ -f "model/.endpoint_info.json" ]; then
    ENDPOINT_CONFIG_NAME=$(python3 -c "import json; info=json.load(open('model/.endpoint_info.json')); print(info['endpoint_config_name'])" 2>/dev/null)
    if [ ! -z "$ENDPOINT_CONFIG_NAME" ]; then
        echo "Deleting endpoint config: $ENDPOINT_CONFIG_NAME"
        aws sagemaker delete-endpoint-config --endpoint-config-name $ENDPOINT_CONFIG_NAME 2>/dev/null || true
    fi
else
    echo -e "${RED}Warning: No endpoint info file found. Endpoint config may not be deleted.${NC}"
fi

# Delete SageMaker models
echo -e "${YELLOW}Deleting SageMaker models...${NC}"
if [ -f "model/.model_info.json" ]; then
    MODEL_NAME=$(python3 -c "import json; info=json.load(open('model/.model_info.json')); print(info['model_name'])" 2>/dev/null)
    if [ ! -z "$MODEL_NAME" ]; then
        echo "Deleting model: $MODEL_NAME"
        aws sagemaker delete-model --model-name $MODEL_NAME 2>/dev/null || true
    fi
else
    echo -e "${RED}Warning: No model info file found. Model may not be deleted.${NC}"
fi


# Empty S3 buckets before stack deletion
echo -e "${YELLOW}Emptying S3 buckets...${NC}"
for bucket in $DATA_BUCKET $MODEL_BUCKET $LAMBDA_BUCKET; do
    if [ ! -z "$bucket" ] && [ "$bucket" != "None" ]; then
        echo "Emptying bucket: $bucket"
        aws s3 rm s3://$bucket --recursive 2>/dev/null || true
    fi
done

# Empty and delete CloudFormation templates bucket
echo -e "${YELLOW}Cleaning up CloudFormation templates bucket...${NC}"
if [ "$ACCOUNT_ID" != "unknown" ]; then
    if aws s3 ls "s3://${PACKAGE_BUCKET}" >/dev/null 2>&1; then
        echo "Emptying templates bucket: $PACKAGE_BUCKET"
        aws s3 rm s3://$PACKAGE_BUCKET --recursive 2>/dev/null || true
        echo "Deleting templates bucket: $PACKAGE_BUCKET"
        aws s3 rb s3://$PACKAGE_BUCKET 2>/dev/null || true
    fi
fi

# Delete CloudFormation stack
echo -e "${YELLOW}Deleting CloudFormation stack...${NC}"
aws cloudformation delete-stack --stack-name $STACK_NAME --region $REGION

# Wait for stack deletion with timeout handling
echo "Waiting for stack deletion to complete..."

# Use aws cloudformation wait with built-in timeout
aws cloudformation wait stack-delete-complete --stack-name $STACK_NAME --region $REGION
WAIT_EXIT_CODE=$?

if [ $WAIT_EXIT_CODE -eq 0 ]; then
    echo -e "${GREEN}Stack deletion completed successfully${NC}"
elif [ $WAIT_EXIT_CODE -eq 255 ]; then
    echo -e "${YELLOW}Stack deletion timed out or failed. Checking final status...${NC}"
    STACK_STATUS=$(aws cloudformation describe-stacks --stack-name $STACK_NAME --query 'Stacks[0].StackStatus' --output text 2>/dev/null || echo "DELETE_COMPLETE")
    if [ "$STACK_STATUS" = "DELETE_COMPLETE" ] || [ "$STACK_STATUS" = "" ]; then
        echo -e "${GREEN}Stack deletion completed successfully${NC}"
    else
        echo -e "${RED}Warning: Stack may not have deleted completely. Status: $STACK_STATUS${NC}"
        echo "You may need to manually check and clean up remaining resources."
    fi
else
    echo -e "${RED}Unexpected error during stack deletion wait (exit code: $WAIT_EXIT_CODE)${NC}"
fi

# Delete CloudWatch log groups
echo -e "${YELLOW}Deleting CloudWatch log groups...${NC}"
# Get Lambda function name from config
LAMBDA_FUNCTION_NAME=$(python3 -c "import yaml; config = yaml.safe_load(open('config/config.yaml')); print(config['lambda']['function_name'])" 2>/dev/null || echo "anomaly-detector")
for log_group in $(aws logs describe-log-groups --log-group-name-prefix "/aws/lambda/${LAMBDA_FUNCTION_NAME}" --query 'logGroups[*].logGroupName' --output text 2>/dev/null); do
    echo "Deleting log group: $log_group"
    aws logs delete-log-group --log-group-name "$log_group" 2>/dev/null || true
done
for log_group in $(aws logs describe-log-groups --log-group-name-prefix "/aws/sagemaker/TrainingJobs" --query 'logGroups[*].logGroupName' --output text 2>/dev/null); do
    if [[ "$log_group" == *"anomaly-detection"* ]]; then
        echo "Deleting SageMaker log group: $log_group"
        aws logs delete-log-group --log-group-name "$log_group" 2>/dev/null || true
    fi
done

# Clean up local files
echo -e "${YELLOW}Cleaning up local files...${NC}"
rm -f data/.s3_train_location
rm -f data/.s3_validation_location
rm -f data/training_data.csv
rm -f data/training_data_full.csv
rm -f data/validation_data.csv
rm -f data/validation_data_full.csv
rm -f data/training_manifest.json
rm -f model/.training_info.json
rm -f model/.model_info.json
rm -f model/.endpoint_info.json
rm -f lambda/function.zip
rm -f infrastructure/packaged-main.yaml

echo -e "${GREEN}Cleanup complete!${NC}"
echo "All AWS resources have been deleted."