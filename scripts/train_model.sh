#!/bin/bash
# Train the anomaly detection model on SageMaker

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Training Anomaly Detection Model${NC}"
echo ""

# Check if training data has been uploaded
if [ ! -f "data/.s3_train_location" ]; then
    echo -e "${RED}Error: Training data not found in S3${NC}"
    echo "Please run 'make upload-data' first"
    exit 1
fi

# Check if validation data has been uploaded
if [ ! -f "data/.s3_validation_location" ]; then
    echo -e "${RED}Error: Validation data not found in S3${NC}"
    echo "Please run 'make upload-data' first"
    exit 1
fi

S3_TRAIN_LOCATION=$(cat data/.s3_train_location)
S3_VAL_LOCATION=$(cat data/.s3_validation_location)
echo "Training data location: $S3_TRAIN_LOCATION"
echo "Validation data location: $S3_VAL_LOCATION"
echo ""

# Set up Python path for virtual environment
PYTHON=".venv/bin/python"

# Check if virtual environment exists
if [ ! -f "$PYTHON" ]; then
    echo -e "${RED}Error: Virtual environment not found${NC}"
    echo "Please run 'make install' to set up the virtual environment first"
    exit 1
fi

# Run training script
echo -e "${YELLOW}Starting SageMaker training job...${NC}"
$PYTHON model/train.py

# Check if training was successful
if [ -f "model/.training_info.json" ]; then
    echo -e "${GREEN}Training completed successfully!${NC}"
    
    # Display training info
    echo -e "\n${YELLOW}Training Job Info:${NC}"
    cat model/.training_info.json | python -m json.tool
    
    echo -e "\n${GREEN}Model training complete!${NC}"
    echo "Next step: Run 'make deploy-model' to create an endpoint"
else
    echo -e "${RED}Training failed!${NC}"
    exit 1
fi