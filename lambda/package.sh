#!/bin/bash
# Package Lambda function with dependencies

set -e

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Packaging Lambda function${NC}"

# Clean up any existing package
rm -rf lambda/package/
rm -f lambda/function.zip

# Create package directory
mkdir -p lambda/package

# Install dependencies to package directory
echo -e "${YELLOW}Installing dependencies...${NC}"
pip install --target ./lambda/package -r lambda/requirements.txt

# Copy Lambda function
echo -e "${YELLOW}Copying Lambda function...${NC}"
cp lambda/handler.py lambda/package/

# Create deployment package
echo -e "${YELLOW}Creating deployment package...${NC}"
(cd lambda/package && zip -r ../function.zip . -x "*.pyc" -x "*__pycache__*")

# Clean up
rm -rf lambda/package/

# Display package info
echo -e "${GREEN}Lambda package created successfully!${NC}"
echo "Package: lambda/function.zip"
echo "Size: $(du -h lambda/function.zip | cut -f1)"

# Verify package contents
echo -e "\n${YELLOW}Package contents:${NC}"
unzip -l lambda/function.zip | head -20