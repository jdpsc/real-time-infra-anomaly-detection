# Makefile for Real-Time Anomaly Detection Pipeline

# Variables
VENV := .venv
PYTHON := $(VENV)/bin/python
UV := uv
AWS := aws
STACK_NAME := anomaly-detection-stack
REGION := $(shell $(AWS) configure get region || echo "us-east-1")
AWS_ACCOUNT_ID := $(shell $(AWS) sts get-caller-identity --query Account --output text 2>/dev/null || echo "")
S3_BUCKET = $(shell $(AWS) cloudformation describe-stacks --stack-name $(STACK_NAME) --query 'Stacks[0].Outputs[?OutputKey==`DataBucketName`].OutputValue' --output text 2>/dev/null || echo "anomaly-detection-data-bucket")
LAMBDA_BUCKET = $(shell $(AWS) cloudformation describe-stacks --stack-name $(STACK_NAME) --query 'Stacks[0].Outputs[?OutputKey==`LambdaBucketName`].OutputValue' --output text 2>/dev/null || echo "anomaly-detection-lambda-bucket")

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help install setup-venv setup-permissions deploy-infra generate-data upload-data train-model deploy-model deploy-lambda test-stream cleanup-all clean-venv

help: ## Show this help message
	@echo "$(GREEN)Real-Time Anomaly Detection Pipeline$(NC)"
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-20s$(NC) %s\n", $$1, $$2}'

$(VENV)/pyvenv.cfg:
	@echo "$(GREEN)Creating virtual environment with uv...$(NC)"
	$(UV) venv $(VENV)

setup-venv: check-uv $(VENV)/pyvenv.cfg setup-permissions ## Create virtual environment

setup-permissions: ## Set executable permissions on scripts
	@echo "$(GREEN)Setting executable permissions on scripts...$(NC)"
	chmod +x scripts/*.sh
	chmod +x lambda/package.sh

install: setup-venv setup-permissions ## Install all dependencies using uv, set up permissions, and create virtual environment
	@echo "$(GREEN)Installing dependencies with uv...$(NC)"
	$(UV) pip install --python $(PYTHON) -r requirements.txt
	$(UV) pip install --python $(PYTHON) -r data/requirements.txt
	$(UV) pip install --python $(PYTHON) -r model/requirements.txt
	$(UV) pip install --python $(PYTHON) -r client/requirements.txt
	@echo "$(GREEN)All dependencies installed successfully$(NC)"

clean-venv: ## Remove virtual environment
	@echo "$(YELLOW)Removing virtual environment...$(NC)"
	rm -rf $(VENV)

deploy-infra: ## Deploy AWS infrastructure using CloudFormation
	@echo "$(GREEN)Deploying infrastructure...$(NC)"
	./scripts/deploy_infrastructure.sh

check-infra: ## Check infrastructure deployment status
	@echo "$(GREEN)Checking infrastructure status...$(NC)"
	$(AWS) cloudformation describe-stacks --stack-name $(STACK_NAME) --region $(REGION)

generate-data: ## Generate synthetic training data
	@echo "$(GREEN)Generating training data...$(NC)"
	$(PYTHON) data/generate_training_data.py

upload-data: ## Upload training data to S3
	@echo "$(GREEN)Uploading data to S3...$(NC)"
	AWS_ACCOUNT_ID=$(AWS_ACCOUNT_ID) $(PYTHON) data/upload_to_s3.py

train-model: ## Train the anomaly detection model
	@echo "$(GREEN)Training model on SageMaker...$(NC)"
	./scripts/train_model.sh

deploy-model: ## Deploy the trained model
	@echo "$(GREEN)Deploying model endpoint...$(NC)"
	$(PYTHON) model/deploy.py

package-lambda: ## Package Lambda function
	@echo "$(GREEN)Packaging Lambda function...$(NC)"
	./lambda/package.sh

deploy-lambda: package-lambda ## Deploy Lambda function
	@echo "$(GREEN)Deploying Lambda function...$(NC)"
	$(AWS) s3 cp lambda/function.zip s3://$(LAMBDA_BUCKET)/lambda/function.zip
	$(AWS) cloudformation update-stack --stack-name $(STACK_NAME) --use-previous-template --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM

test-stream: ## Test the streaming pipeline
	@echo "$(GREEN)Testing streaming pipeline...$(NC)"
	$(PYTHON) client/stream_client.py

cleanup-infra: ## Clean up all AWS resources
	@echo "$(RED)Cleaning up resources...$(NC)"
	@read -p "Are you sure you want to delete all resources? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		./scripts/cleanup.sh; \
	fi

cleanup-all: cleanup-infra clean-venv ## Clean up AWS resources and virtual environment

# Deployment shortcuts
quick-deploy: install deploy-infra generate-data upload-data train-model deploy-model deploy-lambda ## Full deployment pipeline
	@echo "$(GREEN)Deployment complete!$(NC)"

validate-templates: ## Validate CloudFormation templates
	@echo "$(GREEN)Validating CloudFormation templates...$(NC)"
	$(AWS) cloudformation validate-template --template-body file://infrastructure/main.yaml
	$(AWS) cloudformation validate-template --template-body file://infrastructure/s3.yaml
	$(AWS) cloudformation validate-template --template-body file://infrastructure/kinesis.yaml
	$(AWS) cloudformation validate-template --template-body file://infrastructure/lambda.yaml
	$(AWS) cloudformation validate-template --template-body file://infrastructure/sagemaker.yaml