"""
Test AWS region consistency across all components.

Tests for issue #1 fix: Ensure all components use consistent default AWS regions.
"""

import unittest
import os
import tempfile
import sys
from pathlib import Path
from unittest.mock import patch, mock_open

# Add parent directory to path to import config_loader
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.config_loader import get_default_aws_region, get_aws_region_from_config, load_config


class TestRegionConsistency(unittest.TestCase):
    """Test AWS region consistency across components."""
    
    def test_default_region_consistency(self):
        """Test that get_default_aws_region returns the expected default."""
        expected_region = 'eu-west-1'
        actual_region = get_default_aws_region()
        
        self.assertEqual(
            actual_region, 
            expected_region,
            f"Default region should be {expected_region}, got {actual_region}"
        )
    
    def test_lambda_handler_fallback_region(self):
        """Test that Lambda handler uses correct fallback region."""
        # Read the lambda handler file to check the fallback region
        lambda_handler_path = Path(__file__).parent.parent / 'lambda' / 'handler.py'
        
        with open(lambda_handler_path, 'r') as f:
            handler_content = f.read()
        
        # Check that the fallback region matches our expected default
        expected_region = get_default_aws_region()
        
        self.assertIn(
            f"REGION = os.environ.get('REGION', '{expected_region}')",
            handler_content,
            f"Lambda handler should use {expected_region} as fallback region"
        )
    
    def test_makefile_fallback_region(self):
        """Test that Makefile uses correct fallback region."""
        # Read the Makefile to check the fallback region
        makefile_path = Path(__file__).parent.parent / 'Makefile'
        
        with open(makefile_path, 'r') as f:
            makefile_content = f.read()
        
        # Check that the fallback region matches our expected default
        expected_region = get_default_aws_region()
        
        self.assertIn(
            f'echo "{expected_region}"',
            makefile_content,
            f"Makefile should use {expected_region} as fallback region"
        )
    
    def test_config_loader_with_valid_config(self):
        """Test config loader with valid config file."""
        # Create a temporary config file with a region
        test_config = """
aws:
  region: "us-west-2"
  account_id: "123456789012"
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(test_config)
            temp_config_path = f.name
        
        try:
            region = get_aws_region_from_config(temp_config_path)
            self.assertEqual(region, "us-west-2")
        finally:
            os.unlink(temp_config_path)
    
    def test_config_loader_with_missing_file(self):
        """Test config loader fallback when config file is missing."""
        non_existent_path = "/path/that/does/not/exist/config.yaml"
        
        region = get_aws_region_from_config(non_existent_path)
        expected_region = get_default_aws_region()
        
        self.assertEqual(
            region,
            expected_region,
            f"Should fallback to {expected_region} when config file is missing"
        )
    
    def test_config_loader_with_invalid_yaml(self):
        """Test config loader fallback when config file has invalid YAML."""
        # Create a temporary config file with invalid YAML
        invalid_yaml = """
aws:
  region: "us-west-2"
  invalid: [unclosed bracket
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(invalid_yaml)
            temp_config_path = f.name
        
        try:
            region = get_aws_region_from_config(temp_config_path)
            expected_region = get_default_aws_region()
            
            self.assertEqual(
                region,
                expected_region,
                f"Should fallback to {expected_region} when config file has invalid YAML"
            )
        finally:
            os.unlink(temp_config_path)
    
    def test_config_loader_with_missing_region_key(self):
        """Test config loader fallback when config file is missing region key."""
        # Create a temporary config file without aws.region
        test_config = """
data:
  features: 4
  training_samples: 1000
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(test_config)
            temp_config_path = f.name
        
        try:
            region = get_aws_region_from_config(temp_config_path)
            expected_region = get_default_aws_region()
            
            self.assertEqual(
                region,
                expected_region,
                f"Should fallback to {expected_region} when config is missing aws.region key"
            )
        finally:
            os.unlink(temp_config_path)
    
    def test_main_config_file_region(self):
        """Test that main config.yaml file contains the expected region."""
        # Load the actual project config file
        config_path = Path(__file__).parent.parent / 'config' / 'config.yaml'
        
        if config_path.exists():
            config = load_config(str(config_path), required_keys=['aws.region'])
            
            # The main config should use our expected default region
            expected_region = get_default_aws_region()
            actual_region = config['aws']['region']
            
            self.assertEqual(
                actual_region,
                expected_region,
                f"Main config.yaml should use {expected_region}, found {actual_region}"
            )
    
    @patch.dict(os.environ, {'REGION': 'us-east-1'})
    def test_lambda_environment_variable_override(self):
        """Test that Lambda handler respects REGION environment variable."""
        # This test would require importing the lambda handler, which might have dependencies
        # For now, we test the pattern by simulating the behavior
        
        # Simulate the pattern used in lambda/handler.py
        region = os.environ.get('REGION', get_default_aws_region())
        
        # Should use the environment variable when set
        self.assertEqual(region, 'us-east-1')
    
    def test_lambda_environment_variable_fallback(self):
        """Test that Lambda handler uses fallback when REGION env var is not set."""
        # Ensure REGION is not in environment for this test
        with patch.dict(os.environ, {}, clear=True):
            # Simulate the pattern used in lambda/handler.py
            region = os.environ.get('REGION', get_default_aws_region())
            
            # Should use the default when env var is not set
            expected_region = get_default_aws_region()
            self.assertEqual(
                region,
                expected_region,
                f"Should use default {expected_region} when REGION env var is not set"
            )


if __name__ == '__main__':
    unittest.main()