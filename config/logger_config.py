"""
Centralized logging configuration for the real-time anomaly detection pipeline.
"""

import logging
import os
import yaml
from pathlib import Path


def load_config():
    """Load configuration from config.yaml file."""
    config_path = Path(__file__).parent / 'config.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables
    if 'aws' in config and 'account_id' in config['aws']:
        config['aws']['account_id'] = os.environ.get('AWS_ACCOUNT_ID', '')
    
    return config


def setup_logging(name=None):
    """
    Set up logging configuration using centralized settings.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
    
    Returns:
        Configured logger instance.
    """
    config = load_config()
    logging_config = config.get('logging', {})
    
    # Get log level from config, fallback to environment variable, then INFO
    log_level = (
        logging_config.get('level') or 
        os.environ.get('LOG_LEVEL', 'INFO')
    ).upper()
    
    # Get log format from config
    log_format = logging_config.get('format', '%(asctime)s - %(levelname)s - %(message)s')
    
    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format=log_format,
        force=True  # Override any existing configuration
    )
    
    # Return logger for the specified name or calling module
    return logging.getLogger(name)


def get_logger(name=None):
    """
    Get a configured logger instance.
    
    Args:
        name: Logger name. If None, uses the calling module's name.
    
    Returns:
        Configured logger instance.
    """
    return setup_logging(name)