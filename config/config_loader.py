"""
Centralized configuration loader with selective environment variable resolution.
"""

import os
import yaml
import re
import logging
from pathlib import Path
from typing import Dict, Any, List


def _resolve_env_vars_recursive(config: Any, required_keys: List[str], current_path: str = "") -> Any:
    """
    Recursively resolve environment variables in config values.
    
    Args:
        config: Configuration value to process
        required_keys: List of required config paths (e.g., ['aws.region', 'data.features'])
        current_path: Current path in the config structure
    
    Returns:
        Config with environment variables resolved
    """
    if isinstance(config, dict):
        return {
            k: _resolve_env_vars_recursive(
                v, 
                required_keys, 
                f"{current_path}.{k}" if current_path else k
            ) 
            for k, v in config.items()
        }
    elif isinstance(config, list):
        return [_resolve_env_vars_recursive(item, required_keys, current_path) for item in config]
    elif isinstance(config, str):
        # Replace ${VAR_NAME} with environment variable value
        pattern = r'\$\{([^}]+)\}'
        matches = re.findall(pattern, config)
        for var_name in matches:
            env_value = os.getenv(var_name)
            if env_value is None:
                # Only warn if this path is required
                if _is_path_required(current_path, required_keys):
                    logging.getLogger(__name__).warning(
                        f"Environment variable {var_name} not set for required key {current_path}, keeping placeholder"
                    )
                continue
            config = config.replace(f"${{{var_name}}}", env_value)
        return config
    else:
        return config


def _is_path_required(current_path: str, required_keys: List[str]) -> bool:
    """Check if a config path is required based on the required_keys list."""
    if not required_keys:
        return True  # If no specific keys requested, warn for all
    
    # Check if current path matches any required key or is a parent of one
    for req_key in required_keys:
        if current_path == req_key or req_key.startswith(f"{current_path}."):
            return True
    return False


def _validate_required_keys(config: Dict[str, Any], required_keys: List[str]) -> None:
    """Validate that required configuration keys are present."""
    if not required_keys:
        return
    
    missing_keys = []
    for key_path in required_keys:
        keys = key_path.split('.')
        current = config
        try:
            for key in keys:
                current = current[key]
        except (KeyError, TypeError):
            missing_keys.append(key_path)
    
    if missing_keys:
        raise ValueError(f"Required configuration keys missing: {', '.join(missing_keys)}")


def load_config(config_path: str = None, required_keys: List[str] = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file with selective environment variable resolution.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        required_keys: List of required config paths (e.g., ['aws.region', 'data.features']).
                      Only paths needed by the calling script to avoid unnecessary warnings.
    
    Returns:
        Configuration dictionary with environment variables resolved
    
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing or YAML is invalid
        RuntimeError: If config loading fails
    """
    # Default config path
    if config_path is None:
        config_path = Path(__file__).parent / 'config.yaml'
    
    # Convert to absolute path if relative
    if not os.path.isabs(config_path):
        config_path = os.path.join(os.path.dirname(__file__), config_path)
    
    # Check if config file exists
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Resolve environment variables only for required paths
        config = _resolve_env_vars_recursive(config, required_keys or [])
        
        # Validate required keys are present
        _validate_required_keys(config, required_keys or [])
        
        return config
        
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML configuration: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to load configuration: {e}")


def get_default_aws_region() -> str:
    """
    Get the default AWS region used consistently across the project.
    
    This function provides a single source of truth for the default AWS region
    to ensure consistency between all components (Lambda, Makefile, scripts).
    
    Returns:
        str: Default AWS region ('eu-west-1')
    """
    return 'eu-west-1'


def get_aws_region_from_config(config_path: str = None) -> str:
    """
    Get AWS region from configuration file with consistent fallback.
    
    Args:
        config_path: Path to config file. If None, uses default config/config.yaml
        
    Returns:
        str: AWS region from config or default fallback
    """
    try:
        config = load_config(config_path, required_keys=['aws.region'])
        return config['aws']['region']
    except Exception:
        return get_default_aws_region()