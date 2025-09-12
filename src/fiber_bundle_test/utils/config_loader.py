"""Configuration loading utilities."""

import json
import yaml
from typing import Dict, Any
from pathlib import Path


class ConfigLoader:
    """Load and manage configuration settings."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load configuration from JSON or YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                config = json.load(f)
            else:
                raise ValueError("Unsupported config format. Use .json, .yml, or .yaml")
        
        return config
    
    @staticmethod
    def save_config(config: Dict[str, Any], config_path: str):
        """
        Save configuration to file.
        
        Args:
            config: Configuration dictionary
            config_path: Path to save configuration
        """
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yml', '.yaml']:
                yaml.dump(config, f, default_flow_style=False, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(config, f, indent=2)
            else:
                raise ValueError("Unsupported config format. Use .json, .yml, or .yaml")
    
    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        Get default configuration for fiber bundle tests.
        
        Returns:
            Default configuration dictionary
        """
        return {
            'test_parameters': {
                'r_min': 0.01,
                'r_max': 20.0,
                'n_r': 200,
                'alpha': 0.05,
                'window_size': 10
            },
            'embedding_parameters': {
                'model_name': 'bert-base-uncased',
                'max_length': 512,
                'batch_size': 32
            },
            'visualization': {
                'figsize': [12, 8],
                'dpi': 300,
                'style': 'seaborn-v0_8'
            },
            'output': {
                'save_embeddings': True,
                'save_results': True,
                'save_plots': True,
                'output_dir': './output',
                'plots_dir': './plots'
            },
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            }
        }
    
    @staticmethod
    def create_default_config_file(config_path: str = './config/default_config.yaml'):
        """
        Create a default configuration file.
        
        Args:
            config_path: Path where to save the default configuration
        """
        default_config = ConfigLoader.get_default_config()
        ConfigLoader.save_config(default_config, config_path)
        print(f"Default configuration saved to: {config_path}")
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any], 
                     override_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge two configuration dictionaries.
        
        Args:
            base_config: Base configuration
            override_config: Configuration values to override
            
        Returns:
            Merged configuration dictionary
        """
        merged = base_config.copy()
        
        for key, value in override_config.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = ConfigLoader.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
