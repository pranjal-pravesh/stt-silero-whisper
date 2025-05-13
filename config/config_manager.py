"""
Configuration manager for the application.
Loads configuration from YAML files and provides access to settings.
"""

import os
import yaml
from typing import Any, Dict, Optional, Union


class ConfigManager:
    """
    Manages application configuration loaded from YAML files.
    Provides access to config values via dot notation or dictionary-style access.
    """
    
    _instance = None
    
    def __new__(cls, config_path=None):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path=None):
        """
        Initialize the config manager with a path to the config file.
        
        Args:
            config_path: Path to the YAML config file
        """
        if self._initialized:
            return
            
        self.config_path = config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config",
            "config.yaml"
        )
        self.config = self._load_config()
        self._initialized = True
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from the YAML file.
        
        Returns:
            Dict containing configuration values
        """
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except (IOError, yaml.YAMLError) as e:
            print(f"Error loading configuration: {e}")
            return {}
    
    def reload(self):
        """Reload configuration from file."""
        self.config = self._load_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get a configuration value by its key path.
        
        Args:
            key_path: Dot-separated path to the config value (e.g. 'audio.sample_rate')
            default: Default value to return if key doesn't exist
            
        Returns:
            Configuration value or default
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if not isinstance(value, dict) or key not in value:
                return default
            value = value[key]
            
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Enable dictionary-style access to config values."""
        return self.get(key)
    
    def set(self, key_path: str, value: Any) -> None:
        """
        Set a configuration value.
        
        Args:
            key_path: Dot-separated path to the config value
            value: Value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        # Navigate to the last dict that should contain the value
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
            
        # Set the value
        config[keys[-1]] = value


# Global instance accessor
_config_instance = None


def get_config(config_path: Optional[str] = None) -> ConfigManager:
    """
    Get the singleton ConfigManager instance.
    
    Args:
        config_path: Optional path to config file (used only on first call)
        
    Returns:
        ConfigManager instance
    """
    global _config_instance
    if _config_instance is None:
        _config_instance = ConfigManager(config_path)
    return _config_instance 