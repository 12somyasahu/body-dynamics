"""
Configuration management for Body-Dynamics.
Loads settings from YAML file with fallback defaults.
"""
import yaml
import os
from pathlib import Path
from typing import Any, Dict


class Config:
    """Application configuration with YAML file support."""
    
    def __init__(self, config_path: str = None):
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to config.yaml (default: config/default.yaml)
        """
        # Default values (fallback if no config file)
        self._config = {
            "app": {
                "name": "Body-Dynamics",
                "version": "1.0.0",
                "environment": "development"
            },
            "pose": {
                "model_complexity": 0,
                "min_detection_confidence": 0.5,
                "min_tracking_confidence": 0.5,
                "smooth_landmarks": False,
                "min_visibility": 0.2
            },
            "processing": {
                "max_workers": 2,
                "timeout_seconds": 0.2,
                "stats_interval": 10
            },
            "biomechanics": {
                "ground_epsilon": 0.08,
                "foot_contact_duration": 0.05,
                "com_trail_length": 30,
                "stability": {
                    "margin_epsilon": 0.015,
                    "unstable_duration": 0.25,
                    "hysteresis": 0.01
                },
                "support": {
                    "persistence_duration": 0.35
                },
                "phase": {
                    "recovery_duration": 0.4
                },
                "joints": {
                    "elbow": {
                        "flexion_threshold": 140,
                        "extension_threshold": 160
                    }
                }
            },
            "websocket": {
                "max_buffer_bytes": 50000,
                "ping_interval": 30,
                "timeout": 60
            },
            "logging": {
                "level": "INFO",
                "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            },
            "metrics": {
                "enabled": True,
                "performance_window": 100
            }
        }
        
        # Try to load from file
        if config_path is None:
            # Look for config in standard locations
            possible_paths = [
                Path("config/default.yaml"),
                Path("config.yaml"),
                Path("../config/default.yaml"),
            ]
            for path in possible_paths:
                if path.exists():
                    config_path = str(path)
                    break
        
        if config_path and os.path.exists(config_path):
            self._load_from_file(config_path)
    
    def _load_from_file(self, path: str):
        """Load configuration from YAML file."""
        try:
            with open(path, 'r') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    self._deep_update(self._config, file_config)
                    print(f"✓ Loaded configuration from {path}")
        except Exception as e:
            print(f"⚠ Warning: Could not load config from {path}: {e}")
            print("  Using default configuration")
    
    def _deep_update(self, base: Dict, updates: Dict):
        """Recursively update nested dictionaries."""
        for key, value in updates.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value
    
    def get(self, *keys, default=None) -> Any:
        """
        Get configuration value using dot notation.
        
        Example:
            config.get('biomechanics', 'ground_epsilon')
            config.get('pose', 'model_complexity')
        """
        value = self._config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value
    
    # Convenience properties for common config values
    @property
    def ground_epsilon(self) -> float:
        return self.get('biomechanics', 'ground_epsilon', default=0.08)
    
    @property
    def foot_contact_time(self) -> float:
        return self.get('biomechanics', 'foot_contact_duration', default=0.05)
    
    @property
    def max_trail(self) -> int:
        return self.get('biomechanics', 'com_trail_length', default=30)
    
    @property
    def processing_timeout(self) -> float:
        return self.get('processing', 'timeout_seconds', default=0.2)
    
    @property
    def stats_interval(self) -> int:
        return self.get('processing', 'stats_interval', default=10)
    
    @property
    def max_workers(self) -> int:
        return self.get('processing', 'max_workers', default=2)
    
    @property
    def log_level(self) -> str:
        return self.get('logging', 'level', default='INFO')
    
    @property
    def log_format(self) -> str:
        return self.get('logging', 'format', 
                       default='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    def to_dict(self) -> Dict:
        """Export all configuration as dictionary."""
        return self._config.copy()


# Global config instance
config = Config()
