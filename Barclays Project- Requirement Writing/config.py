import os
import yaml
from typing import Dict, Any

class ConfigManager:
    """
    Centralized configuration management for the requirements AI system
    """
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config()
        return cls._instance
    
    def _load_config(self):
        """
        Load configuration from multiple sources
        Priority: Environment Variables > Config File > Default Settings
        """
        # Default configuration
        self.config = {
            'system': {
                'version': '1.0.0',
                'debug_mode': False,
                'log_level': 'INFO'
            },
            'paths': {
                'output_dir': './outputs',
                'temp_dir': './temp',
                'knowledge_base': './knowledge_base'
            },
            'nlp': {
                'model': 'en_core_web_sm',
                'similarity_threshold': 0.7
            },
            'requirements': {
                'max_extract_length': 500,
                'priority_keywords': {
                    'must_have': ['must', 'critical', 'essential', 'mandatory'],
                    'should_have': ['should', 'important', 'recommended'],
                    'could_have': ['could', 'optional', 'nice to have'],
                    'wont_have': ['won\'t', 'future', 'out of scope']
                }
            },
            'integrations': {
                'jira': {
                    'enabled': False,
                    'url': None,
                    'token': None
                },
                'confluence': {
                    'enabled': False,
                    'url': None,
                    'token': None
                }
            }
        }
        
        # Try to load from config file
        config_paths = [
            './config.yaml',
            './config.yml',
            os.path.expanduser('~/.requirements_ai/config.yaml')
        ]
        
        for path in config_paths:
            if os.path.exists(path):
                try:
                    with open(path, 'r') as file:
                        file_config = yaml.safe_load(file)
                        self._merge_configs(self.config, file_config)
                except Exception as e:
                    print(f"Warning: Could not load config from {path}: {e}")
        
        # Override with environment variables
        self._load_env_overrides()
        
        # Ensure output directories exist
        self._create_directories()
    
    def _merge_configs(self, base: Dict, update: Dict):
        """
        Recursively merge configuration dictionaries
        """
        for key, value in update.items():
            if isinstance(value, dict):
                base[key] = self._merge_configs(base.get(key, {}), value)
            else:
                base[key] = value
        return base
    
    def _load_env_overrides(self):
        """
        Override config with environment variables
        """
        env_mapping = {
            'REQUIREMENTS_AI_DEBUG': ('system.debug_mode', bool),
            'REQUIREMENTS_AI_LOG_LEVEL': ('system.log_level', str),
            'REQUIREMENTS_AI_OUTPUT_DIR': ('paths.output_dir', str),
            'JIRA_INTEGRATION_ENABLED': ('integrations.jira.enabled', bool),
            'JIRA_URL': ('integrations.jira.url', str),
            'JIRA_TOKEN': ('integrations.jira.token', str)
        }
        
        for env_var, (config_path, convert_func) in env_mapping.items():
            value = os.environ.get(env_var)
            if value is not None:
                try:
                    # Navigate through nested dictionary
                    config_keys = config_path.split('.')
                    target = self.config
                    for key in config_keys[:-1]:
                        target = target[key]
                    target[config_keys[-1]] = convert_func(value)
                except Exception as e:
                    print(f"Warning: Could not process environment variable {env_var}: {e}")
    
    def _create_directories(self):
        """
        Create necessary directories for the system
        """
        dirs_to_create = [
            self.config['paths']['output_dir'],
            self.config['paths']['temp_dir'],
            self.config['paths']['knowledge_base']
        ]
        
        for directory in dirs_to_create:
            os.makedirs(directory, exist_ok=True)
    
    def get(self, key: str, default=None):
        """
        Retrieve configuration value using dot notation
        """
        keys = key.split('.')
        value = self.config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k, default)
            else:
                return default
        return value
    
    def update(self, key: str, value: Any):
        """
        Update configuration dynamically
        """
        keys = key.split('.')
        target = self.config
        for k in keys[:-1]:
            target = target.setdefault(k, {})
        target[keys[-1]] = value

# Singleton instance
config = ConfigManager()
