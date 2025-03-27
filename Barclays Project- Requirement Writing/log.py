import os
import logging
from logging.handlers import RotatingFileHandler
from typing import Optional

class LoggerManager:
    """
    Centralized logging management for the requirements AI system
    """
    _instance = None
    
    def __new__(cls):
        if not cls._instance:
            cls._instance = super(LoggerManager, cls).__new__(cls)
            cls._instance._setup_logging()
        return cls._instance
    
    def _setup_logging(self):
        """
        Configure logging with multiple handlers
        """
        # Get configuration
        from config_management import config
        
        # Root logger
        self.logger = logging.getLogger('RequirementsAI')
        
        # Log level from config
        log_level_str = config.get('system.log_level', 'INFO')
        log_level = getattr(logging, log_level_str.upper(), logging.INFO)
        self.logger.setLevel(log_level)
        
        # Clear any existing handlers
        self.logger.handlers.clear()
        
        # Log directory
        log_dir = config.get('paths.output_dir', './logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        
        # File Handler (Rotating)
        file_handler = RotatingFileHandler(
            os.path.join(log_dir, 'requirements_ai.log'),
            maxBytes=10*1024*1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(log_level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        
        # Add handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
    
    def get_logger(self, name: Optional[str] = None):
        """
        Get a logger for a specific module
        """
        if name:
            return logging.getLogger(f'RequirementsAI.{name}')
        return self.logger
    
    def log_exception(self, message: str, exc_info=True):
        """
        Log an exception with additional context
        """
        self.logger.exception(message, exc_info=exc_info)

# Singleton instance
logger_manager = LoggerManager()
