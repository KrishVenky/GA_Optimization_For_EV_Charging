"""
Centralized logging setup for the EV optimization project
"""

import logging
import sys
from pathlib import Path
from .config import LOG_LEVEL, LOG_FORMAT, LOG_FILE


def setup_logger(name: str = "ev_optimization", level: str = None) -> logging.Logger:
    """
    Set up and return a configured logger
    
    Args:
        name: Logger name (typically __name__ from calling module)
        level: Logging level (overrides config if provided)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Set level
    log_level = level if level else LOG_LEVEL
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    # Create formatters
    formatter = logging.Formatter(LOG_FORMAT)
    
    # Console handler (stdout)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    file_handler = logging.FileHandler(LOG_FILE, mode='a')
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger


def log_section(logger: logging.Logger, title: str, char: str = "="):
    """
    Log a formatted section header
    
    Args:
        logger: Logger instance
        title: Section title
        char: Character to use for decoration
    """
    separator = char * 60
    logger.info(separator)
    logger.info(f" {title}")
    logger.info(separator)


def log_data_summary(logger: logging.Logger, data, name: str):
    """
    Log summary statistics for a DataFrame or GeoDataFrame
    
    Args:
        logger: Logger instance
        data: DataFrame/GeoDataFrame to summarize
        name: Name of the dataset
    """
    logger.info(f"{name} Summary:")
    logger.info(f"  Shape: {data.shape}")
    logger.info(f"  Columns: {list(data.columns)}")
    logger.info(f"  Null values: {data.isnull().sum().sum()}")
    if hasattr(data, 'crs'):
        logger.info(f"  CRS: {data.crs}")
