# src/utils/logging_config.py  (Place this in src/utils)

import logging
import logging.config
import os

# Define log levels (This part is fine as is)
LOG_LEVELS = {
    'DEBUG': logging.DEBUG,
    'INFO': logging.INFO,
    'WARNING': logging.WARNING,
    'ERROR': logging.ERROR,
    'CRITICAL': logging.CRITICAL,
}

# Default log level (Good practice to use environment variables)
DEFAULT_LOG_LEVEL = os.environ.get('LOG_LEVEL', 'DEBUG').upper()
LOGGING_LEVEL = LOG_LEVELS.get(DEFAULT_LOG_LEVEL, logging.DEBUG)
LOGGING_LEVEL = logging.DEBUG

# Define log directory (Making this relative to the project root)
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "logs")


# Create logs directory if it doesn't exist
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Define logging configuration
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            # 'format': "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            'format': "%(module)s:%(lineno)d - %(levelname)s - %(message)s"
        },
        'detailed': {
            'format': "%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s"
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': LOGGING_LEVEL,
            'formatter': 'simple',
            'stream': 'ext://sys.stdout',  # Use ext:// for clarity
        },
        'file_handler': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': LOGGING_LEVEL,
            'formatter': 'detailed',
            'filename': os.path.join(LOG_DIR, 'assessment_gen.log'),  # More descriptive name
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,  # Keep 5 backup files
        },
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file_handler'],
            'level': 'INFO',  # Default root level to INFO
        },
        'src': {  # Logger for your application code
            'handlers': ['console', 'file_handler'],
            'level': LOGGING_LEVEL,
            'propagate': False,  # Prevent double logging
        },
        'langchain': { # example for library
            'handlers': ['console', 'file_handler'],
            'level': 'INFO', # Set to INFO/WARN to reduce verbosity
            'propagate': False,
        },

    },
}

def configure_logging():
    """Configures logging using the LOGGING_CONFIG dictionary."""
    logging.config.dictConfig(LOGGING_CONFIG)