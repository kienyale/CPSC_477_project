import os
import logging
from typing import Dict, Optional

def setup_logging(
    config: Dict,
    log_file: Optional[str] = None
) -> None:
    """
    sets up logging config
    
    args:
    - config: logging settings
    - log_file: optional log file path
    """
    # make logs dir
    if log_file and not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    
    # basic config
    logging_config = {
        'level': getattr(logging, config['logging']['level']),
        'format': config['logging']['format'],
        'datefmt': '%Y-%m-%d %H:%M:%S'
    }
    
    # add file handler if needed
    if log_file:
        logging_config['handlers'] = [
            logging.StreamHandler(),  # console
            logging.FileHandler(log_file)  # file
        ]
    
    # apply config
    logging.basicConfig(**logging_config)
    
    # quiet noisy loggers
    logging.getLogger('transformers').setLevel(logging.WARNING)
    logging.getLogger('torch').setLevel(logging.WARNING)
    
    logger = logging.getLogger(__name__)
    logger.info("logging setup done")

if __name__ == "__main__":
    # quick test
    import yaml
    
    with open("config/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    setup_logging(
        config,
        log_file=os.path.join(config['paths']['logs'], 'training.log')
    ) 