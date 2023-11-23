# logging_config.py
import logging

def configure_logging(log_file="app.log", log_level=logging.INFO):
    """
    Configure logging for the application.

    Parameters:
    - log_file: str, optional, default: "app.log"
        The name of the log file.
    - log_level: int, optional, default: logging.INFO
        The logging level.

    Returns:
    - logger: logging.Logger
        The configured logger.
    """
    logger = logging.getLogger("my_app")
    logger.setLevel(log_level)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
