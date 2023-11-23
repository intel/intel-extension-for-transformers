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
    # 创建 logger
    logger = logging.getLogger("my_app")
    logger.setLevel(log_level)

    # 创建文件处理器并设置日志级别
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # 创建控制台处理器并设置日志级别
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)

    # 创建格式化器
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 将格式化器添加到处理器
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # 将处理器添加到 logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
