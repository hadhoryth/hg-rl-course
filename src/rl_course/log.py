import logging
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class LogColors:
    """ANSI color codes for log formatting"""

    GREY: str = "\x1b[38;20m"
    GREEN: str = "\x1b[32;20m"
    YELLOW: str = "\x1b[33;20m"
    RED: str = "\x1b[31;20m"
    BOLD_RED: str = "\x1b[31;1m"
    RESET: str = "\x1b[0m"


class ColoredFormatter(logging.Formatter):
    """Custom formatter adding colors to log outputs"""

    def __init__(
        self, log_format: Optional[str] = None, colors: Optional[LogColors] = None
    ):
        """
        Initialize the colored formatter.

        Args:
            log_format: Custom log format string. If None, uses default format
            colors: Custom color scheme. If None, uses default colors
        """
        self.colors = colors or LogColors()
        self.log_format = log_format or (
            "[%(levelname)s][%(filename)s@%(lineno)d]: %(message)s"
        )

        self.FORMATS: Dict[int, str] = {
            logging.DEBUG: self.colors.GREY + self.log_format + self.colors.RESET,
            logging.INFO: self.colors.GREEN + self.log_format + self.colors.RESET,
            logging.WARNING: self.colors.YELLOW + self.log_format + self.colors.RESET,
            logging.ERROR: self.colors.RED + self.log_format + self.colors.RESET,
            logging.CRITICAL: self.colors.BOLD_RED
            + self.log_format
            + self.colors.RESET,
        }

        super().__init__(fmt=self.log_format)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with appropriate colors.

        Args:
            record: The log record to format

        Returns:
            Formatted log string with colors
        """
        log_format = self.FORMATS.get(record.levelno, self.log_format)
        formatter = logging.Formatter(log_format)
        return formatter.format(record)


def create_logger(
    name: str = "main_logger",
    level: int = logging.INFO,
    log_format: Optional[str] = None,
    colors: Optional[LogColors] = None,
) -> logging.Logger:
    """
    Create or get a logger with colored output.

    Args:
        name: Name of the logger
        level: Logging level
        log_format: Custom log format string
        colors: Custom color scheme

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)

    # Return existing logger if already configured
    if logger.hasHandlers():
        return logger

    # Remove any existing handlers
    for handler in logger.handlers:
        logger.removeHandler(handler)

    # Configure stream handler
    handler = logging.StreamHandler()
    handler.setLevel(level)
    handler.setFormatter(ColoredFormatter(log_format, colors))

    # Configure logger
    logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False

    return logger


# Create default logger instance
logger = create_logger()
