__version__ = "0.0.2"

from loguru import logger as _logger
import sys

_format = "[<green>{time:MM/DD HH:mm:ss}</green> <magenta>{name}</magenta>]: {message}"
_logger.remove()
_logger.add(sys.stderr, colorize=True, format=_format)
