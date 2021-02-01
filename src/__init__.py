__version__ = "0.0.1"

from loguru import logger as _logger
import sys

_logger.remove()
_fmt = "<green>{time:MM/DD HH:mm}</green> - <cyan>{name}</cyan> - <level>{level}</level> - {message}"
_logger.add(sys.stderr, format=_fmt, colorize=True)
