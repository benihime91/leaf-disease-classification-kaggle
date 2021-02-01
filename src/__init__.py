__version__ = "0.0.2"

from loguru import logger as _logger
import sys

_logger.remove()
format = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> - <cyan>{name:}</cyan>: [<level>{level}</level>] - {message}"
_logger.add(sys.stdout, format=format, colorize=True)
