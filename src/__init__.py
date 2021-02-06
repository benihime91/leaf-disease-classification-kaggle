__version__ = "0.0.2"

from loguru import logger as _logger
import sys

_logger.remove()
format = "<level>{level}</level>:<cyan>{name:}</cyan>:{message}"
_logger.add(sys.stdout, format=format, colorize=True)
