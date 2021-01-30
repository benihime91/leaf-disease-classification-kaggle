__version__ = "0.0.1"

from loguru import logger as _logger
import sys

_logger.remove()
_logger.add(sys.stderr, format="{message}")
