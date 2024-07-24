"""
The format and config of logging.
"""

import logging


LOGGER_TABLE = {}

FORMATTER = logging.Formatter(
    "[%(asctime)s] %(levelname)7s %(name)s: %(message)s", "%Y-%m-%d %H:%M:%S"
)
STREAM_HANDLER = logging.StreamHandler()
STREAM_HANDLER.setFormatter(FORMATTER)


def get_logger(name):
    """Attach to the default logger."""

    if name in LOGGER_TABLE:
        return LOGGER_TABLE[name]

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(STREAM_HANDLER)

    LOGGER_TABLE[name] = logger
    return logger


def enable_log_file(file_name):
    """Add file handler to all loggers."""

    file_handler = logging.FileHandler(file_name)
    file_handler.setFormatter(FORMATTER)

    for logger in LOGGER_TABLE.values():
        logger.addHandler(file_handler)


def disable_stream_handler(func):
    """Disable stream (console) handler when running a function."""

    def _wrapper(*args, **kwargs):
        for logger in LOGGER_TABLE.values():
            logger.removeHandler(STREAM_HANDLER)
        ret = func(*args, **kwargs)
        for logger in LOGGER_TABLE.values():
            logger.addHandler(STREAM_HANDLER)
        return ret

    return _wrapper
