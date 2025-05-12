import logging

from termcolor import colored


class ColoredFormatter(logging.Formatter):
    COLORS = {
        "DEBUG": "cyan",
        "INFO": "green",
        "WARNING": "yellow",
        "ERROR": "red",
        "CRITICAL": "magenta",
    }

    def format(self, record):
        # all line in log will be colored
        log_message = super().format(record)
        return colored(log_message, self.COLORS.get(record.levelname, "white"))

        # only log level will be colored
        # levelname_colored = colored(record.levelname, self.COLORS.get(record.levelname, 'white'))
        # record.levelname = levelname_colored
        # return super().format(record)

        # only keywords will be colored
        # message = record.msg
        # for word, color in self.KEYWORDS.items():
        #     if word in message:
        #         message = message.replace(word, colored(word, color))
        # record.msg = message
        # return super().format(record)


# config log
dev_logger = logging.getLogger("dev")
dev_formatter = ColoredFormatter("%(asctime)s - %(levelname)s - %(message)s")
dev_handler = logging.StreamHandler()
dev_handler.setFormatter(dev_formatter)
dev_logger.addHandler(dev_handler)
dev_logger.setLevel(logging.INFO)

progress_logger = logging.getLogger("progress")
progress_handler = logging.StreamHandler()
progress_handler.setFormatter(ColoredFormatter("%(message)s"))
progress_logger.addHandler(progress_handler)
progress_logger.setLevel(logging.INFO)

dev_mode = False


def set_dev_mode(mode: bool):
    """set dev mode"""
    global dev_mode
    dev_mode = mode


def set_level(level):
    """set log level"""
    dev_logger.setLevel(level)


def debug(message):
    """debug log"""
    if dev_mode:
        dev_logger.debug(message)


def info(message):
    """info log"""
    if dev_mode:
        dev_logger.info(message)


def warning(message):
    """warning log"""
    if dev_mode:
        dev_logger.warning(message)


def error(message):
    """error log"""
    if dev_mode:
        dev_logger.error(message)


def critical(message):
    """critical log"""
    dev_logger.critical(message)
    raise RuntimeError(message)


def color_print(message, **kwargs):
    progress_logger.info(message)
