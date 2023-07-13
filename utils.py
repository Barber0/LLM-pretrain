import logging


def build_logger(
    name,
    log_filename,
    level=logging.INFO,
    str_format='%(asctime)s [%(levelname)s] %(message)s'
):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter(str_format)
    ch.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(logging.FileHandler(log_filename))
    return logger