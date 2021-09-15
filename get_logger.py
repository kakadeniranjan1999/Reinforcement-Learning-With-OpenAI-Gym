import logging


def get_logger(console_output_format, log_filename):
    logging.basicConfig(format=console_output_format, level=logging.INFO,
                        filename=log_filename, filemode='w')
    data_logger = logging.getLogger()
    handler = logging.StreamHandler()
    formatter = logging.Formatter(console_output_format)
    handler.setFormatter(formatter)
    handler.setLevel(logging.INFO)
    data_logger.addHandler(handler)

    return data_logger
