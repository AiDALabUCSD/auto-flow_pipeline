import os
import sys
import logging
from tqdm import tqdm

class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except Exception:
            self.handleError(record)

def setup_logger(patient_name, output_folder, console_log=True):
    """
    Setup a logger for the given patient_name that logs to
    <output_folder>/<patient_name>/<patient_name>.log.

    :param patient_name: Name/ID of the 'patient' or context for the logger
    :param output_folder: Base folder where logs will be stored
    :param console_log: If True, also log to console (stdout)
    :return: Configured Logger instance
    """

    logger = logging.getLogger(patient_name)

    # Avoid adding multiple handlers if logger already exists
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # Make sure the directory for logs exists
        patient_folder = os.path.join(output_folder, patient_name)
        os.makedirs(patient_folder, exist_ok=True)

        # Prepare the log file path
        log_file = os.path.join(patient_folder, f'{patient_name}.log')

        # Create and configure the FileHandler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Optionally add console logging
        if console_log:
            # Use the custom TqdmLoggingHandler to play nicely with tqdm.
            console_handler = TqdmLoggingHandler(sys.stderr)
            console_handler.setLevel(logging.INFO)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

    return logger
