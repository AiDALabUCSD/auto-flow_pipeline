import os
import logging

def setup_logger(patient_name, output_folder):
    logger = logging.getLogger(patient_name)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(output_folder, f'{patient_name}', f'{patient_name}.log')
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger