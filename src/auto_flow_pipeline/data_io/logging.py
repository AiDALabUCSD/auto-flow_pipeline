from tqdm import tqdm

def log_message(message, log_file):
    """
    Logs a message to both the console and a log file.
    
    Parameters:
    message (str): The message to log.
    log_file (str): The path to the log file.
    """
    # tqdm.write(message)
    with open(log_file, 'a') as f:
        f.write(message + '\n')