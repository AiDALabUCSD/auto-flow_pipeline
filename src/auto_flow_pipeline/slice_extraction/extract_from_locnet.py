import numpy as np

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))