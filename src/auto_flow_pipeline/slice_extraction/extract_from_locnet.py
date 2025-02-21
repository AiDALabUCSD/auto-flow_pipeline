import numpy as np
import pandas as pd

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))

def find_all_max_locations(prd):
    """
    Finds the maximum-intensity location for each (timepoint, channel) in a 5D array
    and stores the results in a pandas DataFrame.

    Parameters:
    prd (np.ndarray): 5D array shaped (R, C, S, T, ch).

    Returns:
    pd.DataFrame: DataFrame with columns:
                  ["timepoint", "channel", "r", "c", "s"]
    """
    R, C, S, T, Ch = prd.shape
    rows = []

    for t in range(T):
        for ch in range(Ch):
            loc = _get_MAX(prd[..., t, ch])  # [r, c, s]
            rows.append({
                "timepoint": t,
                "channel": ch,
                "r": loc[0],
                "c": loc[1],
                "s": loc[2]
            })

    return pd.DataFrame(rows, columns=["timepoint", "channel", "r", "c", "s"])