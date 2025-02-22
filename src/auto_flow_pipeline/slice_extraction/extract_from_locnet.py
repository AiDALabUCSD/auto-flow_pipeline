import numpy as np
import pandas as pd

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))

def find_all_max_locations(prd, timepoint=None):
    """
    Finds the maximum-intensity location for each (timepoint, channel) in a 5D array
    and stores the results in a pandas DataFrame, along with channel names.

    If 'timepoint' is specified, only rows matching that timepoint are returned.

    Parameters:
    prd (np.ndarray): 5D array shaped (R, C, S, T, ch).
    timepoint (int): Timepoint to filter on (default=3).

    Returns:
    pd.DataFrame: DataFrame with columns:
                  ["timepoint", "channel", "channel_name", "r", "c", "s"]
    """
    titles = [
        "AV", "Proximal AAo", "Mid AAo", "Full Ao",
        "PV", "Proximal MPA", "MPA", "Full PA with branches"
    ]

    R, C, S, T, Ch = prd.shape
    rows = []

    for t in range(T):
        for ch in range(Ch):
            loc = _get_MAX(prd[..., t, ch])  # [r, c, s]
            rows.append({
                "timepoint": t,
                "channel": ch,
                "channel_name": titles[ch] if ch < len(titles) else f"Channel {ch}",
                "r": loc[0],
                "c": loc[1],
                "s": loc[2]
            })

    df = pd.DataFrame(rows, columns=["timepoint", "channel", "channel_name", "r", "c", "s"])

    # If a specific timepoint is requested, filter the DataFrame
    if timepoint is not None:
        df = df[df["timepoint"] == timepoint].copy()

    return df