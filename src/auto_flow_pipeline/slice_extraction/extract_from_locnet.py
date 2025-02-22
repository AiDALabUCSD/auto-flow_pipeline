import os
import nibabel as nib
import numpy as np
import pandas as pd

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))

def find_all_max_locations(prd=None, patient_name=None, base_folderpath=None, timepoint=3):
    """
    Either pass in a 5D array (prd) or specify 'patient_name' and 'base_folderpath'
    to load 'locnet_pred_processed.nii.gz' automatically.

    Finds the maximum-intensity location for each (timepoint, channel) in a 5D array
    and stores the results in a pandas DataFrame, along with channel names.
    """

    # If no array is provided, load NIfTI
    if prd is None:
        if not (patient_name and base_folderpath):
            raise ValueError("Either 'prd' must be provided, or 'patient_name' and 'base_folderpath' must be specified.")
        nifti_path = os.path.join(base_folderpath, patient_name, "locnet_pred_processed.nii.gz")
        prd_img = nib.load(nifti_path)
        prd = prd_img.get_fdata(dtype=np.float32)

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

    if timepoint is not None:
        df = df[df["timepoint"] == timepoint].copy()

    return df
