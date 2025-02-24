import os
import nibabel as nib
import numpy as np
import pandas as pd
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))

def find_all_max_locations(prd=None, patient_name=None, base_folderpath=None, timepoint=None, logger=None):
    """
    Either pass in a 5D array (prd) or specify 'patient_name' and 'base_folderpath'
    to load 'locnet_pred_processed.nii.gz' automatically.

    Finds the maximum-intensity location for each (timepoint, channel) in a 5D array
    and stores the results in a pandas DataFrame, along with channel names.
    Also saves the DataFrame to the patient folder with the name 'max_points.csv'.
    """
    if logger is None:
        logger = setup_logger(patient_name, base_folderpath)
    
    logger.info("Starting find_all_max_locations...")

    # If no array is provided, load NIfTI
    if prd is None:
        if not (patient_name and base_folderpath):
            raise ValueError("Either 'prd' must be provided, or 'patient_name' and 'base_folderpath' must be specified.")
        nifti_path = os.path.join(base_folderpath, patient_name, "locnet_pred_processed.nii.gz")
        logger.info(f"Loading NIfTI file from {nifti_path}...")
        prd_img = nib.load(nifti_path)
        prd = prd_img.get_fdata(dtype=np.float32)
        logger.info("NIfTI file loaded.")

    titles = [
        "AV", "Proximal AAo", "Mid AAo", "Full Ao",
        "PV", "Proximal MPA", "MPA", "Full PA with branches"
    ]

    R, C, S, T, Ch = prd.shape
    rows = []

    logger.info(f"Processing 5D array with shape {prd.shape}...")

    timepoints = range(T) if timepoint is None else [timepoint]

    for t in timepoints:
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
            logger.info(f"Processed timepoint {t}, channel {ch}.")

    df = pd.DataFrame(rows, columns=["timepoint", "channel", "channel_name", "r", "c", "s"])

    # Save the DataFrame to the patient folder
    if patient_name and base_folderpath:
        output_path = os.path.join(base_folderpath, patient_name, "max_points.csv")
        df.to_csv(output_path, index=False)
        logger.info(f"DataFrame saved to {output_path}.")

    logger.info(df)
    logger.info("find_all_max_locations completed.")
    return df

def main():
    patient_name = "Bulosul"
    base_folderpath = (
        "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    )

    df = find_all_max_locations(patient_name=patient_name, base_folderpath=base_folderpath, timepoint=3)

if __name__ == "__main__":
    main()