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
    Also saves the DataFrame to the patient folder with the name 'max_points.csv'.
    """
    print("Starting find_all_max_locations...")

    # If no array is provided, load NIfTI
    if prd is None:
        if not (patient_name and base_folderpath):
            raise ValueError("Either 'prd' must be provided, or 'patient_name' and 'base_folderpath' must be specified.")
        nifti_path = os.path.join(base_folderpath, patient_name, "locnet_pred_processed.nii.gz")
        print(f"Loading NIfTI file from {nifti_path}...")
        prd_img = nib.load(nifti_path)
        prd = prd_img.get_fdata(dtype=np.float32)
        print("NIfTI file loaded.")

    titles = [
        "AV", "Proximal AAo", "Mid AAo", "Full Ao",
        "PV", "Proximal MPA", "MPA", "Full PA with branches"
    ]

    R, C, S, T, Ch = prd.shape
    rows = []

    print(f"Processing 5D array with shape {prd.shape}...")

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
            print(f"Processed timepoint {t}, channel {ch}.")

    df = pd.DataFrame(rows, columns=["timepoint", "channel", "channel_name", "r", "c", "s"])

    if timepoint is not None:
        df = df[df["timepoint"] == timepoint].copy()
        print(f"Filtered DataFrame to timepoint {timepoint}.")

    # Save the DataFrame to the patient folder
    if patient_name and base_folderpath:
        output_path = os.path.join(base_folderpath, patient_name, "max_points.csv")
        df.to_csv(output_path, index=False)
        print(f"DataFrame saved to {output_path}.")

    print("find_all_max_locations completed.")
    return df

def main():
    patient_name = "Bulosul"
    base_folderpath = (
        "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    )

    df = find_all_max_locations(patient_name=patient_name, base_folderpath=base_folderpath)
    print(df)

if __name__ == "__main__":
    main()