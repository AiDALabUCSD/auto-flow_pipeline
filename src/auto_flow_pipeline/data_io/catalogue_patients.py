import os
import pandas as pd
from auto_flow_pipeline.data_io.dicom_to_nifti import (
    find_difference_between_slices,
    find_cross_product_orientation
)

def load_patient_catalogue(base_output_folder: str = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/") -> pd.DataFrame:
    """
    Loads the patient catalogue DataFrame from 'patient_catalogue.csv'.

    Parameters:
        base_output_folder (str): Path to the folder where 'patients.csv' is stored. Defaults to the predefined path.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the patient catalogue data.
    """
    patients_path = os.path.join(base_output_folder, "patient_catalogue.csv")
    return pd.read_csv(patients_path)

def save_patient_catalogue(df: pd.DataFrame, base_output_folder: str = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/"):
    """
    Saves the patient catalogue DataFrame to 'patient_catalogue.csv'.

    Parameters:
        df (pd.DataFrame): The DataFrame to save.
        base_output_folder (str): Path to the folder where 'patients.csv' is stored. Defaults to the predefined path.
    """
    patients_path = os.path.join(base_output_folder, "patient_catalogue.csv")
    df.to_csv(patients_path, index=False)

def get_vel_npy_shape(flow_info_df: pd.DataFrame) -> tuple:
    """
    Extracts the vel_npy_shape field from an already-loaded flow_info DataFrame.

    Parameters:
        flow_info_df (pd.DataFrame): The patient's flow info DataFrame, already loaded.

    Returns:
        tuple: The shape of the velocity npy data as a tuple.
    """
    vel_npy_shape = flow_info_df['vel_npy_shape'].iloc[0]
    return tuple(map(int, vel_npy_shape.strip('()').split(',')))


def get_flow_info_df(pid: str, base_output_folder: str) -> pd.DataFrame:
    """
    Loads the patient-specific flow_info dataframe from 'flow_info.csv'.

    Parameters:
        pid (str): Patient ID.
        base_output_folder (str): Path to the folder where 'flow_info.csv' is stored.

    Returns:
        pd.DataFrame: A pandas DataFrame containing the patient's flow_info data.
    """
    flow_info_path = os.path.join(base_output_folder, pid, "flow_info.csv")
    return pd.read_csv(flow_info_path)


def get_patient_info(pid: str, base_output_folder: str) -> dict:
    """
    Gathers all relevant information for a patient and returns it as a dictionary.

    Parameters:
        pid (str): Patient ID.
        base_output_folder (str): Path to the folder where 'flow_info.csv' is stored.

    Returns:
        dict: A dictionary containing the patient's information.
    """
    df = get_flow_info_df(pid, base_output_folder)
    vel_shape = get_vel_npy_shape(df)
    slice_diff = find_difference_between_slices(df)
    cross_prod = find_cross_product_orientation(df)

    return {
        "patient_id": pid,
        "vel_shape": vel_shape,
        "slice_diff": slice_diff,
        "cross_prod": cross_prod
    }

def get_slice_diff(patient_name, catalogue_df):
    """
    Gets the slice_diff for a given patient.
    
    Args:
        patient_name (str): The name of the patient.
        catalogue_df (pd.DataFrame): The patient catalogue dataframe.
    
    Returns:
        float: The slice_diff for the patient.
    """
    return float(catalogue_df.loc[catalogue_df['patient_id'] == patient_name, 'slice_diff'].values[0])

def main():
    """
    Example main function to demonstrate directly using functions from dicom_to_nifti.
    """
    patient_name = "Bulosul"
    base_output_folder = (
        "/home/ayeluru/mnt/maxwell/projects/"
        "Aorta_pulmonary_artery_localization/ge_testing/patients"
    )

    flow_info_path = os.path.join(base_output_folder, patient_name, "flow_info.csv")
    df = pd.read_csv(flow_info_path)

    # Show how we can get the velocity shape:
    vel_shape = get_vel_npy_shape(df)
    print(f"Velocity numpy shape for patient '{patient_name}': {vel_shape}")

    # Directly call functions from dicom_to_nifti instead of using wrapper functions
    slice_diff = find_difference_between_slices(df)
    print(f"Slice difference (z-direction) for '{patient_name}': {slice_diff}")

    cross_prod = find_cross_product_orientation(df)
    print(f"Cross product orientation (z-direction) for '{patient_name}': {cross_prod}")


if __name__ == "__main__":
    main()