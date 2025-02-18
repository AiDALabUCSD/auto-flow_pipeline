import os
import pandas as pd
from auto_flow_pipeline.data_io.catalogue_patients import get_patient_info
from auto_flow_pipeline import main_logger

def update_patient_catalogue(base_output_folder: str, catalogue_path: str):
    """
    Updates the central patient catalogue with information for all patients.

    Parameters:
        base_output_folder (str): Path to the folder where patient data is stored.
        catalogue_path (str): Path to the central patient catalogue CSV file.
    """
    # Log the start of the update process
    main_logger.info("Starting update of patient catalogue.")
    
    # Get the list of patient directories
    patient_dirs = [
        d for d in os.listdir(base_output_folder)
        if os.path.isdir(os.path.join(base_output_folder, d))
    ]
    main_logger.info(f"Found {len(patient_dirs)} patient directories.")

    # Initialize an empty DataFrame or load existing catalogue
    if os.path.exists(catalogue_path):
        catalogue_df = pd.read_csv(catalogue_path)
        main_logger.info("Loaded existing patient catalogue.")
    else:
        catalogue_df = pd.DataFrame(columns=["patient_id", "vel_shape", "slice_diff", "cross_prod"])
        main_logger.info("Initialized new patient catalogue.")

    # Update the catalogue with information for each patient
    for pid in patient_dirs:
        main_logger.info(f"Processing patient directory: {pid}")
        patient_info = get_patient_info(pid, base_output_folder)
        
        # Check if the patient ID already exists in the catalogue
        if pid in catalogue_df['patient_id'].values:
            # Update the existing entry
            catalogue_df.loc[catalogue_df['patient_id'] == pid, 'vel_shape'] = str(patient_info['vel_shape'])
            catalogue_df.loc[catalogue_df['patient_id'] == pid, 'slice_diff'] = patient_info['slice_diff']
            catalogue_df.loc[catalogue_df['patient_id'] == pid, 'cross_prod'] = str(patient_info['cross_prod'])
            main_logger.info(f"Updated existing entry for patient {pid}.")
        else:
            # Append the new entry
            patient_info['vel_shape'] = str(patient_info['vel_shape'])
            patient_info['cross_prod'] = str(patient_info['cross_prod'])
            catalogue_df = pd.concat([catalogue_df, pd.DataFrame([patient_info])], ignore_index=True)
            main_logger.info(f"Added new entry for patient {pid}.")

    # Save the updated catalogue
    catalogue_df.to_csv(catalogue_path, index=False)
    main_logger.info(f"Saved updated patient catalogue to {catalogue_path}")

def main():
    base_output_folder = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    catalogue_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patient_catalogue.csv"
    
    update_patient_catalogue(base_output_folder, catalogue_path)

if __name__ == "__main__":
    main()