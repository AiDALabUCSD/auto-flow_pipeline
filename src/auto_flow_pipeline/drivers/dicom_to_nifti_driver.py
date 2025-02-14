import os
import logging
from tqdm import tqdm
from auto_flow_pipeline.data_io.dicom_to_nifti import patient_to_nifti
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def process_all_patients(dicom_base_folder, output_base_folder, velocity_base_folder, max_patients=-1, overwrite=False, main_logger=None):
    """
    Iterates through all directories in the dicom_base_folder and uses the
    patient_to_nifti function to process each patient to the output folder.

    Parameters:
        dicom_base_folder (str): Path to the folder containing DICOM files.
        output_base_folder (str): Folder to save the processed NIfTI files.
        velocity_base_folder (str): Folder containing the velocity numpy files.
        max_patients (int): Maximum number of patients to process. Default is -1 (process all patients).
        overwrite (bool): Whether to overwrite existing NIfTI files. Default is False.
        main_logger (logging.Logger): Logger instance for logging driver processes.
    """
    # Get the list of patient directories
    patient_dirs = [
        d for d in os.listdir(dicom_base_folder)
        if os.path.isdir(os.path.join(dicom_base_folder, d))
    ]

    # Limit the number of patients to process if max_patients is not -1
    if max_patients != -1:
        patient_dirs = patient_dirs[:max_patients]

    # Iterate through all directories with a progress bar
    for patient_id in tqdm(patient_dirs, desc="Processing all patients", position=0):
        patient_path = os.path.join(dicom_base_folder, patient_id)
        if os.path.isdir(patient_path):
            try:
                if main_logger:
                    main_logger.info(f"Beginning: {patient_id}")
                # Process each patient
                patient_to_nifti(
                    pid=patient_id,
                    base_dicom_folder=dicom_base_folder,
                    base_output_folder=output_base_folder,
                    base_velocity_folder=velocity_base_folder,
                    overwrite=overwrite
                )
                if main_logger:
                    main_logger.info(f"Successfully or previously processed patient {patient_id}")
            except Exception as e:
                error_message = f"Error processing patient {patient_id}: {e}"
                if main_logger:
                    main_logger.error(error_message)
                else:
                    print(error_message)

def main():
    dicom_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    velocity_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities'
    max_patients = -1
    overwrite=False
    
    # Setup main logger
    main_logger = setup_logger('ge_testing', '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/', console_log=True)
    main_logger.info("dicom_to_nifti_driver.py is about to commence.")
    main_logger.info(f"Preferences: dicom_base_folder={dicom_base_folder}, output_base_folder={output_base_folder}, velocity_base_folder={velocity_base_folder}, max_patients={max_patients}, overwrite={overwrite}")
    
    process_all_patients(dicom_base_folder, output_base_folder, velocity_base_folder, max_patients=max_patients, overwrite=overwrite, main_logger=main_logger)

if __name__ == "__main__":
    main()
