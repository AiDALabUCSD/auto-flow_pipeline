import os
from tqdm import tqdm
from auto_flow_pipeline.data_io.parse_dicom_files import parse_patient
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def parse_all_dicoms(dicom_folder_path, output_folder_path, overwrite=False, main_logger=None):
    """
    Iterates through all directories in the dicom_folder_path and uses the
    parse_patient function to parse each patient to the output folder.

    Parameters:
        dicom_folder_path (str): Path to the folder containing DICOM files.
        output_folder_path (str): Folder to save the parsed DICOM information.
        overwrite (bool): Flag to control whether to overwrite existing patient folders.
        main_logger (logging.Logger): Logger instance for logging driver processes.
    """
    # Get the list of patient directories
    patient_dirs = [
        d for d in os.listdir(dicom_folder_path)
        if os.path.isdir(os.path.join(dicom_folder_path, d))
    ]

    # Iterate through all directories with a progress bar
    for i, patient_id in enumerate(
        tqdm(patient_dirs, desc="Parsing all patients", position=0)
    ):
        patient_path = os.path.join(dicom_folder_path, patient_id)
        if os.path.isdir(patient_path):
            try:
                if main_logger:
                    main_logger.info(f"Now parsing: {patient_id}")
                # Let parse_patient handle logger creation + logging
                parse_patient(
                    pid=patient_id,
                    dicom_folder_path=dicom_folder_path,
                    output_folder_path=output_folder_path,
                    overwrite=overwrite,
                    position=i+1
                )
                if main_logger:
                    main_logger.info(f"Successfully or previously parsed patient: {patient_id}")
            except Exception as e:
                error_message = f"Error parsing patient: {patient_id}: {e}"
                if main_logger:
                    main_logger.error(error_message)
                else:
                    print(error_message)

def main():
    dicom_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    overwrite = False
    
    # Setup main logger
    main_logger = setup_logger('ge_testing', '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/', console_log=True)
    main_logger.info("parse_dicom_files_driver.py is about to commence.")
    main_logger.info(f"Preferences: dicom_folder_path={dicom_folder_path}, output_folder_path={output_folder_path}, overwrite={overwrite}")
    
    parse_all_dicoms(dicom_folder_path, output_folder_path, overwrite=overwrite, main_logger=main_logger)

if __name__ == "__main__":
    main()
