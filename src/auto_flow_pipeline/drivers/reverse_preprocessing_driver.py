import os
from tqdm import tqdm
from auto_flow_pipeline.postprocessing.locnet.reverse_preprocessing import reverse_preprocessing_for_patient
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def reverse_preprocessing_for_all_patients(base_folderpath, max_patients=-1):
    """
    Reverse preprocessing for all patients in the base folder.

    :param base_folderpath: Base folder path where patient data is stored.
    :param max_patients: Maximum number of patients to process. Default is -1 (process all patients).
    """
    main_logger.info("Starting reverse preprocessing for all patients.")

    patient_names = [name for name in os.listdir(base_folderpath) if os.path.isdir(os.path.join(base_folderpath, name))]
    
    if max_patients != -1:
        patient_names = patient_names[:max_patients]

    for patient_name in tqdm(patient_names, desc="Processing all patients", position=0):
        patient_logger = setup_logger(patient_name, base_folderpath)
        try:
            patient_logger.info(f"Processing patient: {patient_name}")
            reverse_preprocessing_for_patient(patient_name, base_folderpath, logger=patient_logger)
            patient_logger.info(f"Completed processing for patient: {patient_name}")
        except Exception as e:
            patient_logger.error(f"Error processing patient {patient_name}: {e}")

    main_logger.info("Completed reverse preprocessing for all patients.")

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"  # Update this path as needed
    max_patients = -1  # Update this value as needed

    main_logger.info("reverse_preprocessing_driver.py is about to commence.")
    main_logger.info(f"Preferences: base_folderpath={base_folderpath}, max_patients={max_patients}")

    reverse_preprocessing_for_all_patients(base_folderpath, max_patients=max_patients)

if __name__ == "__main__":
    main()
