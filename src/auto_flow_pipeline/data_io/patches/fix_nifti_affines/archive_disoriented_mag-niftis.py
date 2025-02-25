import os
import shutil
from tqdm import tqdm
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline import main_logger

def archive_nifti_file(patient_name, base_folder_path):
    """
    Moves the mag_4dflow.nii.gz file to the archive folder for a given patient.
    
    Args:
        patient_name (str): The name of the patient.
        base_folder_path (str): The base folder path where the patient's data is stored.
    """
    logger = setup_logger(patient_name, base_folder_path)
    
    source_path = os.path.join(base_folder_path, patient_name, 'mag_4dflow.nii.gz')
    parent_folder_path = os.path.dirname(base_folder_path)
    archive_folder = os.path.join(parent_folder_path, 'archive', patient_name)
    os.makedirs(archive_folder, exist_ok=True)
    destination_path = os.path.join(archive_folder, 'mag_4dflow.nii.gz')
    
    if os.path.exists(source_path):
        shutil.move(source_path, destination_path)
        logger.info(f"Moved {source_path} to {destination_path}")
    else:
        logger.warning(f"File {source_path} does not exist")

def archive_all_patients(base_folder_path):
    """
    Goes through all the patients in the base patient folder and performs the archiving operation.
    
    Args:
        base_folder_path (str): The base folder path where the patients' data is stored.
    """
    patients = [name for name in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, name))]
    
    for patient_name in tqdm(patients, desc="Archiving patients"):
        main_logger.info(f"Processing patient: {patient_name}")
        archive_nifti_file(patient_name, base_folder_path)

def main():
    base_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    main_logger.info("Running archive_disoriented_mag-niftis.py")
    archive_all_patients(base_folder_path)
    main_logger.info("All patients processed.")

if __name__ == "__main__":
    main()
