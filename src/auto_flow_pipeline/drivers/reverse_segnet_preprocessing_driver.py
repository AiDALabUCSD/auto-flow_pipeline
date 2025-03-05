import os
from tqdm import tqdm
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.postprocessing.segnet.reverse_preprocessing import reverse_segmentation
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def process_patient(base_folderpath: str, patient_name: str):
    try:
        main_logger.info(f"Starting reverse preprocessing for patient: {patient_name}")
        logger = setup_logger(patient_name, base_folderpath)
        reverse_segmentation(patient_name, base_folderpath, logger=logger)
        main_logger.info(f"Completed reverse preprocessing for patient: {patient_name}")
    except Exception as e:
        main_logger.error(f"Failed reverse preprocessing for patient: {patient_name} with error: {str(e)}")

def run_reverse_preprocessing_for_all_patients(base_folderpath: str):
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]

    main_logger.info("Starting reverse preprocessing for all patients.")
    for pid in tqdm(patient_dirs, desc="Running reverse preprocessing"):
        process_patient(base_folderpath, pid)

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    run_reverse_preprocessing_for_all_patients(base_folderpath)
    main_logger.info("Reverse preprocessing for all patients complete.")

if __name__ == "__main__":
    main()
