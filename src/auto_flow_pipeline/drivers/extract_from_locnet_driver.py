import os
import pandas as pd
from tqdm import tqdm
from auto_flow_pipeline.slice_extraction.extract_from_locnet import find_all_max_locations
from auto_flow_pipeline import main_logger

def extract_for_all_patients(base_folderpath: str):
    main_logger.info("Starting locnet extraction for all patients.")
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    for pid in tqdm(patient_dirs, desc="Processing patients"):
        main_logger.info(f"Processing patient directory: {pid}")
        try:
            df = find_all_max_locations(patient_name=pid, base_folderpath=base_folderpath, timepoint=3)
            main_logger.info(f"Extraction complete for {pid}, found {len(df)} entries.")
        except Exception as e:
            main_logger.error(f"Error processing {pid}: {e}")

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    extract_for_all_patients(base_folderpath)
    main_logger.info("All patients processed.")

if __name__ == "__main__":
    main()