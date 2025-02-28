import os
from tqdm import tqdm
import concurrent.futures
from auto_flow_pipeline.preprocessing.segnet.prepare_for_segnet import compose_and_save_splines
from auto_flow_pipeline import main_logger

def process_patient(pid, base_folderpath: str):
    main_logger.info(f"Preparing patient for segnet: {pid}")
    try:
        compose_and_save_splines(pid, base_folderpath)
        main_logger.info(f"Preparation complete for {pid}.")
    except Exception as e:
        main_logger.error(f"Error preparing {pid}: {e}")

def prepare_for_all_patients(base_folderpath: str):
    main_logger.info("Starting preparation for all patients.")
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    
    # Using ThreadPoolExecutor to process patients concurrently.
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = {executor.submit(process_patient, pid, base_folderpath): pid for pid in patient_dirs}
        # tqdm.with .as_completed to update the progress bar as tasks complete.
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Preparing patients"):
            pass

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    main_logger.info("running prepare_for_segnet_driver.py")
    prepare_for_all_patients(base_folderpath)
    main_logger.info("All patients prepared for segnet.")

if __name__ == "__main__":
    main()
