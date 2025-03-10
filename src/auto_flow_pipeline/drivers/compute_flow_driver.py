import os
from tqdm import tqdm
from multiprocessing import Pool
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.flow_calculation.compute_flow import process_patient_flow

def process_patient(args):
    patient_name, base_path = args
    process_patient_flow(patient_name, base_path)

def process_all_patients(base_path: str):
    """
    Process flow computations for all patients in the base folder.

    Parameters:
        base_path (str): Base path where patient data is stored.
    """
    main_logger.info("Starting to process all patients.")
    
    # List all patient directories
    patient_dirs = [d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))]
    
    # Prepare arguments for multiprocessing
    args = [(patient_name, base_path) for patient_name in patient_dirs]
    
    # Use multiprocessing to process patients in parallel
    with Pool() as pool:
        list(tqdm(pool.imap(process_patient, args), total=len(patient_dirs), desc="Processing patients"))
    
    main_logger.info("Completed processing all patients.")

def main():
    base_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    main_logger.info("Starting the flow computation driver.")
    process_all_patients(base_path)
    main_logger.info("Flow computation driver completed.")

if __name__ == "__main__":
    main()
