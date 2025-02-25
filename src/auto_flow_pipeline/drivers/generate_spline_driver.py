import os
from tqdm import tqdm
from auto_flow_pipeline.slice_extraction.generate_spline import generate_aortic_spline, generate_pulmonary_spline
from auto_flow_pipeline import main_logger

def generate_splines_for_all_patients(base_folderpath: str):
    main_logger.info("Starting spline generation for all patients.")
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    for pid in tqdm(patient_dirs, desc="Processing patients"):
        main_logger.info(f"Processing patient directory: {pid}")
        try:
            generate_aortic_spline(pid, base_folderpath)
            generate_pulmonary_spline(pid, base_folderpath)
            main_logger.info(f"Spline generation complete for {pid}.")
        except Exception as e:
            main_logger.error(f"Error processing {pid}: {e}")

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    main_logger.info("running generate_spline_driver.py")
    generate_splines_for_all_patients(base_folderpath)
    main_logger.info("All patients processed.")

if __name__ == "__main__":
    main()
