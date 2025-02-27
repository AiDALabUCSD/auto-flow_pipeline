import os
from tqdm import tqdm
import nibabel as nib
import concurrent.futures
from auto_flow_pipeline.slice_extraction.reslice import setup_patient_rgi, sample_aortic_spline, sample_pulmonary_spline
from auto_flow_pipeline import main_logger

def process_patient(pid, base_folderpath: str):
    main_logger.info(f"Processing patient directory: {pid}")
    try:
        # Set up the RGIs and affine.
        mag_rgi, flow_rgi = setup_patient_rgi(pid, base_folderpath)
        mag_path = os.path.join(base_folderpath, pid, "mag_4dflow.nii.gz")
        mag_img = nib.load(mag_path)
        affine = mag_img.affine
        
        # Example indices to sample along the spline.
        aortic_indices = [5, 10, 15, 20, 25]
        pulmonary_indices = [5, 15, 25, 35, 45]
        
        # Sample the aortic spline and save the result.
        sample_aortic_spline(pid, base_folderpath, aortic_indices, mag_rgi, flow_rgi, affine)
        
        # Sample the pulmonary spline and save the result.
        sample_pulmonary_spline(pid, base_folderpath, pulmonary_indices, mag_rgi, flow_rgi, affine)
        
        main_logger.info(f"Reslicing complete for {pid}.")
    except Exception as e:
        main_logger.error(f"Error processing {pid}: {e}")

def reslice_for_all_patients(base_folderpath: str):
    main_logger.info("Starting reslicing for all patients.")
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    
    # Using ThreadPoolExecutor to process patients concurrently.
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_patient, pid, base_folderpath): pid for pid in patient_dirs}
        # tqdm.with .as_completed to update the progress bar as tasks complete.
        for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing patients"):
            pass

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    main_logger.info("running reslice_driver.py")
    reslice_for_all_patients(base_folderpath)
    main_logger.info("All patients processed.")

if __name__ == "__main__":
    main()
