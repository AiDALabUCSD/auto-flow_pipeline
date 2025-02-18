import os
from tqdm import tqdm
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.preprocessing.locnet.prepare_for_locnet import preprocess_nifti_for_inference
from auto_flow_pipeline.visualization.preprocessing.locnet.prepare_for_locnet.generate_gifs import generate_gif_from_preprocessed_nifti

def process_patient(patient_name, base_folderpath):
    try:
        main_logger.info(f"Starting preprocessing for patient {patient_name}")
        preprocessed = preprocess_nifti_for_inference(patient_name, base_folderpath, overwrite=False)
        main_logger.info(f"Preprocessing completed for patient {patient_name}")
        
        # Generate GIF from preprocessed NIfTI
        output_gif_path = f"{base_folderpath}/{patient_name}/mag_for_locnet.gif"
        generate_gif_from_preprocessed_nifti(f"{base_folderpath}/{patient_name}/mag_4dflow_for_locnet.nii.gz", output_gif_path, main_logger)
        main_logger.info(f"GIF generation completed for patient {patient_name}")
    except Exception as e:
        main_logger.error(f"Error processing patient {patient_name}: {e}")

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    
    for patient_name in tqdm(patient_dirs, desc="Processing patients"):
        process_patient(patient_name, base_folderpath)

if __name__ == "__main__":
    main()
