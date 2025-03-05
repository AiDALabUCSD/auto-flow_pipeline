import os
from tqdm import tqdm
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.inference.segnet.model_loader import load_segnet
from auto_flow_pipeline.inference.segnet.run_segnet import run_a_and_p_segnet_inference
from auto_flow_pipeline.visualization.inference.segnet.run_segnet.generate_gifs import generate_four_row_gifs_for_slices_w_pred

def process_patient(model, base_folderpath: str, patient_name: str):
    try:
        main_logger.info(f"Starting inference for patient: {patient_name}")
        aorta_input, aorta_prediction, pulmonary_input, pulmonary_prediction = run_a_and_p_segnet_inference(model, base_folderpath, patient_name)
        main_logger.info(f"Completed inference for patient: {patient_name}")
        
        main_logger.info(f"Generating Aorta SegNet GIF for patient: {patient_name}")
        generate_four_row_gifs_for_slices_w_pred(patient_name, base_folderpath, 'aorta_segnet_predictions', aorta_input, aorta_prediction)
        main_logger.info(f"Completed Aorta SegNet GIF generation for patient: {patient_name}")

        main_logger.info(f"Generating Pulmonary SegNet GIF for patient: {patient_name}")
        generate_four_row_gifs_for_slices_w_pred(patient_name, base_folderpath, 'pulmonary_segnet_predictions', pulmonary_input, pulmonary_prediction)
        main_logger.info(f"Completed Pulmonary SegNet GIF generation for patient: {patient_name}")
    except Exception as e:
        main_logger.error(f"Failed inference for patient: {patient_name} with error: {str(e)}")

def run_inference_for_all_patients(base_folderpath: str):
    main_logger.info("Loading SegNet model...")
    model = load_segnet()  # Load the SegNet model

    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]

    main_logger.info("Starting inference for all patients.")
    for pid in tqdm(patient_dirs, desc="Running inference"):
        process_patient(model, base_folderpath, pid)

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    run_inference_for_all_patients(base_folderpath)
    main_logger.info("Inference for all patients complete.")

if __name__ == "__main__":
    main()
