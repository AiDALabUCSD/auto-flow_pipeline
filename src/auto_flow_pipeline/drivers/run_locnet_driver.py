import os
from tqdm import tqdm
from auto_flow_pipeline.inference.locnet.run_locnet import run_locnet_inference
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.inference.locnet.model_loader import load_locnet
from auto_flow_pipeline.visualization.inference.locnet.run_locnet.generate_gifs import generate_gif
from auto_flow_pipeline import main_logger

def run_inference_for_all_patients(base_output_folder):
    """
    Helper function to run locnet inference for all patients in the base folder.

    Parameters:
    model (object): The model used for inference
    base_output_folder (str): The base folder where patient data is stored
    """
    main_logger.info("Inferencing is beginning...")

    try:
        main_logger.info(f"Loading LocNet...")
        model = load_locnet()  # Load the LocNet model
        main_logger.info(f"LocNet loaded successfully.")
    except Exception as e:
        main_logger.error(f"Failed to load LocNet with error: {str(e)}")
    
    patients = [name for name in os.listdir(base_output_folder) if os.path.isdir(os.path.join(base_output_folder, name))]
    
    for patient_name in tqdm(patients, desc="Running inference for patients"):
        
        try:
            main_logger.info(f"Starting inference for patient: {patient_name}")
            input_data, prediction = run_locnet_inference(model, base_output_folder, patient_name)
            main_logger.info(f"Completed inference for patient: {patient_name}")
            
            main_logger.info(f"Generating GIF for patient: {patient_name}")
            generate_gif(patient_name, base_output_folder, input_data, prediction)
            main_logger.info(f"Completed GIF generation for patient: {patient_name}")
        except Exception as e:
            main_logger.error(f"Failed inference for patient: {patient_name} with error: {str(e)}")

def main():
    """
    Main function to set paths and call the helper function to run inference.
    """
    base_output_folder = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"  # Update this path as needed
    run_inference_for_all_patients(base_output_folder)

if __name__ == "__main__":
    main()
