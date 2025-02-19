import os
import nibabel as nib
import numpy as np
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.inference.locnet.model_loader import load_locnet
from auto_flow_pipeline.visualization.inference.locnet.run_locnet.generate_gifs import generate_gif

def run_locnet_inference(model, base_output_folder, patient_name):
    """
    Loads the patient's mag dataset, runs inference via `model.predict`, 
    and saves the predicted output to 'pred_from_locnet.nii.gz'.

    Parameters:
    model (object): The model used for inference
    base_output_folder (str): The base folder where patient data is stored
    patient_name (str): The name/ID of the patient

    Returns:
    np.ndarray: The input data loaded from the NIfTI file
    np.ndarray: The predicted output from the model
    """
    logger = setup_logger(patient_name, base_output_folder)
    try:
        path_to_patient = os.path.join(base_output_folder, patient_name)
        input_nii_path = os.path.join(path_to_patient, 'mag_4dflow_for_locnet.nii.gz')
        if not os.path.exists(input_nii_path):
            raise FileNotFoundError(f"Could not find {input_nii_path}.")

        logger.info(f"Loading data from {input_nii_path}")
        nii = nib.load(input_nii_path)
        data = nii.get_fdata()
        
        logger.info("Running model inference")
        pred = model.predict(data)  # pred is a dict with "outputs" key
        outputs = pred['outputs'] if isinstance(pred, dict) else pred

        out_nii = nib.Nifti1Image(outputs, affine=nii.affine)
        save_path = os.path.join(path_to_patient, 'pred_from_locnet.nii.gz')
        nib.save(out_nii, save_path)
        
        logger.info(f"Inference complete. Output saved at: {save_path}")
        return data, outputs
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        raise

def main():
    """
    Main function to locally initialize LocNet, call run_locnet_inference for one patient, and create a GIF.
    """
    base_output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'  # Update this path as needed
    patient_name = "Ackoram"  # Update this patient name as needed
    
    logger = setup_logger('main_logger', base_output_folder)
    logger.info("Loading LocNet model...")
    model = load_locnet()  # Load the LocNet model
    
    try:
        logger.info(f"Starting inference for patient: {patient_name}")
        input_data, prediction = run_locnet_inference(model, base_output_folder, patient_name)
        logger.info(f"Completed inference for patient: {patient_name}")
        
        logger.info(f"Generating GIF for patient: {patient_name}")
        generate_gif(patient_name, base_output_folder, input_data, prediction)
        logger.info(f"Completed GIF generation for patient: {patient_name}")
    except Exception as e:
        logger.error(f"Failed inference for patient: {patient_name} with error: {str(e)}")

if __name__ == "__main__":
    main()
