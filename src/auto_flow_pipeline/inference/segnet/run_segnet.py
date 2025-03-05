import os
import nibabel as nib
import numpy as np
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.inference.segnet.model_loader import load_segnet
import traceback
import logging
from typing import Any, Tuple
from auto_flow_pipeline.visualization.inference.segnet.run_segnet.generate_gifs import generate_four_row_gifs_for_slices_w_pred

def collapse_data(data: np.ndarray) -> np.ndarray:
    """
    Collapse the data array from shape (time, rows, cols, slices, channels) to
    (time*slices, rows, cols, channels).

    Parameters
    ----------
    data : np.ndarray
        5D input array.

    Returns
    -------
    np.ndarray
        4D collapsed array.
    """
    # Get the shape of the input data
    time, rows, cols, slices, channels = data.shape

    # Transpose the data to bring slices to the first dimension
    data = np.transpose(data, (0, 3, 1, 2, 4))

    # Reshape the data array
    return data.reshape(time * slices, rows, cols, channels)

def undo_collapse_data(
    pred: np.ndarray,
    original_time: int,
    original_slices: int,
    rows: int,
    cols: int
) -> np.ndarray:
    """
    Reshape array from (time*slices, rows, cols) back to
    (time, rows, cols, slices).

    Parameters
    ----------
    pred : np.ndarray
        Collapsed prediction array.
    original_time : int
    original_slices : int
    rows : int
    cols : int

    Returns
    -------
    np.ndarray
        Restored prediction array.
    """
    # Reshape the prediction array back to (time, slices, rows, cols)
    pred = pred.reshape(original_time, original_slices, rows, cols)

    # Transpose the prediction to bring it back to the original shape (time, rows, cols, slices)
    return np.transpose(pred, (0, 2, 3, 1))

def run_segnet_inference(
    model: Any,
    data: np.ndarray,
    logger: logging.Logger
) -> np.ndarray:
    """
    Run inference using the provided SegNet model.

    Parameters
    ----------
    model : Any
        SegNet model with a predict() method.
    data : np.ndarray
        5D input data.
    logger : logging.Logger
        Logger instance for messages.

    Returns
    -------
    np.ndarray
        The prediction reshaped to the original data structure.
    """
    # Get the shape of the input data
    time, rows, cols, slices, channels = data.shape

    logger.info(f"Input data shape: {data.shape}")
    logger.info("Starting SegNet inference.")

    # Collapse the data
    collapsed_data = collapse_data(data)

    logger.info(f"Collapsed to shape: {collapsed_data.shape}")

    # Run inference on the collapsed data
    pred = model.predict(collapsed_data)

    # Remove the extra channel dimension from the prediction
    pred = np.squeeze(pred['output'], axis=-1)

    # Reshape the prediction back to the original shape
    restored_pred = undo_collapse_data(pred, time, slices, rows, cols)

    logger.info("Model prediction successful.")
    logger.info(f"Prediction shape: {restored_pred.shape}")
    return restored_pred

def run_a_and_p_segnet_inference(
    model: Any,
    base_output_folder: str,
    patient_name: str
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load aorta and pulmonary NIfTI files, run inference,
    and save compressed outputs.

    Parameters
    ----------
    model : Any
    base_output_folder : str
    patient_name : str

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        (aorta_data, pulmonary_data, aorta_pred, pulmonary_pred).
    """
    logger = setup_logger(patient_name, base_output_folder)
    try:
        path_to_patient = os.path.join(base_output_folder, patient_name)
        input_aorta_nii_path = os.path.join(path_to_patient, 'aorta_spline_composed.nii.gz')
        if not os.path.exists(input_aorta_nii_path):
            raise FileNotFoundError(f"Could not find {input_aorta_nii_path}.")

        logger.info(f"Loading composed aorta spline from {input_aorta_nii_path}")
        nii = nib.load(input_aorta_nii_path)
        aorta_data = nii.get_fdata()

        logger.info("Running model inference for aorta spline")
        aorta_pred = run_segnet_inference(model, aorta_data, logger)
        
        out_nii = nib.Nifti1Image(aorta_pred, affine=nii.affine)
        save_path = os.path.join(path_to_patient, 'segnet_aorta-pred_processed.nii.gz')
        nib.save(out_nii, save_path)

        input_pulmonary_nii_path = os.path.join(path_to_patient, 'pulmonary_spline_composed.nii.gz')
        if not os.path.exists(input_pulmonary_nii_path):
            raise FileNotFoundError(f"Could not find {input_pulmonary_nii_path}.")
        
        logger.info(f"Loading composed pulmonary spline from {input_pulmonary_nii_path}")
        nii = nib.load(input_pulmonary_nii_path)
        pulmonary_data = nii.get_fdata()

        logger.info("Running model inference for pulmonary spline")
        pulmonary_pred = run_segnet_inference(model, pulmonary_data, logger)

        out_nii = nib.Nifti1Image(pulmonary_pred, affine=nii.affine)
        save_path = os.path.join(path_to_patient, 'segnet_pulmonary-pred_processed.nii.gz')
        nib.save(out_nii, save_path)
        
        logger.info(f"Inference complete. Output saved at: {save_path}")
        return aorta_data, aorta_pred, pulmonary_data, pulmonary_pred
    except Exception as e:
        logger.error(f"Error during inference: {str(e)}")
        logger.error(traceback.format_exc())
        raise

def main() -> None:
    """
    Main function to set up logging, load SegNet model,
    and handle inference for a specific patient.
    """
    base_output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'  # Update this path as needed
    patient_name = "Bepemhir"  # Update this patient name as needed
    
    main_logger.info("Loading SegNet model...")
    model = load_segnet()  # Load the SegNet model
    
    try:
        main_logger.info(f"Starting inference for patient: {patient_name}")
        aorta_input, aorta_prediction, pulmonary_input, pulmonary_prediction = run_a_and_p_segnet_inference(model, base_output_folder, patient_name)
        main_logger.info(f"Completed inference for patient: {patient_name}")
        
        main_logger.info(f"Generating Aorta SegNet GIF for patient: {patient_name}")
        generate_four_row_gifs_for_slices_w_pred(patient_name, base_output_folder,'aorta_segnet_predictions', aorta_input, aorta_prediction)
        main_logger.info(f"Completed Aorta SegNet GIF generation for patient: {patient_name}")

        main_logger.info(f"Generating Pulmonary SegNet GIF for patient: {patient_name}")
        generate_four_row_gifs_for_slices_w_pred(patient_name, base_output_folder,'pulmonary_segnet_predictions', pulmonary_input, pulmonary_prediction)
        main_logger.info(f"Completed Pulmonary SegNet GIF generation for patient: {patient_name}")
    except Exception as e:
        main_logger.error(f"Failed inference for patient: {patient_name} with error: {str(e)}")
        main_logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()