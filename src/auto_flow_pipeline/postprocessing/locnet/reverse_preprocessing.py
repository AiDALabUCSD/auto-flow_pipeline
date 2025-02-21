import nibabel as nib
import numpy as np
import cupy as cp
from cupyx.scipy.ndimage import zoom
import os
from auto_flow_pipeline.postprocessing.locnet.clean_prediction import center_box_clean, zero_out_values
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.visualization.postprocessing.reverse_preprocessing.generate_gifs import generate_gif

def reverse_reorder_5d(data, logger=None):
    """
    Reorders data from shape (T, R, C, S, Channels)
    back to shape (R, C, S, T, Channels).
    
    :param data: 5D numpy array of shape (T, R, C, S, Channels)
    :param logger: optional logger
    :return: 5D numpy array of shape (R, C, S, T, Channels)
    """
    if logger:
        logger.info("Reversing reorder of 5D volume from (T, R, C, S, Channels) to (R, C, S, T, Channels)")
    return np.transpose(data, (1, 2, 3, 0, 4))

def upsample_5d_volume(data, target_shape_4d, order=0, use_gpu=False, logger=None):
    """
    Upsamples a 5D volume (R, C, S, T, Channels) to match the first 4D shape 
    (R_orig, C_orig, S_orig, T_orig). The last dimension (Channels) is kept.
    
    :param data: 5D numpy array of shape (R, C, S, T, Channels)
    :param target_shape_4d: tuple (R_orig, C_orig, S_orig, T_orig)
                            from the original pre-downsample shape
    :param use_gpu: boolean flag to use GPU for upsampling
    :param logger: optional logger
    :return: 5D numpy array of shape (R_orig, C_orig, S_orig, T_orig, Channels)
    """
    if logger:
        logger.info(f"Upsampling 5D volume to original 4D shape {target_shape_4d} (keeping channels).")

    R_orig, C_orig, S_orig, T_orig = target_shape_4d
    R_in, C_in, S_in, T_in, ch_in = data.shape

    if use_gpu:
        # Log initial shape
        if logger:
            logger.info(f"Initial data shape: {data.shape}")

        # Move entire volume to GPU
        data_gpu = cp.array(data, dtype=cp.float32)

        # Perform 3D upscaling (R, C, S) while preserving T and Channels
        scale_factors = (R_orig / R_in, C_orig / C_in, S_orig / S_in, 1, 1)
        upsampled_gpu = zoom(data_gpu, scale_factors, order=order)

        # Log shape after upscaling
        if logger:
            logger.info(f"Shape after upscaling on GPU: {upsampled_gpu.shape}")

        # Move back to CPU
        upsampled = cp.asnumpy(upsampled_gpu)

        # Log final shape
        if logger:
            logger.info(f"Final upsampled shape: {upsampled.shape}")
    else:
        rfactor = R_orig / R_in
        cfactor = C_orig / C_in
        sfactor = S_orig / S_in
        tfactor = T_orig / T_in
        upsampled = zoom(data, (rfactor, cfactor, sfactor, tfactor, order), order=1)

        # Log shape after CPU upsampling
        if logger:
            logger.info(f"Final upsampled shape (CPU): {upsampled.shape}")

    return upsampled

def reverse_preprocessing_for_patient(
    patient_name: str,
    base_folderpath: str,
    generate_gif: bool = True,
    logger=None
):
    """
    1. Load predicted data (shape = (T, R, C, S, Channels)).
    2. Clean the prediction by zeroing out the boundaries.
    3. Reverse reorder to (R, C, S, T, Channels).
    4. Upsample to original (R, C, S, T), preserving channels.
    5. Save to new NIfTI file.
    
    :param patient_name: Name/ID of the patient.
    :param base_folderpath: Base folder path where patient data is stored.
    :param logger: optional logger
    :return: Processed 5D numpy array of shape (R_orig, C_orig, S_orig, T_orig, Channels)
    """

    if logger is None:
        logger = setup_logger(patient_name, base_folderpath)
    
    pred_nifti_path = os.path.join(base_folderpath, patient_name, "pred_from_locnet.nii.gz")
    original_nifti_path = os.path.join(base_folderpath, patient_name, "mag_4dflow.nii.gz")
    output_nifti_path = os.path.join(base_folderpath, patient_name, "locnet_pred_processed.nii.gz")

    if logger:
        logger.info(f"Loading prediction NIfTI from {pred_nifti_path}")
    pred_img = nib.load(pred_nifti_path)
    pred_data = pred_img.get_fdata(dtype=np.float32)

    if logger:
        logger.info(f"Prediction data shape (as loaded) = {pred_data.shape}")
    
    if logger:
        logger.info(f"Loading original NIfTI from {original_nifti_path}")
    original_img = nib.load(original_nifti_path)
    original_shape_4d = original_img.shape

    if len(original_shape_4d) != 4:
        raise ValueError("Original NIfTI does not appear in the correct shape.")

    cleaned_pred_data = center_box_clean(pred_data, b=45)
    cleaned_pred_data = zero_out_values(cleaned_pred_data, 0.1)
    reversed_order = reverse_reorder_5d(cleaned_pred_data, logger=logger)
    upsampled = upsample_5d_volume(reversed_order, original_shape_4d,order=1, use_gpu=True, logger=logger)

    if logger:
        logger.info(f"Saving reversed-preprocessing NIfTI to {output_nifti_path}")
    new_nifti = nib.Nifti1Image(upsampled, original_img.affine)
    nib.save(new_nifti, output_nifti_path)

    generate_gif(patient_name, base_folderpath, input_array=original_img.get_fdata(dtype=np.float32),n_jobs=4, pred_array=upsampled)
    logger.info("Generated GIF from processed array.")


    if logger:
        logger.info("Reverse preprocessing complete.")

    
    return upsampled

def main():
    patient_name = "Ackoram"
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    logger = setup_logger(patient_name, base_folderpath)

    processed_array = reverse_preprocessing_for_patient(
        patient_name,
        base_folderpath,
        logger=logger
    )
    logger.info("Done reversing preprocessing steps.")

if __name__ == "__main__":
    main()
