import copy
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage import exposure
from auto_flow_pipeline.data_io.logging_setup import setup_logger
import os
from auto_flow_pipeline.visualization.preprocessing.locnet.prepare_for_locnet.generate_gifs import generate_gif_from_preprocessed_nifti

def load_nifti(nifti_path: str, logger) -> np.ndarray:
    """
    Loads a NIfTI file and returns the 4D data array.
    """
    try:
        logger.info(f"Loading NIfTI file from {nifti_path}")
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata(dtype=np.float32)
        return data
    except Exception as e:
        logger.error(f"Error loading NIfTI file from {nifti_path}: {e}")
        raise

def downsample_4d_volume(data: np.ndarray,
                         target_dims=(192, 192, 64, 20), logger=None) -> np.ndarray:
    """
    Downsamples a 4D volume (R, C, S, T) to target_dims using a simple zoom.
    The last dimension (time) is preserved if it already matches.
    """
    try:
        logger.info(f"Downsampling 4D volume to target dimensions {target_dims}")
        rfactor = target_dims[0] / data.shape[0]
        cfactor = target_dims[1] / data.shape[1]
        sfactor = target_dims[2] / data.shape[2]
        tfactor = target_dims[3] / data.shape[3] if data.shape[3] != target_dims[3] else 1

        return zoom(data, (rfactor, cfactor, sfactor, tfactor), order=1)
    except Exception as e:
        logger.error(f"Error downsampling 4D volume: {e}")
        raise

def reorder_4d_volume(data: np.ndarray, logger=None) -> np.ndarray:
    """
    Reorders data from shape (R, C, S, T) to (T, R, C, S, 1).
    """
    try:
        logger.info("Reordering 4D volume")
        R, C, S, T = data.shape
        out = np.zeros((T, R, C, S, 1), dtype=data.dtype)
        for t in range(T):
            out[t, ..., 0] = data[..., t]
        return out
    except Exception as e:
        logger.error(f"Error reordering 4D volume: {e}")
        raise

def min_max_normalize_4d(data: np.ndarray, logger=None) -> np.ndarray:
    """
    Performs min-max normalization on each time-slice individually.
    Assumes shape (T, R, C, S, 1).
    """
    try:
        logger.info("Performing min-max normalization on 4D volume")
        T, R, C, S, _ = data.shape
        for t in range(T):
            mini = np.amin(data[t])
            maxi = np.amax(data[t])
            if maxi > mini:
                data[t] = (data[t] - mini) / (maxi - mini)
            else:
                data[t] = 0
        return data
    except Exception as e:
        logger.error(f"Error performing min-max normalization: {e}")
        raise

def percentile_rescale(data: np.ndarray, p_lower=5, p_upper=95, logger=None) -> np.ndarray:
    """
    Rescales intensities to the [p_lower, p_upper] percentile range
    for each time-slice. Assumes shape (T, R, C, S, 1).
    """
    try:
        logger.info(f"Rescaling intensities to the [{p_lower}, {p_upper}] percentile range")
        T = data.shape[0]
        for t in range(T):
            p_low_val = np.percentile(data[t], p_lower)
            p_high_val = np.percentile(data[t], p_upper)
            data[t] = exposure.rescale_intensity(
                data[t],
                in_range=(p_low_val, p_high_val)
            )
        return data
    except Exception as e:
        logger.error(f"Error rescaling intensities: {e}")
        raise

def preprocess_nifti_for_inference(patient_name: str, base_folderpath: str, overwrite: bool = False) -> np.ndarray:
    """
    Combines all steps:
      1. Load a 4D NIfTI file
      2. Downsample to (192,192,64,20)
      3. Reorder to (20,192,192,64,1)
      4. Min-max normalize each time-slice
      5. Rescale intensities with [5,95] percentile range
    Returns the preprocessed data ready for inference.
    """
    logger = setup_logger(patient_name, base_folderpath)
    nifti_path = f"{base_folderpath}/{patient_name}/mag_4dflow.nii.gz"
    output_path = f"{base_folderpath}/{patient_name}/mag_4dflow_for_locnet.nii.gz"
    logger.info(f"Starting preprocessing for patient {patient_name}")

    if os.path.exists(output_path) and not overwrite:
        logger.info(f"Preprocessed file already exists at {output_path}. Loading existing file.")
        return load_nifti(output_path, logger)

    try:
        data_4d = load_nifti(nifti_path, logger)
        downsampled = downsample_4d_volume(data_4d, (192, 192, 64, data_4d.shape[3]), logger)
        reordered = reorder_4d_volume(downsampled, logger)
        normalized = min_max_normalize_4d(reordered, logger)
        rescaled = percentile_rescale(normalized, p_lower=5, p_upper=95, logger=logger)
        
        # Save the preprocessed data
        nifti_img = nib.Nifti1Image(rescaled, np.eye(4))
        nib.save(nifti_img, output_path)
        logger.info(f"Saved preprocessed NIfTI file to {output_path}")

        logger.info(f"Completed preprocessing for patient {patient_name}")
        return rescaled
    except Exception as e:
        logger.error(f"Failed to preprocess NIfTI file for patient {patient_name}: {e}")
        raise

def main():
    patient_name = "Ackoram"
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    preprocessed = preprocess_nifti_for_inference(patient_name, base_folderpath, overwrite=False)
    print(preprocessed.shape)
    
    # Generate GIF from preprocessed NIfTI
    output_gif_path = f"{base_folderpath}/{patient_name}/mag_for_locnet.gif"
    logger = setup_logger(patient_name, base_folderpath)
    generate_gif_from_preprocessed_nifti(f"{base_folderpath}/{patient_name}/mag_4dflow_for_locnet.nii.gz", output_gif_path, logger)

if __name__ == "__main__":
    main()