import copy
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from skimage import exposure

def load_nifti(nifti_path: str) -> np.ndarray:
    """
    Loads a NIfTI file and returns the 4D data array.
    """
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata(dtype=np.float32)
    return data

def downsample_4d_volume(data: np.ndarray,
                         target_dims=(192, 192, 64, 20)) -> np.ndarray:
    """
    Downsamples a 4D volume (R, C, S, T) to target_dims using a simple zoom.
    The last dimension (time) is preserved if it already matches.
    """
    rfactor = target_dims[0] / data.shape[0]
    cfactor = target_dims[1] / data.shape[1]
    sfactor = target_dims[2] / data.shape[2]
    # Time dimension factor is 1 if target and data match; otherwise adjust as needed
    tfactor = target_dims[3] / data.shape[3] if data.shape[3] != target_dims[3] else 1

    return zoom(data, (rfactor, cfactor, sfactor, tfactor), order=1)

def reorder_4d_volume(data: np.ndarray) -> np.ndarray:
    """
    Reorders data from shape (R, C, S, T) to (T, R, C, S, 1).
    """
    R, C, S, T = data.shape
    out = np.zeros((T, R, C, S, 1), dtype=data.dtype)
    for t in range(T):
        out[t, ..., 0] = data[..., t]
    return out

def min_max_normalize_4d(data: np.ndarray) -> np.ndarray:
    """
    Performs min-max normalization on each time-slice individually.
    Assumes shape (T, R, C, S, 1).
    """
    T, R, C, S, _ = data.shape
    for t in range(T):
        mini = np.amin(data[t])
        maxi = np.amax(data[t])
        if maxi > mini:
            data[t] = (data[t] - mini) / (maxi - mini)
        else:
            data[t] = 0
    return data

def percentile_rescale(data: np.ndarray, p_lower=5, p_upper=95) -> np.ndarray:
    """
    Rescales intensities to the [p_lower, p_upper] percentile range
    for each time-slice. Assumes shape (T, R, C, S, 1).
    """
    T = data.shape[0]
    for t in range(T):
        p_low_val = np.percentile(data[t], p_lower)
        p_high_val = np.percentile(data[t], p_upper)
        data[t] = exposure.rescale_intensity(
            data[t],
            in_range=(p_low_val, p_high_val)
        )
    return data

def preprocess_nifti_for_inference(nifti_path: str) -> np.ndarray:
    """
    Combines all steps:
      1. Load a 4D NIfTI file
      2. Downsample to (192,192,64,20)
      3. Reorder to (20,192,192,64,1)
      4. Min-max normalize each time-slice
      5. Rescale intensities with [5,95] percentile range
    Returns the preprocessed data ready for inference.
    """
    data_4d = load_nifti(nifti_path)
    downsampled = downsample_4d_volume(data_4d, (192, 192, 64, data_4d.shape[3]))
    reordered = reorder_4d_volume(downsampled)
    normalized = min_max_normalize_4d(reordered)
    rescaled = percentile_rescale(normalized, p_lower=5, p_upper=95)
    return rescaled