from typing import Dict
import os
import nibabel as nib
import numpy as np
from skimage import exposure
from scipy.ndimage import zoom
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.visualization.preprocessing.segnet.prepare_for_segnet.generate_gifs import generate_two_row_gifs_for_slices

def load_spline(patient_name: str, base_path: str, spline_type: str, logger) -> Dict[str, nib.Nifti1Image]:
    """
    Load spline data (either aortic or pulmonary) for a given patient.
    
    This function expects the following files in the patient folder under base_path:
    
    For "aortic":
      - "aortic_spline_mag.nii.gz"
      - "aortic_spline_flow-through.nii.gz"
    
    For "pulmonary":
      - "pulmonary_spline_mag.nii.gz"
      - "pulmonary_spline_flow-through.nii.gz"
    
    Parameters:
        patient_name (str): Patient identifier.
        base_path (str): Base folder path containing patient directories.
        spline_type (str): Type of spline to load ("aortic" or "pulmonary").
    
    Returns:
        dict: A dictionary with keys 'mag' and 'through_flow' holding the
              corresponding nibabel image objects.
    
    Raises:
        ValueError: If an invalid spline_type is provided.
    """
    logger.info("Loading %s spline data for patient %s", spline_type, patient_name)
    patient_folder = os.path.join(base_path, patient_name)
    spline_type = spline_type.lower()
    
    if spline_type == "aortic":
        mag_filename = "aortic_spline_mag.nii.gz"
        flow_filename = "aortic_spline_flow-through.nii.gz"
    elif spline_type == "pulmonary":
        mag_filename = "pulmonary_spline_mag.nii.gz"
        flow_filename = "pulmonary_spline_flow-through.nii.gz"
    else:
        logger.error("Invalid spline_type: %s", spline_type)
        raise ValueError("spline_type must be either 'aortic' or 'pulmonary'.")
    
    mag_path = os.path.join(patient_folder, mag_filename)
    flow_path = os.path.join(patient_folder, flow_filename)
    
    try:
        spline_mag = nib.load(mag_path)
        spline_through_flow = nib.load(flow_path)
        logger.info("%s spline data loaded successfully for patient %s", spline_type.capitalize(), patient_name)
    except Exception as e:
        logger.exception("Error loading %s spline data for patient %s", spline_type, patient_name)
        raise e
    
    return {
        'mag': spline_mag,
        'through_flow': spline_through_flow
    }

def compose_images(mag: np.ndarray, through_flow: np.ndarray, logger) -> np.ndarray:
    """
    Compose preprocessed images from mag and through_flow arrays.
    
    Parameters:
        mag (np.ndarray): Array of shape (r, c, slices, t) representing magnitude data.
        through_flow (np.ndarray): Array of shape (r, c, slices, t) representing through flow data.
    
    Returns:
        np.ndarray: Composed array of shape (t, r, c, slices, 6) where channels are:
            0: mag at t-1 (wrapped around),
            1: mag at t,
            2: mag at t+1 (wrapped around),
            3: through_flow at time index 3,
            4: through_flow at time index 4,
            5: through_flow at time index 5.
    """
    logger.info("Composing images from magnitude and through_flow arrays")
    r, c, num_slices, T = mag.shape
    composed = np.zeros((T, r, c, num_slices, 6), dtype=mag.dtype)
    
    for i in range(T):
        composed[i, :, :, :, 0] = mag[:, :, :, (i - 1) % T]  # wrap t-1
        composed[i, :, :, :, 1] = mag[:, :, :, i]
        composed[i, :, :, :, 2] = mag[:, :, :, (i + 1) % T]  # wrap t+1
        # through_flow channels: select slices at indices 2, 3, 4
        composed[i, :, :, :, 3:6] = through_flow[:, :, :, 2:5]
    
    logger.info("Images composed successfully")
    return composed

def dynamic_range(img: np.ndarray, percentile: float, vel_cap: int, logger) -> np.ndarray:
    """
    Perform dynamic range scaling on a composed image array.

    The image is assumed to have the following channel organization:
        - Channels 0, 1, 2 correspond to magnitude data.
        - Channels 3, ... correspond to through_flow (velocity) data.

    Magnitude channels are scaled to the range [0, 1] based on the lower and upper
    percentile values computed from the first three channels. Velocity channels are
    scaled to the range [-1, 1] based on the provided vel_cap.

    Parameters:
        img (np.ndarray): The input image array (e.g., from compose_images) with shape
                          (T, r, c, slices, channels).
        percentile (float): The percentile (e.g., 95) used to determine the scaling range 
                            for the magnitude channels.
        vel_cap (int): The maximum absolute value for scaling velocity channels (default is 1500).

    Returns:
        np.ndarray: The dynamically scaled image.
    """
    logger.info("Applying dynamic range scaling")
    scaled_img = np.zeros_like(img)

    try:
        # Determine intensity range for magnitude channels using provided percentile.
        pL = np.percentile(img[..., :3], 100 - percentile)
        pU = np.percentile(img[..., :3], percentile)
        
        # Scale magnitude channels from [pL, pU] to [0, 1].
        scaled_img[..., :3] = exposure.rescale_intensity(
            img[..., :3], in_range=(pL, pU), out_range=(0.0, 1.0)
        )
        
        # Scale through_flow (velocity) channels from [-vel_cap, vel_cap] to [-1, 1].
        scaled_img[..., 3:] = exposure.rescale_intensity(
            img[..., 3:], in_range=(-vel_cap, vel_cap), out_range=(-1.0, 1.0)
        )
        logger.info("Dynamic range scaling applied successfully")
    except Exception as e:
        logger.exception("Error applying dynamic range scaling")
        raise e
    
    return scaled_img

def center_crop_zoom(in_img: np.ndarray, center_x: int, center_y: int, dim: int, logger) -> np.ndarray:
    """
    Center crops and zooms each time slice of an input image.
    
    For each time slice in in_img (assumed shape (T, r, c, slices, channels)),
    this function crops a region of size (dim, dim) centered at (center_x, center_y)
    in the spatial dimensions, then zooms the crop back to the original spatial size.
    
    Parameters:
        in_img (np.ndarray): Input image array with shape (T, r, c, slices, channels).
        center_x (int): X-coordinate of the crop center.
        center_y (int): Y-coordinate of the crop center.
        dim (int): Dimension of the square crop.
    
    Returns:
        np.ndarray: Output image array after center cropping and zooming, with same shape as in_img.
    """
    logger.info("Applying center crop and zoom")
    try:
        l_x = center_x - dim // 2
        r_x = center_x + dim // 2
        l_y = center_y - dim // 2
        r_y = center_y + dim // 2
    except Exception:
        logger.warning("Invalid crop center coordinates provided, defaulting to (128, 128)")
        l_x = 128 - dim // 2
        r_x = 128 + dim // 2
        l_y = 128 - dim // 2
        r_y = 128 + dim // 2

    out_image = np.zeros_like(in_img)
    # Get target spatial dimensions from in_img shape (assume in_img has shape: (T, r, c, slices, channels)).
    target_r = in_img.shape[1]
    target_c = in_img.shape[2]
    
    try:
        for t in range(in_img.shape[0]):
            # Crop the image at time t
            temp = in_img[t, l_y:r_y, l_x:r_x, :]
            # Compute zoom factors so that the crop is resized to the original spatial dimensions.
            zoom_factor_r = target_r / temp.shape[0]
            zoom_factor_c = target_c / temp.shape[1]
            # Leave the slices and channel dimensions unchanged (factors = 1).
            temp_zoomed = zoom(temp, (zoom_factor_r, zoom_factor_c, 1, 1), order=1)
            out_image[t, ...] = temp_zoomed
        logger.info("Center crop and zoom applied successfully")
    except Exception as e:
        logger.exception("Error applying center crop and zoom")
        raise e
    
    return out_image

def preprocess_for_segnet(logger,
    mag: np.ndarray,
    through_flow: np.ndarray,
    percentile: float = 95,
    vel_cap: int = 1500,
    crop_center_x: int = 128,
    crop_center_y: int = 128,
    crop_dim: int = 128
) -> np.ndarray:
    """
    Perform full preprocessing on magnitude and through_flow arrays for SegNet.
    
    This function executes the following steps:
      1. Composes the images into a single array using the provided channel conventions.
      2. Applies dynamic range scaling:
          - Magnitude channels are scaled to [0, 1] based on the provided percentile.
          - Through_flow (velocity) channels are scaled to [-1, 1] based on vel_cap.
      3. Performs a center crop on each time slice (of size crop_dim x crop_dim)
         and zooms the cropped region back to the original spatial dimensions.
    
    Parameters:
        mag (np.ndarray): Magnitude array of shape (r, c, slices, t).
        through_flow (np.ndarray): Through flow array of shape (r, c, slices, t).
        percentile (float): Percentile for scaling the magnitude channels (default 95).
        vel_cap (int): Maximum absolute value for scaling velocity channels (default 1500).
        crop_center_x (int): X-coordinate of the center for center cropping (default 128).
        crop_center_y (int): Y-coordinate of the center for center cropping (default 128).
        crop_dim (int): Dimension of the square crop (default 128).
    
    Returns:
        np.ndarray: Fully preprocessed image array with shape (t, r, c, slices, channels),
                    where t is the number of time slices.
    """
    logger.info("Starting full preprocessing for SegNet")
    try:
        # Step 1: Compose images.
        composed = compose_images(mag, through_flow, logger)
        
        # Step 2: Apply dynamic range scaling.
        scaled = dynamic_range(composed, percentile, vel_cap, logger)
        
        # Step 3: Center crop and zoom each time slice back to original spatial dims.
        preprocessed = center_crop_zoom(scaled, crop_center_x, crop_center_y, crop_dim, logger)
        logger.info("Full preprocessing completed successfully")
    except Exception as e:
        logger.exception("Error during full preprocessing for SegNet")
        raise e

    return preprocessed

def compose_and_save_splines(patient_name: str, base_path: str) -> None:
    logger = setup_logger(patient_name, base_path)
    logger.info("Starting to process splines for %s", patient_name)

    # Aorta
    logger.info("Loading aorta spline data...")
    try:
        aorta_data = load_spline(patient_name, base_path, "aortic", logger)
        aorta_mag = aorta_data["mag"].get_fdata()
        aorta_flow = aorta_data["through_flow"].get_fdata()
        logger.info("Aorta spline data loaded successfully.")
    except Exception as e:
        logger.exception("Error loading aorta spline data.")
        raise e

    logger.info("Preprocessing aorta data...")
    aorta_preprocessed = preprocess_for_segnet(logger, aorta_mag, aorta_flow, 95, 1500, 128, 128, 128)
    logger.info("Aorta data preprocessed. Saving results...")
    aorta_nii = nib.Nifti1Image(aorta_preprocessed, aorta_data["mag"].affine)
    nib.save(aorta_nii, os.path.join(base_path, patient_name, "aorta_spline_composed.nii.gz"))
    logger.info("Aorta preprocessed data saved as aorta_spline_composed.nii.gz")
    logger.info("Generating GIFs for aorta data...")
    generate_two_row_gifs_for_slices(
        patient_name,
        base_path,
        "aorta_spline_composed_gifs",
        aorta_preprocessed,
        duration=0.5,
        value_range=(-1, 1)
    )
    logger.info("GIFs generated for aorta data.")

    # Pulmonary
    logger.info("Loading pulmonary spline data...")
    try:
        pulmonary_data = load_spline(patient_name, base_path, "pulmonary", logger)
        pulmonary_mag = pulmonary_data["mag"].get_fdata()
        pulmonary_flow = pulmonary_data["through_flow"].get_fdata()
        logger.info("Pulmonary spline data loaded successfully.")
    except Exception as e:
        logger.exception("Error loading pulmonary spline data.")
        raise e

    logger.info("Preprocessing pulmonary data...")
    pulmonary_preprocessed = preprocess_for_segnet(logger, pulmonary_mag, pulmonary_flow, 95, 1500, 128, 128, 128)
    logger.info("Pulmonary data preprocessed. Saving results...")
    pulmonary_nii = nib.Nifti1Image(pulmonary_preprocessed, pulmonary_data["mag"].affine)
    nib.save(pulmonary_nii, os.path.join(base_path, patient_name, "pulmonary_spline_composed.nii.gz"))
    logger.info("Pulmonary preprocessed data saved as pulmonary_spline_composed.nii.gz")
    logger.info("Generating GIFs for pulmonary data...")
    generate_two_row_gifs_for_slices(
        patient_name,
        base_path,
        "pulmonary_spline_composed_gifs",
        pulmonary_preprocessed,
        duration=0.5,
        value_range=(-1, 1)
    )
    logger.info("GIFs generated for pulmonary data.")

def main():
    patient_name = "Bulosul"
    base_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    compose_and_save_splines(patient_name, base_path)

if __name__ == "__main__":
    main()