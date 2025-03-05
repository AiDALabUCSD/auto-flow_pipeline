import os
import nibabel as nib
import numpy as np
from scipy.ndimage import zoom
from auto_flow_pipeline.visualization.postprocessing.segnet.reverse_preprocessing.generate_gifs import generate_combined_gif_with_segmentation
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def reverse_crop_zoom_and_reorder(seg_nii, mag_nii, 
                                  crop_center_x: int = 128, crop_center_y: int = 128, 
                                  crop_dim: int = 128, logger=None) -> np.ndarray:
    """
    Reverse the center cropping and zooming applied to segmentations and reorder axes.

    Parameters:
        seg_nii (nib.Nifti1Image): Segmentation NIfTI image.
        mag_nii (nib.Nifti1Image): Magnitude NIfTI image to get original shape.
        crop_center_x (int): X-center for cropping.
        crop_center_y (int): Y-center for cropping.
        crop_dim (int): Crop dimension used in preprocessing.
    
    Returns:
        np.ndarray: Segmentation mask aligned with the original image, shape (r, c, slices, t).
    """
    if logger:
        logger.info("Reversing center cropping and zooming for segmentations")

    segmentation = seg_nii.get_fdata()
    original_shape = mag_nii.shape

    t, cropped_r, cropped_c, slices = segmentation.shape
    original_r, original_c, original_slices, original_t = original_shape

    # Step 1: Resize back to cropped size
    resized_seg = np.zeros((t, crop_dim, crop_dim, slices), dtype=segmentation.dtype)

    for i in range(t):
        resized_seg[i] = zoom(segmentation[i], (crop_dim / cropped_r, crop_dim / cropped_c, 1), order=0)  # Nearest neighbor

    # Step 2: Place back into original size
    full_size_seg = np.zeros((t, original_r, original_c, original_slices), dtype=segmentation.dtype)

    l_x = crop_center_x - crop_dim // 2
    r_x = crop_center_x + crop_dim // 2
    l_y = crop_center_y - crop_dim // 2
    r_y = crop_center_y + crop_dim // 2

    full_size_seg[:, l_y:r_y, l_x:r_x, :] = resized_seg  # Place back in original location

    # Step 3: Reorder axes from (t, r, c, slices) -> (r, c, slices, t)
    aligned_segmentation = np.moveaxis(full_size_seg, 0, -1)

    if logger:
        logger.info("Reordering completed, segmentation is now aligned with original data")

    return aligned_segmentation

def reverse_segmentation(patient_name: str, base_path: str, 
                         crop_center_x: int = 128, crop_center_y: int = 128, 
                         crop_dim: int = 128, logger=None) -> None:
    """
    Load aortic and pulmonary segmentations and magnitude NIfTIs, reverse the segmentations, and save the results.

    Parameters:
        patient_name (str): Patient identifier.
        base_path (str): Base folder path containing patient directories.
        crop_center_x (int): X-center for cropping.
        crop_center_y (int): Y-center for cropping.
        crop_dim (int): Crop dimension used in preprocessing.
    """
    def reverse_and_save(seg_path, mag_path, flow_path, output_path, logger):
        seg_nii = nib.load(seg_path)
        mag_nii = nib.load(mag_path)
        flow_nii = nib.load(flow_path)
        reversed_segmentation = reverse_crop_zoom_and_reorder(seg_nii, mag_nii, crop_center_x, crop_center_y, crop_dim, logger)
        reversed_nii = nib.Nifti1Image(reversed_segmentation, mag_nii.affine)
        nib.save(reversed_nii, output_path)
        if logger:
            logger.info(f"Reversed segmentation saved to {output_path}")
        return reversed_segmentation, mag_nii, flow_nii

    if logger:
        logger.info("Starting to reverse segmentations for patient %s", patient_name)

    # Aortic segmentation
    aortic_seg_path = os.path.join(base_path, patient_name, 'segnet_aorta-pred_processed.nii.gz')
    aortic_mag_path = os.path.join(base_path, patient_name, 'aortic_spline_mag.nii.gz')
    aortic_flow_path = os.path.join(base_path, patient_name, 'aortic_spline_flow-through.nii.gz')
    aortic_output_path = os.path.join(base_path, patient_name, 'segnet_aorta_segmentation.nii.gz')
    aortic_reversed_seg, aortic_mag_nii, aortic_flow_nii = reverse_and_save(aortic_seg_path, aortic_mag_path,aortic_flow_path, aortic_output_path, logger)

    # Generate GIF for aortic segmentation
    logger.info("Generating GIF for aortic segmentation")
    generate_combined_gif_with_segmentation(aortic_mag_nii.get_fdata(), aortic_flow_nii.get_fdata(), aortic_reversed_seg, os.path.join(base_path, patient_name, 'aorta_segmentation.gif'), value_range=(-1500, 1500))
    logger.info("GIF for aortic segmentation generated")

    # Pulmonary segmentation
    pulmonary_seg_path = os.path.join(base_path, patient_name, 'segnet_pulmonary-pred_processed.nii.gz')
    pulmonary_mag_path = os.path.join(base_path, patient_name, 'pulmonary_spline_mag.nii.gz')
    pulmonary_flow_path = os.path.join(base_path, patient_name, 'pulmonary_spline_flow-through.nii.gz')
    pulmonary_output_path = os.path.join(base_path, patient_name, 'segnet_pulmonary_segmentation.nii.gz')
    pulmonary_reversed_seg, pulmonary_mag_nii, pulmonary_flow_nii = reverse_and_save(pulmonary_seg_path, pulmonary_mag_path, pulmonary_flow_path, pulmonary_output_path, logger)

    # Generate GIF for pulmonary segmentation
    logger.info("Generating GIF for pulmonary segmentation")
    generate_combined_gif_with_segmentation(pulmonary_mag_nii.get_fdata(), pulmonary_flow_nii.get_fdata(), pulmonary_reversed_seg, os.path.join(base_path, patient_name, 'pulmonary_segmentation.gif'), value_range=(-1500, 1500))
    logger.info("GIF for pulmonary segmentation generated")

    if logger:
        logger.info("Reversed segmentations for patient %s completed", patient_name)

def main():
    patient_name = "Bepemhir"  # Update this patient name as needed
    base_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"  # Update this path as needed
    logger = setup_logger(patient_name, base_path)
    
    reverse_segmentation(patient_name, base_path, logger=logger)

if __name__ == "__main__":
    main()
