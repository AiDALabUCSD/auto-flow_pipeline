import os
import nibabel as nib
import numpy as np
import pandas as pd
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.data_io.catalogue_patients import load_patient_catalogue, save_patient_catalogue

def calculate_flow(flow_nii, seg_nii, conversion_factor: float = (0.1)**3 * 1/1000, pixel_area: float = 1.0) -> pd.DataFrame:
    """
    Calculate the flow rate using the segmentation and spline_flow-through NIfTI images.

    Parameters:
        flow_nii (nib.Nifti1Image): Spline flow-through NIfTI image.
        seg_nii (nib.Nifti1Image): Segmentation NIfTI image.
        conversion_factor (float): Conversion factor to convert units from cm^3/s to L/s.
        pixel_area (float): Surface area of each pixel in mm^2.

    Returns:
        pd.DataFrame: DataFrame with rows as times and columns as slices, containing flow rates.
    """
    flow_data = flow_nii.get_fdata() * conversion_factor * pixel_area
    seg_data = seg_nii.get_fdata()

    # Multiply flow data by segmentation mask
    flow_rate_data = flow_data * seg_data

    # Sum the flow rates across rows and columns for each slice and time point
    flow_rate_sums = np.sum(np.sum(flow_rate_data, axis=0), axis=0)

    # Create DataFrame with times as rows and slices as columns
    times, slices = flow_rate_sums.shape
    flow_rate_df = pd.DataFrame(flow_rate_sums, index=range(times), columns=range(slices))

    return flow_rate_df

def main():
    patient_name = "Bepemhir"  # Update this patient name as needed
    base_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"  # Update this path as needed
    logger = setup_logger(patient_name, base_path)
    
    # Load the reversed segmentation and flow-through NIfTI images
    aortic_flow_nii = nib.load(os.path.join(base_path, patient_name, 'aortic_spline_flow-through.nii.gz'))
    aortic_seg_nii = nib.load(os.path.join(base_path, patient_name, 'segnet_aorta_segmentation.nii.gz'))
    
    # Calculate flow for aortic segmentation
    aortic_flow_df = calculate_flow(aortic_flow_nii, aortic_seg_nii)
    aortic_flow_df.to_csv(os.path.join(base_path, patient_name, 'aortic_flow_rates.csv'))
    
    pulmonary_flow_nii = nib.load(os.path.join(base_path, patient_name, 'pulmonary_spline_flow-through.nii.gz'))
    pulmonary_seg_nii = nib.load(os.path.join(base_path, patient_name, 'segnet_pulmonary_segmentation.nii.gz'))
    
    # Calculate flow for pulmonary segmentation
    pulmonary_flow_df = calculate_flow(pulmonary_flow_nii, pulmonary_seg_nii)
    pulmonary_flow_df.to_csv(os.path.join(base_path, patient_name, 'pulmonary_flow_rates.csv'))

if __name__ == "__main__":
    main()
