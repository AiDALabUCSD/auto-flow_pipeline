import os
import numpy as np
import nibabel as nib
import pandas as pd
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.data_io.catalogue_patients import load_patient_catalogue, get_slice_diff
from auto_flow_pipeline import main_logger

def fix_affine_and_save_nifti(patient_name, base_folder_path):
    """
    Loads the mag_4dflow.nii.gz for the given patient from its archive folder,
    pulls the affine and the image data, corrects the affine to reflect the z-axis flip if needed,
    saves the original affine in qform and the updated affine in sform,
    and then saves the image back to the original folder as mag_4dflow.nii.gz.
    
    Args:
        patient_name (str): The name of the patient.
        base_folder_path (str): The base folder path where the patient's data is stored.
    """
    logger = setup_logger(patient_name, base_folder_path)

    parent_folder_path = os.path.dirname(base_folder_path)
    archive_folder = os.path.join(parent_folder_path, 'archive', patient_name)
    os.makedirs(archive_folder, exist_ok=True)
    source_path = os.path.join(archive_folder, 'mag_4dflow.nii.gz')
    destination_path = os.path.join(base_folder_path, patient_name, 'mag_4dflow.nii.gz')
    
    if not os.path.exists(source_path):
        logger.warning(f"File {source_path} does not exist")
        return
    
    # Load the NIfTI file
    nifti_img = nib.load(source_path)
    original_affine = nifti_img.affine
    img_data = nifti_img.get_fdata()
    
    # Load the patient catalogue
    catalogue_df = load_patient_catalogue()
    
    # Get slice_diff for the patient
    slice_diff = get_slice_diff(patient_name, catalogue_df)
    
    if slice_diff > 0:
        # Correct the affine to reflect the z-axis flip
        z_flip_affine = np.diag([1, 1, -1, 1])
        z_flip_affine[2, 3] = img_data.shape[2] - 1  # Adjust the translation component
        new_affine = original_affine @ z_flip_affine
        logger.info(f"Z-axis flip affine:\n{z_flip_affine}")
        logger.info(f"Affine corrected")
    else:
        # Use the identity matrix for the affine correction
        new_affine = original_affine
        logger.info(f"No correction needed")
    
    # Log the original and updated affine
    logger.info(f"Original affine:\n{original_affine}")
    logger.info(f"Updated affine:\n{new_affine}")

    # Log the shape of the image data
    logger.info(f"Image data shape: {img_data.shape}")

    # Create a new NIfTI image with the modified affine
    new_nifti_img = nib.Nifti1Image(img_data, new_affine)
    
    # Save the updated affine in qform and the original affine in sform
    new_nifti_img.set_qform(original_affine, code=1)
    new_nifti_img.set_sform(new_affine, code=1)
    
    # Save the new NIfTI image to the original folder
    nib.save(new_nifti_img, destination_path)
    logger.info(f"Saved fixed affine NIfTI to {destination_path}")

def get_cross_prod(patient_name, catalogue_df):
    """
    Gets the cross_prod for a given patient.
    
    Args:
        patient_name (str): The name of the patient.
        catalogue_df (pd.DataFrame): The patient catalogue dataframe.
    
    Returns:
        float: The cross_prod for the patient.
    """
    return float(catalogue_df.loc[catalogue_df['patient_id'] == patient_name, 'cross_prod'].values[0])

def load_and_print_affine(patient_name, base_folder_path):
    """
    Loads the new mag_4dflow.nii.gz for the given patient, reads the affine, and prints it. Used for debugging output of img.affine.
    
    Args:
        patient_name (str): The name of the patient.
        base_folder_path (str): The base folder path where the patient's data is stored.
    """
    logger = setup_logger(patient_name, base_folder_path)
    
    nifti_path = os.path.join(base_folder_path, patient_name, 'mag_4dflow.nii.gz')
    
    if not os.path.exists(nifti_path):
        logger.warning(f"File {nifti_path} does not exist")
        return
    
    # Load the NIfTI file
    nifti_img = nib.load(nifti_path)
    affine = nifti_img.affine
    
    # Print the affine
    logger.info(f"Affine for {patient_name}:\n{affine}")

def fix_affine_for_all_patients(base_folder_path):
    """
    Goes through all the patients in the base folder and runs fix_affine_and_save_nifti for them.
    
    Args:
        base_folder_path (str): The base folder path where the patients' data is stored.
    """
    patients = [name for name in os.listdir(base_folder_path) if os.path.isdir(os.path.join(base_folder_path, name))]
    
    for patient_name in patients:
        main_logger.info(f"Processing patient: {patient_name}")
        fix_affine_and_save_nifti(patient_name, base_folder_path)
        load_and_print_affine(patient_name, base_folder_path)

def main():
    base_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    
    fix_affine_for_all_patients(base_folder_path)

if __name__ == "__main__":
    main()
