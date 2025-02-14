import os
from tqdm import tqdm
from auto_flow_pipeline.data_io.dicom_to_nifti import patient_to_nifti

def process_all_patients(dicom_base_folder, output_base_folder, velocity_base_folder, max_patients=-1, overwrite=False):
    """
    Iterates through all directories in the dicom_base_folder and uses the
    patient_to_nifti function to process each patient to the output folder.

    Parameters:
        dicom_base_folder (str): Path to the folder containing DICOM files.
        output_base_folder (str): Folder to save the processed NIfTI files.
        velocity_base_folder (str): Folder containing the velocity numpy files.
        max_patients (int): Maximum number of patients to process. Default is -1 (process all patients).
        overwrite (bool): Whether to overwrite existing NIfTI files. Default is False.
    """
    # Get the list of patient directories
    patient_dirs = [
        d for d in os.listdir(dicom_base_folder)
        if os.path.isdir(os.path.join(dicom_base_folder, d))
    ]

    # Limit the number of patients to process if max_patients is not -1
    if max_patients != -1:
        patient_dirs = patient_dirs[:max_patients]

    # Iterate through all directories with a progress bar
    for patient_id in tqdm(patient_dirs, desc="Processing all patients", position=0):
        patient_path = os.path.join(dicom_base_folder, patient_id)
        if os.path.isdir(patient_path):
            # Process each patient
            patient_to_nifti(
                pid=patient_id,
                base_dicom_folder=dicom_base_folder,
                base_output_folder=output_base_folder,
                base_velocity_folder=velocity_base_folder,
                overwrite=overwrite
            )

def main():
    dicom_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    velocity_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities'
    process_all_patients(dicom_base_folder, output_base_folder, velocity_base_folder, max_patients=-1, overwrite=False)

if __name__ == "__main__":
    main()
