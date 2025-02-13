import os
from tqdm import tqdm
from auto_flow_pipeline.data_io.parse_dicom_files import parse_patient

def parse_all_dicoms(dicom_folder_path, output_folder_path, overwrite=False):
    """
    Iterates through all directories in the dicom_folder_path and uses the parse_patient function to parse each patient to the output folder.
    
    Parameters:
    dicom_folder_path (str): Path to the folder containing DICOM files.
    output_folder_path (str): Folder to save the parsed DICOM information.
    overwrite (bool): Flag to control whether to overwrite existing patient folders.
    """
    # Get the list of patient directories
    patient_dirs = [d for d in os.listdir(dicom_folder_path) if os.path.isdir(os.path.join(dicom_folder_path, d))]
    
    # Iterate through all directories in the dicom_folder_path with a progress bar
    for i, patient_id in enumerate(tqdm(patient_dirs, desc="Parsing all patients", position=0)):
        patient_path = os.path.join(dicom_folder_path, patient_id)
        if os.path.isdir(patient_path):
            tqdm.write(f"Parsing patient: {patient_id}")
            parse_patient(patient_id, dicom_folder_path, output_folder_path, overwrite, position=i+1)

if __name__ == "__main__":
    dicom_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    parse_all_dicoms(dicom_folder_path, output_folder_path, overwrite=False)
