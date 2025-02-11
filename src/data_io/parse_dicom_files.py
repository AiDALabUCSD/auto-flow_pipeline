import os
import shutil
import pandas as pd
import pydicom
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

def is_dicom_file(file_path):
    """
    Checks if a file is a DICOM file by reading its content.
    
    Parameters:
    file_path (str): Path to the file.
    
    Returns:
    bool: True if the file is a DICOM file, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            file.seek(128)
            magic = file.read(4)
            return magic == b'DICM'
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False

def parse_dicom_file(dicom_path):
    """
    Parses a single DICOM file to extract relevant information.
    
    Parameters:
    dicom_path (str): Path to the DICOM file.
    
    Returns:
    dict: A dictionary containing the extracted information.
    """
    try:
        dicom = pydicom.dcmread(dicom_path)
        info = {
            'FilePath': dicom_path,
            'PatientID': dicom.PatientID,
            'StudyInstanceUID': dicom.StudyInstanceUID,
            'SeriesInstanceUID': dicom.SeriesInstanceUID,
            'SOPInstanceUID': dicom.SOPInstanceUID,
            'Modality': dicom.Modality,
            'StudyDate': dicom.StudyDate,
            'SeriesDescription': dicom.SeriesDescription,
            'InstanceNumber': dicom.InstanceNumber, # [0x0020,0x0013]
            'ImagePositionPatient': dicom.ImagePositionPatient, # [0x0020,0x0032]
            'ImageOrientationPatient': dicom.ImageOrientationPatient, # [0x0020,0x0037]
            'PixelSpacing': dicom.PixelSpacing, # [0x0028,0x0030]
            'SliceThickness': dicom.SliceThickness, # [0x0018,0x0050]
            'Tag_0019_10B3': dicom.get((0x0019, 0x10B3), 'N/A').value if (0x0019, 0x10B3) in dicom else 'N/A',
            'Tag_0043_1030': dicom.get((0x0043, 0x1030), 'N/A').value if (0x0043, 0x1030) in dicom else 'N/A'
        }
        return info
    except Exception as e:
        print(f"Error reading {dicom_path}: {e}")
        return None

def process_file(args):
    dicom_path, = args
    if is_dicom_file(dicom_path) or dicom_path.endswith('.dcm'):
        return parse_dicom_file(dicom_path)
    return None

def parse_dicom_folder(folder_path):
    """
    Iterates through all DICOM files in a specified folder and extracts relevant information.
    
    Parameters:
    folder_path (str): Path to the folder containing DICOM files.
    
    Returns:
    pd.DataFrame: A DataFrame containing the extracted information from all DICOM files.
    """
    dicom_info_list = []
    file_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            dicom_path = os.path.join(root, file)
            file_paths.append((dicom_path,))

    with Pool(cpu_count()) as pool:
        for dicom_info in tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths), desc="Processing DICOM files"):
            if dicom_info:
                dicom_info_list.append(dicom_info)

    return pd.DataFrame(dicom_info_list)

def save_dicom_info(dicom_info_df, output_folder):
    """
    Saves the DICOM information DataFrame to both a CSV file and a pickle file.
    
    Parameters:
    dicom_info_df (pd.DataFrame): DataFrame containing the DICOM information.
    output_folder (str): Folder to save the CSV and pickle files.
    """
    csv_path = os.path.join(output_folder, 'dicom_info.csv')
    pickle_path = os.path.join(output_folder, 'dicom_info.pkl')
    
    dicom_info_df.to_csv(csv_path, index=False)
    dicom_info_df.to_pickle(pickle_path)

def main(dicom_folder, output_folder, overwrite=False):
    """
    Main function to parse DICOM files in a folder and save the extracted information to both a CSV file and a pickle file.
    
    Parameters:
    dicom_folder (str): Path to the folder containing DICOM files.
    output_folder (str): Folder to save the CSV and pickle files.
    overwrite (bool): Flag to control whether to overwrite existing patient folder.
    """
    # Extract patient name from the lowest level folder in dicom_folder
    patient_name = os.path.basename(os.path.normpath(dicom_folder))
    patient_output_folder = os.path.join(output_folder, patient_name)
    
    # Debugging statements
    # print(f"Checking if the patient output folder exists: {patient_output_folder}")
    # print(f"Absolute path: {os.path.abspath(patient_output_folder)}")
    # print(f"Exists: {os.path.exists(patient_output_folder)}")
    # print(f"Is directory: {os.path.isdir(patient_output_folder)}")
    # print(f"Is file: {os.path.isfile(patient_output_folder)}")
    
    # Check if the patient folder exists
    if os.path.exists(patient_output_folder):
        if overwrite:
            shutil.rmtree(patient_output_folder)
            os.makedirs(patient_output_folder)
        else:
            print(f"Folder {patient_output_folder} already exists. Skipping...")
            return  # Exit the function early if the folder exists and overwrite is False
    else:
        os.makedirs(patient_output_folder)
    
    dicom_info_df = parse_dicom_folder(dicom_folder)
    save_dicom_info(dicom_info_df, patient_output_folder)
    print(f"DICOM information saved to {patient_output_folder}")

# Example usage
if __name__ == "__main__":
    dicom_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/Cakimtol'
    output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    main(dicom_folder, output_folder, overwrite=True)