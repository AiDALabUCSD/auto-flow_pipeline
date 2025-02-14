import os
import shutil
import pandas as pd
import pydicom
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
from auto_flow_pipeline.data_io.logging_setup import setup_logger


def is_dicom_file(file_path, logger):
    """
    Checks if a file is a DICOM file by reading its content.
    
    Parameters:
    file_path (str): Path to the file.
    logger (logging.Logger): Logger instance.
    
    Returns:
    bool: True if the file is a DICOM file, False otherwise.
    """
    try:
        with open(file_path, 'rb') as file:
            file.seek(128)
            magic = file.read(4)
            return magic == b'DICM'
    except Exception as e:
        logger.error(f"Error reading {file_path}: {e}")
        return False

def parse_dicom_file(dicom_path, logger):
    """
    Parses a single DICOM file to extract relevant information.

    Parameters:
        dicom_path (str): Path to the DICOM file.
        logger (logging.Logger): Logger instance.

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
            'InstanceNumber': dicom.InstanceNumber,  # [0x0020,0x0013]
            'ImagePositionPatient': dicom.ImagePositionPatient,  # [0x0020,0x0032]
            'ImageOrientationPatient': dicom.ImageOrientationPatient,  # [0x0020,0x0037]
            'PixelSpacing': dicom.PixelSpacing,  # [0x0028,0x0030]
            'SliceThickness': dicom.SliceThickness,  # [0x0018,0x0050]
            'Tag_0019_10B3': dicom.get((0x0019, 0x10B3), 'N/A').value if (0x0019, 0x10B3) in dicom else 'N/A',
            'Tag_0043_1030': dicom.get((0x0043, 0x1030), 'N/A').value if (0x0043, 0x1030) in dicom else 'N/A',
            'NumberOfTemporalPositions': dicom.get((0x0020, 0x0105), 'N/A').value
                                        if (0x0020, 0x0105) in dicom else 'N/A'
        }
        return info
    except Exception as e:
        logger.error(f"Error reading {dicom_path}: {e}")
        return None

def process_file(args):
    dicom_path, logger = args
    if is_dicom_file(dicom_path, logger) or dicom_path.endswith('.dcm'):
        return parse_dicom_file(dicom_path, logger)
    return None

def parse_dicom_folder(folder_path, logger, position=1):
    """
    Iterates through all DICOM files in a specified folder and extracts relevant information.
    
    Parameters:
    folder_path (str): Path to the folder containing DICOM files.
    logger (logging.Logger): Logger instance.
    position (int): Position of the progress bar for nested progress bars.
    
    Returns:
    pd.DataFrame: A DataFrame containing the extracted information from all DICOM files.
    """
    dicom_info_list = []
    file_paths = []

    for root, _, files in os.walk(folder_path):
        for file in files:
            dicom_path = os.path.join(root, file)
            file_paths.append((dicom_path, logger))

    with Pool(cpu_count()) as pool:
        for dicom_info in tqdm(pool.imap_unordered(process_file, file_paths), total=len(file_paths), desc="Processing DICOM files", position=position):
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

def identify_4_real_series(df):
    """
    Identify exactly 4 SeriesInstanceUIDs that:
      - Each have a single Tag_0043_1030 value,
      - They share an identical mapping of (InstanceNumber -> ImagePositionPatient),
      - Taken together, they have 4 distinct Tag_0043_1030 values (e.g. {2,3,4,5}).

    Returns a filtered DataFrame containing only those 4 series.
    """

    # 1. Build a dictionary or "signature" for each SeriesInstanceUID:
    #    - The single Tag_0043_1030 for that series,
    #    - The mapping of (InstanceNumber -> unique ImagePositionPatient).

    series_map = {}  # { series_uid: (flow_tag, dict_of_inst2pos) }

    for uid, group in df.groupby('SeriesInstanceUID'):
        # Because each series has exactly one Tag_0043_1030, confirm or pick the first unique value
        flow_tags = group['Tag_0043_1030'].unique()
        if len(flow_tags) != 1:
            # This implies the series is inconsistent or not what we expect;
            # skip or handle differently
            continue
        flow_tag = flow_tags[0]

        # Build a dict mapping InstanceNumber -> unique ImagePositionPatient
        inst2pos = {}
        for inst_num, inst_subset in group.groupby('InstanceNumber'):
            # We expect exactly 1 unique image position for this instance
            positions = inst_subset['ImagePositionPatient'].unique()
            if len(positions) == 1:
                inst2pos[inst_num] = positions[0]
            else:
                # If there's more than one position for the same instance, it's inconsistent
                # We won't keep it
                pass

        series_map[uid] = (flow_tag, inst2pos)

    # 2. Convert that map into a DataFrame we can group by "signature."
    #    A "signature" is how we compare the coverage among different series.

    # We'll convert the inst2pos dict into a stable tuple so it can be compared or hashed
    def dict_to_tuple(d):
        # Sort by instance number so the order is consistent
        return tuple(sorted(d.items()))  # e.g., ((1, [x1,y1,z1]), (2, [x2,y2,z2]), ...)

    rows = []
    for uid, (flow_tag, inst2pos) in series_map.items():
        signature = dict_to_tuple(inst2pos)
        rows.append((uid, flow_tag, signature))

    df_signatures = pd.DataFrame(rows, columns=['uid', 'flow_tag', 'signature'])

    # 3. Group by 'signature' so that all series with the same (InstanceNumber -> Position) coverage
    #    end up together. Then check if that group has exactly 4 distinct flow_tags (2,3,4,5).

    final_uids = set()

    for signature_value, subset in df_signatures.groupby('signature'):
        flow_tags = set(subset['flow_tag'].tolist())

        # If these 4 series match up in coverage and have exactly the four distinct flow_tags:
        # e.g. {2, 3, 4, 5}.  Adjust if your actual "flow tags" differ.
        if flow_tags == {2, 3, 4, 5} and len(subset) == 4:
            # Add these to our final keep list
            final_uids.update(subset['uid'].tolist())

    # 4. Now filter the original DataFrame to only these "real" 4 series
    df_filtered = df[df['SeriesInstanceUID'].isin(final_uids)].copy()
    return df_filtered

def filter_and_save_4d_flow(data_path, logger):
    """
    Loads a CSV or pickle file, keeps only 4D flow rows, 
    adds time_index and slice_index columns based on InstanceNumber,
    saves the shape of the corresponding .npy file,
    and writes them as 'flow_info.csv' and 'flow_info.pkl' 
    in the same directory.

    A file is considered 4D flow if:
      Tag_0019_10B3 > 1
      OR
      (Tag_0043_1030 > 1 AND Tag_0043_1030 < 6)

    time_index = (InstanceNumber - 1) % TDIM
    slice_index = (InstanceNumber - 1) // TDIM

    TDIM is read from the .npy file in the velocities folder, named "<patient-name>.npy".

    Args:
        data_path (str): Path to the CSV or pickle file with DICOM info.
        logger (logging.Logger): Logger instance.
    """
    # 1. Load the DICOM info DataFrame
    _, ext = os.path.splitext(data_path)
    if ext.lower() in ('.pkl', '.pickle'):
        df = pd.read_pickle(data_path)
    else:
        df = pd.read_csv(data_path)

    # 2. Filter for 4D flow
    mask_4d_flow = (
        (df['Tag_0019_10B3'].astype(float) > 1) |
        ((df['Tag_0043_1030'].astype(float) > 1) & (df['Tag_0043_1030'].astype(float) < 6))
    )
    df_4d = df[mask_4d_flow].copy()

    # 3. Determine patient name from the path (assuming the folder name is the patient)
    folder = os.path.dirname(data_path)
    patient_name = os.path.basename(folder)

    # 4. Load the corresponding .npy file to get TDIM and track shape
    velocities_dir = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities"
    vel_filepath = os.path.join(velocities_dir, f"{patient_name}.npy")

    # Default values if missing
    TDIM = 1
    vel_shape = None

    if os.path.exists(vel_filepath):
        tempnpy = np.load(vel_filepath)
        TDIM = tempnpy.shape[0]
        vel_shape = tempnpy.shape
    else:
        logger.warning(f"{vel_filepath} not found. Using TDIM=1 and no shape info.")

    # 5. Compute time_index and slice_index from InstanceNumber (subtracting 1 if 1-based)
    if 'InstanceNumber' in df_4d.columns:
        instance_nums = df_4d['InstanceNumber'].astype(int) - 1
        df_4d['time_index'] = instance_nums % TDIM
        df_4d['slice_index'] = instance_nums // TDIM
    else:
        df_4d['time_index'] = np.nan
        df_4d['slice_index'] = np.nan

    # 6. Remove extraneous images
    df_4d = identify_4_real_series(df_4d)

    # 7. Store the numpy shape in a new column
    df_4d['vel_npy_shape'] = str(vel_shape) if vel_shape is not None else "N/A"

    # 8. Save the filtered DataFrame with new columns as "flow_info"
    base = os.path.join(folder, "flow_info")
    csv_out = base + ".csv"
    pkl_out = base + ".pkl"

    df_4d.to_csv(csv_out, index=False)
    df_4d.to_pickle(pkl_out)

    logger.info(f"4D flow (with time_index, slice_index, vel_npy_shape) saved to:\n{csv_out}\n{pkl_out}")
    return df_4d

def parse_patient(pid, dicom_folder_path, output_folder_path, overwrite=False, position=1):
    """
    Main function to parse DICOM files for a specific patient and save the extracted information to both a CSV file and a pickle file.
    
    Parameters:
    pid (str): Patient ID.
    dicom_folder_path (str): General path to the folder containing DICOM files.
    output_folder_path (str): Folder to save the CSV and pickle files.
    overwrite (bool): Flag to control whether to overwrite existing patient folder.
    position (int): Position of the progress bar for nested progress bars.
    """
    # Construct the full path to the patient's DICOM folder
    patient_dicom_folder = os.path.join(dicom_folder_path, pid)
    patient_output_folder = os.path.join(output_folder_path, pid)

    # If the patient folder already exists and we're overwriting, remove it
    if os.path.exists(patient_output_folder):
        if overwrite:
            shutil.rmtree(patient_output_folder)
            os.makedirs(patient_output_folder, exist_ok=True)
        else:
            # If we're not overwriting, just log and return
            # (or do something else, as desired)
            temp_logger = setup_logger(
                patient_name=pid,
                output_folder=output_folder_path
            )
            temp_logger.info(f"Folder {patient_output_folder} already exists. Skipping...")
            return
    else:
        os.makedirs(patient_output_folder, exist_ok=True)

    logger = setup_logger(pid, output_folder_path)
    logger.info(f"Starting parsing patient: {pid}")
    
    # Parse and save full DICOM info
    dicom_info_df = parse_dicom_folder(patient_dicom_folder, logger, position=position)
    save_dicom_info(dicom_info_df, patient_output_folder)
    logger.info(f"DICOM information saved to {patient_output_folder} as dicom_info.csv/pkl")

    # 4D flow filtering step
    dicom_info_csv = os.path.join(patient_output_folder, "dicom_info.csv")
    filter_and_save_4d_flow(dicom_info_csv, logger)

    logger.info(f"4D flow information saved to {patient_output_folder} as flow_info.csv/pkl")
    logger.info(f"Finished parsing patient: {pid}")

# Example usage
if __name__ == "__main__":
    pid = 'Ackoram'
    dicom_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_folder_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    parse_patient(pid, dicom_folder_path, output_folder_path, overwrite=True)