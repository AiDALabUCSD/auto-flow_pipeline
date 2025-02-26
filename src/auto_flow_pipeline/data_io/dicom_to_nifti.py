import os
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
import logging
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.visualization.dicom_to_nifti.generate_gifs import (
    generate_gif_from_nifti, generate_gif_from_nifti_vel,
    generate_gif_from_velocity_nifti, generate_gif_from_nifti_vel
    )

# Add optional imports for parallelization and progress bars
from joblib import Parallel, delayed
from tqdm import tqdm

def load_4dflow_dataframe(path_to_csv_or_pickle, logger):
    """
    Load the dataframe from a CSV or pickle file.
    
    Parameters:
    path_to_csv_or_pickle (str): Path to the CSV or pickle file.
    logger (logging.Logger): Logger instance.
    
    Returns:
    pd.DataFrame: Loaded and sorted dataframe.
    """
    try:
        logger.info(f"Loading dataframe from {path_to_csv_or_pickle}")
        # Load the dataframe from a CSV or pickle file
        _, ext = os.path.splitext(path_to_csv_or_pickle)
        if ext.lower() == '.csv':
            df = pd.read_csv(path_to_csv_or_pickle)
        elif ext.lower() in ['.pkl', '.pickle']:
            df = pd.read_pickle(path_to_csv_or_pickle)
        else:
            raise ValueError("Unsupported file extension.")
        
        # Sort for consistent slice/time ordering
        df.sort_values(by=['time_index', 'slice_index'], inplace=True)
        df.reset_index(drop=True, inplace=True)
        logger.info("Dataframe loaded successfully")
        return df
    except Exception as e:
        logger.error(f"Error loading dataframe: {e}")
        raise

def process_corrected_velocity_npy(npy_path, RDIM, CDIM, SDIM, TDIM, logger):
    """
    Load and process the corrected velocity numpy array.
    
    Parameters:
    npy_path (str): Path to the numpy file.
    RDIM (int): Row dimension.
    CDIM (int): Column dimension.
    SDIM (int): Slice dimension.
    TDIM (int): Time dimension.
    logger (logging.Logger): Logger instance.
    
    Returns:
    np.ndarray: Processed and padded velocity array.
    """
    try:
        logger.info(f"Loading corrected velocity numpy array from {npy_path}")
        # Load and process the corrected velocity numpy array
        tempnpy = np.load(npy_path)
        tempnpy = np.swapaxes(tempnpy, 0, 3)  # Swap (time, component, slice, row, col)
        tempnpy = np.swapaxes(tempnpy, 1, 4)  # Now (row, col, slice, time, component)
        tempnpy[..., 2] = -tempnpy[..., 2]  # Negate SI component
        
        # Ensure correct shape with padding
        ecc_holder = np.zeros((RDIM, CDIM, SDIM, TDIM, 3), dtype=np.int16)
        RDIM_npy, CDIM_npy = tempnpy.shape[0], tempnpy.shape[1]
        Rspacer = (RDIM - RDIM_npy) // 2
        Cspacer = (CDIM - CDIM_npy) // 2
        ecc_holder[Rspacer:Rspacer+RDIM_npy, Cspacer: Cspacer+CDIM_npy, :, :, :] = np.copy(tempnpy)
        
        logger.info("Corrected velocity numpy array processed successfully")
        return ecc_holder
    except Exception as e:
        logger.error(f"Error processing corrected velocity numpy array: {e}")
        raise

def reconstruct_corrected_velocity_nifti(vel_5d, A, output_path, logger):
    """
    Create and save the NIfTI file for corrected velocity data.
    
    Parameters:
    vel_5d (np.ndarray): Corrected velocity 5D array.
    A (np.ndarray): Affine transformation matrix.
    output_path (str): Output path for the corrected velocity NIfTI file.
    logger (logging.Logger): Logger instance.
    """
    try:
        logger.info(f"Creating NIfTI file for corrected velocity data at {output_path}")
        # Create and save the NIfTI file for corrected velocity data
        corrected_vel_nii = nib.Nifti1Image(vel_5d, A)
        nib.save(corrected_vel_nii, output_path)
        logger.info(f"Corrected velocity NIfTI saved to {output_path}")
    except Exception as e:
        logger.error(f"Error creating NIfTI file for corrected velocity data: {e}")
        raise

def create_volume_arrays(df, shape_column='vel_npy_shape'):
    """
    Determine final (nx, ny, nz, nt) dimensions and allocate arrays for magnitude and velocity data.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing DICOM metadata.
    shape_column (str): Column name for the shape information (default is 'vel_npy_shape').
    
    Returns:
    tuple: Allocated magnitude and velocity arrays.
    """
    # Determine final (nx, ny, nz, nt) dimensions
    max_time = df['time_index'].max()
    max_slice = df['slice_index'].max()
    num_time_points = int(max_time) + 1
    num_slices = int(max_slice) + 1
    
    # Read a sample DICOM file to get the dimensions
    sample_file = df['FilePath'].iloc[0]
    pix_read = pydicom.dcmread(sample_file, stop_before_pixels=False)
    pix = pix_read.pixel_array
    nx, ny = pix.shape
    
    # Allocate arrays for magnitude and velocity data
    mag_4d = np.zeros((nx, ny, num_slices, num_time_points), dtype=np.int16)
    vel_5d = np.zeros((nx, ny, num_slices, num_time_points, 3), dtype=np.int16)
    return mag_4d, vel_5d

def fill_volume_arrays(df, mag_4d, vel_5d,
                       tag_col='Tag_0043_1030',
                       filepath_col='FilePath',
                       n_jobs=1):
    """
    Fill the provided mag_4d and vel_5d arrays in either serial or parallel.
    
    Parameters:
    df (pd.DataFrame): Dataframe with DICOM metadata.
    mag_4d (np.ndarray): Pre-allocated magnitude volume array.
    vel_5d (np.ndarray): Pre-allocated velocity volume array.
    tag_col (str): Column name with the tag value used to identify mag/velocity.
    filepath_col (str): Column name containing the DICOM file path.
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    # Fills the provided mag_4d and vel_5d arrays in either serial or parallel

    # Function to process each row of the dataframe
    def process_row(row):
        ds = pydicom.dcmread(row[filepath_col])
        pix = ds.pixel_array.astype(np.int16)
        t = int(row['time_index'])
        s = int(row['slice_index'])
        tag_value = int(row[tag_col])
        return (t, s, tag_value, pix)

    # Parallel read and parse pixel data
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading DICOMs")
    )

    # Fill the arrays with the parsed data
    for (t, s, tag_value, pix) in tqdm(results, desc="Filling arrays"):
        if tag_value == 2:
            mag_4d[..., s, t] = pix
        elif tag_value == 3:
            vel_5d[..., s, t, 0] = pix  # RL velocity
        elif tag_value == 4:
            vel_5d[..., s, t, 1] = pix  # AP velocity
        elif tag_value == 5:
            vel_5d[..., s, t, 2] = pix  # SI velocity

def build_affine(flow_info_df, Nslices):
    """
    Build the affine transformation matrix from DICOM metadata.
    
    Parameters:
    flow_info_df (pd.DataFrame): Dataframe containing DICOM metadata.
    Nslices (int): Number of slices.
    
    Returns:
    tuple: Affine matrix, inverse affine matrix, row resolution, column resolution, slice thickness, and slice spacing.
    """
    # Build the affine transformation matrix from DICOM metadata
    first_slice_path = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == 0)]['FilePath'].iloc[0]
    last_slice_path = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == flow_info_df['slice_index'].max())]['FilePath'].iloc[0]

    first_slice = pydicom.dcmread(first_slice_path)
    last_slice = pydicom.dcmread(last_slice_path)

    dircos = first_slice.ImageOrientationPatient
    dircos = [float(val) for val in dircos]

    F = np.zeros((3, 2), dtype=np.float64)
    F[:, 0] = dircos[3:]
    F[:, 1] = dircos[0:3]

    res = first_slice.PixelSpacing
    rowres = res[0]
    colres = res[1]
    sthick = first_slice.SliceThickness
    normal = np.cross(F[:, 0], F[:, 1])

    impospt = np.array(first_slice.ImagePositionPatient).astype(np.float64)
    impospt_last = np.array(last_slice.ImagePositionPatient).astype(np.float64)
    slice_spacing = (impospt_last - impospt) / (Nslices - 1)

    A = np.zeros((4, 4), dtype=np.float64)
    A[3, 3] = 1
    A[0:3, 0] = rowres * F[:, 0]
    A[0:3, 1] = colres * F[:, 1]
    A[0:3, 2] = slice_spacing
    A[0:3, 3] = impospt
    Ainv = np.linalg.inv(A)

    return A, Ainv, rowres, colres, sthick, slice_spacing

def reconstruct_4dflow_nifti(mag_4d, vel_5d, A, out_mag_path, out_vel_path, logger):
    """
    Create and save the NIfTI files for magnitude and velocity data.
    
    Parameters:
    mag_4d (np.ndarray): Magnitude 4D array.
    vel_5d (np.ndarray): Velocity 5D array.
    A (np.ndarray): Affine transformation matrix.
    out_mag_path (str): Output path for the magnitude NIfTI file.
    out_vel_path (str): Output path for the velocity NIfTI file.
    logger (logging.Logger): Logger instance.
    """
    # Create and save the NIfTI files for magnitude and velocity data
    mag_nii = nib.Nifti1Image(mag_4d, A)
    vel_nii = nib.Nifti1Image(vel_5d, A)

    # Save the NIfTI files
    nib.save(mag_nii, out_mag_path)
    nib.save(vel_nii, out_vel_path)

    logger.info("NIfTI files saved to %s and %s", out_mag_path, out_vel_path)

def find_difference_between_slices(df):
    """
    Find the difference between ImagePositionPatient in the second slice and the first slice.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing DICOM metadata.
    
    Returns:
    float: The difference between the ImagePositionPatient values in the z-direction.
    """
    # Convert the string representation of the list to an actual list and then to a numpy array
    first_slice = np.array(eval(df[(df['slice_index'] == 0) & (df['time_index'] == 0)]['ImagePositionPatient'].iloc[0]), dtype=float)
    second_slice = np.array(eval(df[(df['slice_index'] == 1) & (df['time_index'] == 0)]['ImagePositionPatient'].iloc[0]), dtype=float)
    difference = second_slice - first_slice
    return difference[2]  # Assuming the difference in the z-direction is of interest

def find_cross_product_orientation(df):
    """
    Find the cross product between the row orientation and the column orientation in ImageOrientationPatient.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing DICOM metadata.
    
    Returns:
    float: The cross product of the row and column orientations in the z-direction.
    """
    # Convert the string representation of the list to an actual list and then to a numpy array
    first_slice_orientation = np.array(eval(df[(df['slice_index'] == 0) & (df['time_index'] == 0)]['ImageOrientationPatient'].iloc[0]), dtype=float)
    row_orientation = np.array(first_slice_orientation[:3])
    col_orientation = np.array(first_slice_orientation[3:])
    cross_product = np.cross(row_orientation, col_orientation)
    return cross_product[2]  # Assuming the cross product in the z-direction is of interest

def check_orientation_and_flip(df, mag_4d, vel_5d, corrected_vel_5d, logger):
    """
    Check whether the image and the velocity numpy arrays need to be flipped based on the orientation.
    
    Parameters:
    df (pd.DataFrame): Dataframe containing DICOM metadata.
    mag_4d (np.ndarray): Magnitude 4D array.
    vel_5d (np.ndarray): Velocity 5D array.
    corrected_vel_5d (np.ndarray): Corrected velocity 5D array.
    logger (logging.Logger): Logger instance.
    
    Returns:
    tuple: The potentially flipped mag_4d, vel_5d, and corrected_vel_5d arrays.
    """
    # Check whether the image and the velocity numpy arrays need to be flipped based on the orientation
    difference = find_difference_between_slices(df)
    cross_product = find_cross_product_orientation(df)
    
    if difference > 0 and cross_product < 0:
        # Flip the magnitude array along the slice direction
        logger.info("Flipping magnitude array along the slice direction")
        mag_4d = np.flip(mag_4d, axis=2)
    elif difference < 0 and cross_product > 0:
        # Flip the velocity arrays along the slice direction
        logger.info("Flipping velocity arrays along the slice direction")
        vel_5d = np.flip(vel_5d, axis=2)
        corrected_vel_5d = np.flip(corrected_vel_5d, axis=2)
    elif difference > 0 and cross_product > 0:
        # Flip both the magnitude and velocity arrays along the slice direction
        logger.info("Flipping both magnitude and velocity arrays along the slice direction")
        mag_4d = np.flip(mag_4d, axis=2)
        vel_5d = np.flip(vel_5d, axis=2)
        corrected_vel_5d = np.flip(corrected_vel_5d, axis=2)
    # If both the difference and the cross product are negative, no flip is needed
    
    return mag_4d, vel_5d, corrected_vel_5d

def patient_to_nifti(pid, base_dicom_folder, base_output_folder, base_velocity_folder, overwrite=False):
    """
    Perform the operations to convert DICOM to NIfTI for a single patient.
    
    Parameters:
    pid (str): Patient ID.
    base_dicom_folder (str): Base folder path for DICOM files.
    base_output_folder (str): Base folder path for output files.
    base_velocity_folder (str): Base folder path for velocity files.
    overwrite (bool): Whether to overwrite existing NIfTI files. Default is False.
    """
    # Setup logger
    logger = setup_logger(pid, base_output_folder)
    
    try:
        logger.info(f"Starting conversion for patient {pid}")
        # Construct full paths
        dicom_folder = os.path.join(base_dicom_folder, pid)
        output_folder = os.path.join(base_output_folder, pid)
        velocity_path = os.path.join(base_velocity_folder, f"{pid}.npy")
        csv_path = os.path.join(output_folder, "flow_info.csv")

        # Check if NIfTI files already exist and handle overwrite option
        mag_path = os.path.join(output_folder, 'mag_4dflow.nii.gz')
        vel_path = os.path.join(output_folder, 'vel-uncorrected_4dflow.nii.gz')
        cor_vel_path = os.path.join(output_folder, 'vel-corrected_4dflow.nii.gz')
        
        if not overwrite and all(os.path.exists(path) for path in [mag_path, vel_path, cor_vel_path]):
            logger.info("NIfTI files already exist and overwrite is set to False. Skipping conversion.")
            return

        # Load the 4D flow dataframe
        df_4dflow = load_4dflow_dataframe(csv_path, logger)

        # Create and fill the arrays with the 4D flow data pulled from the DICOM files
        mag_4d, vel_5d = create_volume_arrays(df_4dflow)
        fill_volume_arrays(df_4dflow, mag_4d, vel_5d, n_jobs=-1)

        # Create a correctly oriented array filled with the corrected velocity data downloaded from Tempus
        RDIM, CDIM, SDIM, TDIM = vel_5d.shape[:4]
        corrected_vel_5d = process_corrected_velocity_npy(velocity_path, RDIM, CDIM, SDIM, TDIM, logger)

        # Build the affine transformation matrix
        Nslices = len(df_4dflow['slice_index'].unique())
        A, Ainv, rowres, colres, sthick, slice_spacing = build_affine(df_4dflow, Nslices)

        # Check orientation and flip if necessary
        # TODO (issue #1): need to fix affine when performing the flip by sending the affine in and returning it
        # use z-axis affine flip logic from fix_nifti_affines in patches
        mag_4d, vel_5d, corrected_vel_5d = check_orientation_and_flip(df_4dflow, mag_4d, vel_5d, corrected_vel_5d, logger)

        # Save the 4D flow data as NIfTI files
        reconstruct_4dflow_nifti(mag_4d, vel_5d, A, mag_path, vel_path, logger)
        reconstruct_corrected_velocity_nifti(corrected_vel_5d, A, cor_vel_path, logger)
        
        # Generate GIFs from the NIfTI files
        gif_path = os.path.join(output_folder, 'mag.gif')
        generate_gif_from_nifti(mag_path, gif_path, logger)

        gif_path = os.path.join(output_folder, 'vel-uncorrected.gif')
        generate_gif_from_velocity_nifti(vel_path, gif_path, logger)

        gif_path = os.path.join(output_folder, 'zvel-uncorrected.gif')
        generate_gif_from_nifti_vel(vel_path, gif_path, logger)

        gif_path = os.path.join(output_folder, 'vel-corrected.gif')
        generate_gif_from_velocity_nifti(cor_vel_path, gif_path, logger)

        # Print affine matrix details
        logger.info("Affine matrix A:\n%s", A)
        logger.info("Inverse affine matrix Ainv:\n%s", Ainv)
        logger.info("Row resolution: %s", rowres)
        logger.info("Column resolution: %s", colres)
        logger.info("Slice thickness: %s", sthick)
        logger.info("Slice spacing: %s", slice_spacing)
        logger.info(f"Finished conversion for patient {pid}")
    except Exception as e:
        logger.error(f"Error during conversion for patient {pid}: {e}")
        raise

if __name__ == "__main__":
    # Define base paths for input and output data
    base_dicom_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    base_output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    base_velocity_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities'
    pid = 'Ackoram'
    
    # Process the patient
    patient_to_nifti(pid, base_dicom_folder, base_output_folder, base_velocity_folder, overwrite=False)
