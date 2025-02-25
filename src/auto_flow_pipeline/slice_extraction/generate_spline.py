import os
import numpy as np
import pandas as pd
import nibabel as nib
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def load_affine(patient_name: str, base_folderpath: str) -> np.ndarray:
    """
    Loads the 'mag_4dflow.nii.gz' file for the given patient and returns its affine matrix.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.

    Returns
    -------
    np.ndarray
        The affine transformation matrix from the loaded NIfTI file.
    """
    nifti_path = os.path.join(base_folderpath, patient_name, "mag_4dflow.nii.gz")
    img = nib.load(nifti_path)
    return img.affine

def load_max_points(patient_name: str, base_folderpath: str) -> pd.DataFrame:
    """
    Loads the 'max_points.csv' file created by find_all_max_locations.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing max location information.
    """
    csv_path = os.path.join(base_folderpath, patient_name, "max_points.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"No 'max_points.csv' found at {csv_path}")
    return pd.read_csv(csv_path)

def transform_rcs_to_real_world(patient_name: str, base_folderpath: str) -> pd.DataFrame:
    """
    Transforms the RCS coordinates in the 'max_points.csv' file into real-world coordinates
    using the affine matrix and saves the result as 'real_world_coordinates.csv'.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing real-world coordinates.
    """
    # Load the affine matrix
    affine = load_affine(patient_name, base_folderpath)
    
    # Load the max points DataFrame
    max_points_df = load_max_points(patient_name, base_folderpath)
    
    # Extract RCS coordinates and transform them
    rcs_coords = max_points_df[['r', 'c', 's']].values
    real_world_coords = nib.affines.apply_affine(affine, rcs_coords)
    
    # Create a new DataFrame with real-world coordinates
    real_world_df = max_points_df.copy()
    real_world_df[['x', 'y', 'z']] = real_world_coords
    
    # Save the new DataFrame to a CSV file
    real_world_csv_path = os.path.join(base_folderpath, patient_name, "max_points.csv")
    real_world_df.to_csv(real_world_csv_path, index=False)
    print(f"Saved real-world coordinates to {real_world_csv_path}")

    return real_world_df

def generate_spline(df: pd.DataFrame, start_label: str, end_label: str) -> pd.DataFrame:
    """
    Generates three points that are evenly spaced between the starting and ending points.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame containing real-world coordinates.
    start_label : str
        The label of the starting point.
    end_label : str
        The label of the ending point.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the generated points, including the start and end points.
    """
    # Extract the starting and ending points
    start_point = df[df['channel_name'] == start_label][['x', 'y', 'z']].values[0]
    end_point = df[df['channel_name'] == end_label][['x', 'y', 'z']].values[0]
    
    # Generate three evenly spaced points
    points = np.linspace(start_point, end_point, num=50)
    
    # Create a DataFrame for the points
    points_df = pd.DataFrame(points, columns=['x', 'y', 'z'])
    
    # Calculate the distance from the first point
    distances = np.linalg.norm(points - start_point, axis=1)
    points_df['distance_from_start'] = distances
    
    return points_df

def generate_aortic_spline(patient_name: str, base_folderpath: str):
    """
    Generates the aortic spline with default labels 'AV' and 'Mid AAo'.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.
    """
    logger = setup_logger(patient_name, base_folderpath)
    logger.info("Generating aortic spline.")
    
    # Load the real-world coordinates DataFrame
    real_world_df = transform_rcs_to_real_world(patient_name, base_folderpath)
    
    # Generate the aortic spline
    aortic_spline_df = generate_spline(real_world_df, 'AV', 'Mid AAo')
    
    # Generate the output CSV path
    output_csv_path = os.path.join(base_folderpath, patient_name, "aortic_spline.csv")
    
    # Transform the spline to RCS coordinates and save to a CSV file
    transform_spline_to_rcs(patient_name, base_folderpath, aortic_spline_df, output_csv_path)
    logger.info(f"Saved aortic spline points to {output_csv_path}")

def transform_spline_to_rcs(patient_name: str, base_folderpath: str, spline_df: pd.DataFrame, output_csv_path: str):
    """
    Transforms the spline points to RCS coordinates using the affine matrix and adds them to the same DataFrame.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.
    spline_df : pd.DataFrame
        The DataFrame containing the spline points in real-world coordinates.
    output_csv_path : str
        The path to save the output CSV file.
    """
    logger = setup_logger(patient_name, base_folderpath)
    logger.info("Transforming spline to RCS coordinates.")
    
    # Load the affine matrix
    affine = load_affine(patient_name, base_folderpath)
    
    # Transform the spline points to RCS coordinates
    rcs_coords = nib.affines.apply_affine(np.linalg.inv(affine), spline_df[['x', 'y', 'z']].values)
    
    # Add the RCS coordinates to the same DataFrame
    spline_df[['r', 'c', 's']] = rcs_coords
    
    # Save the DataFrame with both real-world and RCS coordinates to a CSV file
    spline_df.to_csv(output_csv_path, index=False)
    logger.info(f"Saved spline points with RCS coordinates to {output_csv_path}")

def generate_pulmonary_spline(patient_name: str, base_folderpath: str):
    """
    Generates the pulmonary spline with default labels 'PV' and 'MPA'.

    Parameters
    ----------
    patient_name : str
        The name/ID of the patient folder.
    base_folderpath : str
        The base directory containing patient folders.
    """
    logger = setup_logger(patient_name, base_folderpath)
    logger.info("Generating pulmonary spline.")
    
    # Load the real-world coordinates DataFrame
    real_world_df = transform_rcs_to_real_world(patient_name, base_folderpath)
    
    # Generate the pulmonary spline
    pulmonary_spline_df = generate_spline(real_world_df, 'PV', 'MPA')
    
    # Generate the output CSV path
    output_csv_path = os.path.join(base_folderpath, patient_name, "pulmonary_spline.csv")
    
    # Transform the spline to RCS coordinates and save to a CSV file
    transform_spline_to_rcs(patient_name, base_folderpath, pulmonary_spline_df, output_csv_path)
    logger.info(f"Saved pulmonary spline points to {output_csv_path}")

def main():
    patient_name = "Bulosul"
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    generate_pulmonary_spline(patient_name, base_folderpath)
    generate_aortic_spline(patient_name, base_folderpath)

if __name__ == "__main__":
    main()