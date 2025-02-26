from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.ndimage import map_coordinates
import numpy as np
import pandas as pd
import nibabel as nib
import os
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.visualization.slice_extraction.generate_gifs import generate_gif_over_time, generate_gif_over_slices_and_time

def setup_patient_rgi(patient_name: str, base_path: str, mag_data=None, flow_data=None):
    """
    Set up RegularGridInterpolator objects for the patient's magnitude and flow data.
    
    The function assumes that:
      - The magnitude image is stored as 'mag_4dflow.nii.gz'
      - The flow image is stored as 'vel-corrected_4dflow.nii.gz'
    inside a folder named with the patient_name under the given base_path.
    
    Parameters:
        patient_name (str): The patient identifier.
        base_path (str): The base path where the patient folder is located.
        mag_data (np.ndarray, optional): Pre-loaded magnitude data. Defaults to None.
        flow_data (np.ndarray, optional): Pre-loaded flow data. Defaults to None.
        
    Returns:
        tuple: A tuple (mag_rgi, flow_rgi) where each is a RegularGridInterpolator
               set up for the respective data.
    """
    logger = setup_logger(patient_name, base_path)
    logger.info(f"Starting RGI setup for patient {patient_name}")
    
    if mag_data is None or flow_data is None:
        # Build file paths for the magnitude and flow NIfTI files.
        mag_path = os.path.join(base_path, patient_name, "mag_4dflow.nii.gz")
        flow_path = os.path.join(base_path, patient_name, "vel-corrected_4dflow.nii.gz")
        
        logger.info(f"Loading NIfTI files for patient {patient_name}")
        
        # Load the NIfTI images using nibabel.
        mag_img = nib.load(mag_path)
        flow_img = nib.load(flow_path)
        
        # Get image data as numpy arrays.
        mag_data = mag_img.get_fdata()
        flow_data = flow_img.get_fdata()
    
    logger.info(f"Loaded magnitude data shape: {mag_data.shape}")
    logger.info(f"Loaded flow data shape: {flow_data.shape}")
    
    # Create a coordinate grid using the dimensions of the volume.
    # RegularGridInterpolator expects one coordinate array per dimension.
    grid_mag = tuple(np.arange(dim) for dim in mag_data.shape)
    grid_flow = tuple(np.arange(dim) for dim in flow_data.shape)
    
    logger.info(f"Began RGI setup for patient {patient_name}")
    # Set up the RegularGridInterpolators for the magnitude and flow data.
    mag_rgi = RGI(grid_mag, mag_data, bounds_error=False, fill_value=0)
    flow_rgi = RGI(grid_flow, flow_data, bounds_error=False, fill_value=0)
    logger.info(f"RGI setup completed for patient {patient_name}")
    
    return mag_rgi, flow_rgi

def rotation_matrix_from_vectors(vec1, vec2):
    """
    Compute the rotation matrix that rotates vec1 to vec2 using Rodrigues' formula.

    Parameters:
        vec1 (np.ndarray): Source vector.
        vec2 (np.ndarray): Destination vector.

    Returns:
        np.ndarray: A 3x3 rotation matrix.
    """
    a = vec1 / np.linalg.norm(vec1)
    b = vec2 / np.linalg.norm(vec2)
    cross = np.cross(a, b)
    s = np.linalg.norm(cross)
    c = np.dot(a, b)
    if s < 1e-8:
        # Vectors are parallel (or anti-parallel).
        return np.eye(3)
    vx = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0]
    ])
    R = np.eye(3) + vx + vx.dot(vx) * ((1 - c) / (s**2))
    return R

def generate_sampling_plane(spline_df, row: int, plane_dims: tuple = (256, 256), 
                            resolution: tuple = (256, 256), affine: np.ndarray = None):
    """
    Generates a sampling plane at a given spline point.
    
    The spline DataFrame is assumed to contain real‑world coordinates 
    ('x', 'y', 'z'). The plane is defined as the cross‐section of the vessel, 
    that is, the plane perpendicular to the vessel’s tangent. Because the 
    normal vector isn’t stored in the DataFrame, it is computed using the first 
    and last spline points.
    
    The procedure is:
      1. Compute the vessel tangent as the unit vector from the first to last spline point.
      2. Reorient the plane so that the vessel’s tangent (the plane’s normal) is axial ([0, 0, 1]).
      3. Sample a uniform grid in this axial plane (256 mm by 256 mm) at the specified resolution.
      4. Rotate the grid back to the vessel’s original (oblique) orientation.
      5. Convert the resulting real‑world points to RCS space using the provided affine, if given.
    
    Parameters:
        spline_df (pd.DataFrame): DataFrame containing spline points with 'x', 'y', and 'z' columns.
        row (int): Row index at which to generate the plane.
        plane_dims (tuple): Dimensions (height, width) of the plane in mm. Default is (256, 256).
        resolution (tuple): Grid resolution (num_rows, num_cols) of the plane. Default is (256, 256).
        affine (np.ndarray): The affine matrix used to convert real-world to RCS coordinates.
                             If None, conversion is skipped.
                             
    Returns:
        np.ndarray: An array of shape (num_rows, num_cols, 3) with the plane points in RCS space (or real-world if affine is None).
    """
    import numpy as np
    import nibabel as nib

    # Use the spline point at this row as the plane center.
    center = spline_df.loc[row, ['x', 'y', 'z']].to_numpy()

    # Compute vessel tangent from the first and last spline points.
    first = spline_df.loc[0, ['x', 'y', 'z']].to_numpy()
    last = spline_df.loc[len(spline_df) - 1, ['x', 'y', 'z']].to_numpy()
    tangent = last - first
    tangent_norm = np.linalg.norm(tangent)
    if tangent_norm == 0:
        raise ValueError("The computed vessel tangent has zero length.")
    tangent = tangent / tangent_norm

    # For a vessel cross-section, the plane’s normal is the tangent.
    unit_normal = tangent

    # Compute the rotation that rotates the vessel normal to the axial direction [0, 0, 1].
    R = rotation_matrix_from_vectors(unit_normal, np.array([0, 0, 1]))
    # Its inverse (transpose) rotates from axial back to the original orientation.
    R_inv = R.T

    # Unpack plane dimensions and resolution.
    height, width = plane_dims
    num_rows, num_cols = resolution

    # Create a 2D grid of offsets in the axial plane.
    xs = np.linspace(-width/2, width/2, num_cols)
    ys = np.linspace(-height/2, height/2, num_rows)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X)  # In the axial plane, z-offset is zero.
    # Offsets has shape (num_rows, num_cols, 3).
    offsets = np.stack([X, Y, Z], axis=-1)

    # Rotate the grid offsets back to the vessel's oblique orientation
    # and add the center. Each point is given by: point = center + R_inv @ offset.
    plane_real_world = center + np.tensordot(offsets, R_inv, axes=([2], [1]))

    # If an affine is provided, convert the real-world points to RCS space.
    if affine is not None:
        inv_affine = np.linalg.inv(affine)
        flat_points = plane_real_world.reshape(-1, 3)
        plane_rcs = nib.affines.apply_affine(inv_affine, flat_points)
        plane_rcs = plane_rcs.reshape(num_rows, num_cols, 3)
        return plane_rcs
    else:
        return plane_real_world

def sample_aortic_spline(patient_name: str, base_path: str, indices: list, plane_dims: tuple = (256, 256), resolution: tuple = (256, 256)):
    """
    Sample multiple points along the aortic spline over all time points and save the result as aortic_spline.nii.gz.
    
    Parameters:
        patient_name (str): The patient identifier.
        base_path (str): The base path where the patient folder is located.
        indices (list): List of indices along the spline to sample.
        plane_dims (tuple): Dimensions (height, width) of the plane in mm. Default is (256, 256).
        resolution (tuple): Grid resolution (num_rows, num_cols) of the plane. Default is (256, 256).
    """
    logger = setup_logger(patient_name, base_path)
    logger.info(f"Starting aortic spline sampling for patient {patient_name}")
    
    # Set up the RGIs.
    mag_rgi, flow_rgi = setup_patient_rgi(patient_name, base_path)
    
    # Assume the aortic spline is saved as a CSV in the patient's directory.
    spline_csv = os.path.join(base_path, patient_name, "aortic_spline.csv")
    aortic_spline_df = pd.read_csv(spline_csv)
    
    # Read the affine matrix from the patient's mag_4dflow.nii.gz file.
    mag_path = os.path.join(base_path, patient_name, "mag_4dflow.nii.gz")
    mag_img = nib.load(mag_path)
    affine = mag_img.affine
    
    # Sample at all time points
    num_time_points = mag_img.shape[3]
    num_points = resolution[0] * resolution[1]
    sampled_planes = np.zeros((resolution[0], resolution[1], len(indices), num_time_points))
    
    for idx, row_idx in enumerate(indices):
        # Generate the sampling plane for the given index.
        plane_rcs = generate_sampling_plane(aortic_spline_df, row_idx, 
                                            plane_dims=plane_dims, 
                                            resolution=resolution, 
                                            affine=affine)
        
        # Flatten the plane for sampling.
        flat_plane = plane_rcs.reshape(-1, 3)
        
        for time_val in range(num_time_points):
            # Append a time coordinate for each point.
            sample_points = np.hstack([flat_plane, np.full((num_points, 1), time_val)])
            
            # Sample from the magnitude interpolator.
            sampled_values = mag_rgi(sample_points)
            
            # Reshape the sampled values back to the plane grid.
            sampled_plane = sampled_values.reshape(resolution)
            sampled_planes[..., idx, time_val] = sampled_plane
    
    # Save the sampled planes as a NIfTI file.
    output_nifti_path = os.path.join(base_path, patient_name, "aortic_spline.nii.gz")
    sampled_nifti_img = nib.Nifti1Image(sampled_planes, affine)
    nib.save(sampled_nifti_img, output_nifti_path)
    logger.info(f"Saved sampled aortic spline to {output_nifti_path}")

    # Generate a GIF of the sampled planes over time.
    output_gif_path = os.path.join(base_path, patient_name, "aortic_spline.gif")
    generate_gif_over_slices_and_time(sampled_planes, output_gif_path)
    logger.info(f"Saved GIF of sampled aortic spline to {output_gif_path}")

def main():
    patient_name = 'Bulosul'
    base_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    
    # Example indices to sample along the spline.
    indices = [5, 10, 15, 20, 25]
    
    # Sample the aortic spline and save the result.
    sample_aortic_spline(patient_name, base_path, indices)

if __name__ == "__main__":
    main()