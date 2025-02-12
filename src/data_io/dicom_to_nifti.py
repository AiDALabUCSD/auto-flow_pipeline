import os
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib
import matplotlib.pyplot as plt
import imageio

# Add optional imports for parallelization and progress bars
from joblib import Parallel, delayed
from tqdm import tqdm

def load_4dflow_dataframe(path_to_csv_or_pickle):
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
    return df

def process_corrected_velocity_npy(npy_path, RDIM, CDIM, SDIM, TDIM):
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
    
    return ecc_holder

def reconstruct_corrected_velocity_nifti(vel_5d, A, output_path):
    """
    Create and save the NIfTI file for corrected velocity data.
    :param vel_5d: Corrected velocity 5D array.
    :param A: Affine transformation matrix.
    :param output_path: Output path for the corrected velocity NIfTI file.
    """
    corrected_vel_nii = nib.Nifti1Image(vel_5d, A)
    nib.save(corrected_vel_nii, output_path)
    print(f"Corrected velocity NIfTI saved to {output_path}")

def create_volume_arrays(df, shape_column='vel_npy_shape'):
    # Determine final (nx, ny, nz, nt)
    max_time = df['time_index'].max()
    max_slice = df['slice_index'].max()
    num_time_points = int(max_time) + 1
    num_slices = int(max_slice) + 1
    
    # Read a sample DICOM file to get the dimensions
    sample_file = df['FilePath'].iloc[0]
    pix_read = pydicom.dcmread(sample_file, stop_before_pixels=False)
    pix = pix_read.pixel_array
    # print('pix.shape:', pix.shape)
    nx, ny = pix.shape
    
    # Allocate
    mag_4d = np.zeros((nx, ny, num_slices, num_time_points), dtype=np.int16)
    vel_5d = np.zeros((nx, ny, num_slices, num_time_points, 3), dtype=np.int16)
    # Using int16 for magnitude and velocity data to ensure precision in calculations
    return mag_4d, vel_5d

def fill_volume_arrays(df, mag_4d, vel_5d,
                       tag_col='Tag_0043_1030',
                       filepath_col='FilePath',
                       n_jobs=1):
    """
    Fills the provided mag_4d and vel_5d arrays in either serial or parallel.
    :param df: Pandas dataframe with DICOM metadata.
    :param mag_4d: Pre-allocated magnitude volume array.
    :param vel_5d: Pre-allocated velocity volume array.
    :param tag_col: Column name with the tag value used to identify mag/velocity.
    :param filepath_col: Column name containing the DICOM file path.
    :param n_jobs: Number of parallel jobs. Use -1 for all available cores.
    """

    # 1) Parallel read and parse pixel data
    def process_row(row):
        ds = pydicom.dcmread(row[filepath_col])
        pix = ds.pixel_array.astype(np.int16)
        t = int(row['time_index'])
        s = int(row['slice_index'])
        tag_value = int(row[tag_col])
        return (t, s, tag_value, pix)

    results = Parallel(n_jobs=n_jobs)(
        delayed(process_row)(row)
        for _, row in tqdm(df.iterrows(), total=len(df), desc="Reading DICOMs")
    )

    # 2) Fill the arrays
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
    first_slice_path = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == 0)]['FilePath'].iloc[0]
    last_slice_path = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == flow_info_df['slice_index'].max())]['FilePath'].iloc[0]

    first_slice = pydicom.dcmread(first_slice_path)
    last_slice = pydicom.dcmread(last_slice_path)

    dircos = first_slice.ImageOrientationPatient
    # print("dircos:", dircos)
    # Convert dircos to a list of floats
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

def reconstruct_4dflow_nifti(mag_4d, vel_5d, A, out_mag_path, out_vel_path):
    """
    Create and save the NIfTI files for magnitude and velocity data.
    :param mag_4d: Magnitude 4D array.
    :param vel_5d: Velocity 5D array.
    :param A: Affine transformation matrix.
    :param out_mag_path: Output path for the magnitude NIfTI file.
    :param out_vel_path: Output path for the velocity NIfTI file.
    """
    mag_nii = nib.Nifti1Image(mag_4d, A)
    vel_nii = nib.Nifti1Image(vel_5d, A)

    # Save the NIfTI files
    nib.save(mag_nii, out_mag_path)
    nib.save(vel_nii, out_vel_path)

    print("NIfTI files saved to", out_mag_path, "and", out_vel_path)

def generate_gif_from_nifti(nifti_path, output_path, n_jobs=-1):
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    num_slices = data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(data[:, :, i, 0], cmap='gray')  # Using the first time point
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        fig.canvas.draw()
        
        # Extract buffer and reshape correctly (RGBA -> RGB)
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (RGBA)
        image = image[..., :3]  # Convert RGBA to RGB
        
        plt.close(fig)
        return image
    
    images = Parallel(n_jobs=n_jobs)(delayed(process_slice)(i) for i in tqdm(range(num_slices), desc="Generating GIF"))
    
    imageio.mimsave(output_path, images, duration=0.1)  # Save as GIF
    print(f"GIF saved to {output_path}")

def generate_gif_from_nifti_vel(nifti_path, output_path, vel_dir=2, n_jobs=-1):
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    num_slices = data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(data[:, :, i, 0, vel_dir], cmap='jet')  # Using the first time point
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        fig.canvas.draw()
        
        # Extract buffer and reshape correctly (RGBA -> RGB)
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (RGBA)
        image = image[..., :3]  # Convert RGBA to RGB
        
        plt.close(fig)
        return image
    
    images = Parallel(n_jobs=n_jobs)(delayed(process_slice)(i) for i in tqdm(range(num_slices), desc="Generating GIF"))
    
    imageio.mimsave(output_path, images, duration=0.1)  # Save as GIF
    print(f"GIF saved to {output_path}")

def compute_speed_from_velocity_nifti(nifti_path):
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    speed = np.sqrt(np.sum(data ** 2, axis=-1))  # Compute speed magnitude
    return speed

def generate_gif_from_velocity_nifti(nifti_path, output_path, n_jobs=-1):
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    speed_data = np.sqrt(np.sum(data ** 2, axis=-1))  # Compute speed magnitude
    num_slices = speed_data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(speed_data[:, :, i, 0], cmap='jet')  # Using the first time point
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
        fig.canvas.draw()
        
        # Extract buffer and reshape correctly (RGBA -> RGB)
        image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (RGBA)
        image = image[..., :3]  # Convert RGBA to RGB
        
        plt.close(fig)
        return image
    
    images = Parallel(n_jobs=n_jobs)(delayed(process_slice)(i) for i in tqdm(range(num_slices), desc="Generating GIF"))
    
    imageio.mimsave(output_path, images, duration=0.1)  # Save as GIF
    print(f"GIF saved to {output_path}")

if __name__ == "__main__":
    dicom_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/Ackoram'
    output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients/Ackoram'
    velocity_path = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities/Ackoram.npy'
    csv_path = os.path.join(output_folder, "flow_info.csv")
    df_4dflow = load_4dflow_dataframe(csv_path)

    ## create and fill the arrays with the 4d flow data pulled from the dicom files
    mag_4d, vel_5d = create_volume_arrays(df_4dflow)
    # Use fill_volume_arrays with parallelization (set n_jobs to the number of workers)
    fill_volume_arrays(df_4dflow, mag_4d, vel_5d, n_jobs=-1)

    ## create a correctly orriented array filled with the corrected velocity data downloaded from tempus

    # Define dimensions for padding correction
    RDIM, CDIM, SDIM, TDIM = vel_5d.shape[:4]
    corrected_vel_5d = process_corrected_velocity_npy(velocity_path, RDIM, CDIM, SDIM, TDIM)

    Nslices = len(df_4dflow['slice_index'].unique())
    A, Ainv, rowres, colres, sthick, slice_spacing = build_affine(df_4dflow, Nslices)

    # save the 4D flow data as NIfTI files
    mag_path = os.path.join(output_folder, 'mag_4dflow.nii.gz')
    vel_path = os.path.join(output_folder, 'vel-uncorrected_4dflow.nii.gz')
    reconstruct_4dflow_nifti(mag_4d, vel_5d, A, mag_path, vel_path)
    corrected_vel_nifti_path = os.path.join(output_folder, 'vel_corrected_4dflow.nii.gz')
    reconstruct_corrected_velocity_nifti(corrected_vel_5d, A, corrected_vel_nifti_path)

    # Generate GIFs from the NIfTI files
    gif_path = os.path.join(output_folder, 'mag.gif')
    generate_gif_from_nifti(mag_path, gif_path)

    gif_path = os.path.join(output_folder, 'vel-uncorrected.gif')
    generate_gif_from_velocity_nifti(vel_path, gif_path)

    gif_path = os.path.join(output_folder, 'zvel-uncorrected.gif')
    generate_gif_from_nifti_vel(vel_path, gif_path)

    gif_path = os.path.join(output_folder, 'vel-corrected.gif')
    generate_gif_from_velocity_nifti(corrected_vel_nifti_path, gif_path)

    print("Affine matrix A:\n", A)
    print("Inverse affine matrix Ainv:\n", Ainv)
    print("Row resolution:", rowres)
    print("Column resolution:", colres)
    print("Slice thickness:", sthick)
    print("Slice spacing:", slice_spacing)
