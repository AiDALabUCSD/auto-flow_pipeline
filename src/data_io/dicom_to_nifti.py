import os
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib

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
    print('pix.shape:', pix.shape)
    nx, ny = pix.shape
    
    # Allocate
    mag_4d = np.zeros((nx, ny, num_slices, num_time_points), dtype=np.int16)
    vel_5d = np.zeros((nx, ny, num_slices, num_time_points, 3), dtype=np.int16)
    # Using int16 for magnitude and velocity data to ensure precision in calculations
    return mag_4d, vel_5d

def fill_volume_arrays(df, mag_4d, vel_5d,
                       tag_col='Tag_0043_1030',
                       filepath_col='FilePath'):
    for i, row in df.iterrows():
        t = int(row['time_index'])
        s = int(row['slice_index'])
        
        ds = pydicom.dcmread(row[filepath_col])
        pix = ds.pixel_array.astype(np.int16)
        
        tag_value = int(row[tag_col])
        if tag_value == 2:
            mag_4d[..., s, t] = pix
        elif tag_value == 3:
            vel_5d[..., s, t, 0] = pix  # RL velocity
        elif tag_value == 4:
            vel_5d[..., s, t, 1] = pix  # AP velocity
        elif tag_value == 5:
            vel_5d[..., s, t, 2] = pix  # SI velocity

def build_affine(flow_info_df, Nslices):
    first_slice = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == 0)].iloc[0]
    last_slice = flow_info_df[(flow_info_df['time_index'] == 0) & (flow_info_df['slice_index'] == flow_info_df['slice_index'].max())].iloc[0]

    dircos = first_slice['ImageOrientationPatient']
    F = np.zeros((3, 2), dtype=np.float)
    F[:, 0] = dircos[3:]
    F[:, 1] = dircos[0:3]

    res = first_slice['PixelSpacing']
    rowres = res[0]
    colres = res[1]
    sthick = first_slice['SliceThickness']
    normal = np.cross(F[:, 0], F[:, 1])

    impospt = np.array(first_slice['ImagePositionPatient']).astype(np.float)
    impospt_last = np.array(last_slice['ImagePositionPatient']).astype(np.float)
    slice_spacing = (impospt_last - impospt) / (Nslices - 1)

    A = np.zeros((4, 4), dtype=np.float)
    A[3, 3] = 1
    A[0:3, 0] = rowres * F[:, 0]
    A[0:3, 1] = colres * F[:, 1]
    A[0:3, 2] = slice_spacing
    A[0:3, 3] = impospt
    Ainv = np.linalg.inv(A)

    return A, Ainv, rowres, colres, sthick, slice_spacing

# Function to reconstruct 4D flow data into a NIfTI file
def reconstruct_4dflow_nifti(df, out_mag_path, out_vel_path, si_velocity_index=2, negate_si=True):
    # Create the NIfTI files
    mag_nii = nib.Nifti1Image(mag_4d, A)
    vel_nii = nib.Nifti1Image(vel_5d, A)

    # Save the NIfTI files
    nib.save(mag_nii, out_mag_path)
    nib.save(vel_nii, out_vel_path)

    print("NIfTI files saved to", out_mag_path, "and", out_vel_path)

if __name__ == "__main__":
    dicom_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/Cakimtol'
    output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients/Cakimtol'
    csv_path = os.path.join(output_folder, "flow_info.csv")
    df_4dflow = load_4dflow_dataframe(csv_path)

    mag_4d, vel_5d = create_volume_arrays(df_4dflow)
    fill_volume_arrays(df_4dflow, mag_4d, vel_5d)

    Nslices = len(df_4dflow['slice_index'].unique())
    A, Ainv, rowres, colres, sthick, slice_spacing = build_affine(df_4dflow, Nslices)

    mag_path = os.path.join(output_folder, '4dflow_mag.nii.gz')
    vel_path = os.path.join(output_folder, '4dflow_vel-uncor.nii.gz')
    reconstruct_4dflow_nifti(df_4dflow, mag_path, vel_path)

    print("Affine matrix A:\n", A)
    print("Inverse affine matrix Ainv:\n", Ainv)
    print("Row resolution:", rowres)
    print("Column resolution:", colres)
    print("Slice thickness:", sthick)
    print("Slice spacing:", slice_spacing)
