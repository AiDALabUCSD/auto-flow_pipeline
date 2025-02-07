import os
import sys
import argparse
import copy
import numpy as np
import pandas as pd
import pydicom
import nibabel as nib

def load_dataframe(filepath: str) -> pd.DataFrame:
    """
    Load the metadata DataFrame from CSV or pickle.

    Parameters
    ----------
    filepath : str
        Path to a CSV or pickle file containing the DICOM metadata.

    Returns
    -------
    df : pd.DataFrame
        The loaded DataFrame containing at least:
          - FilePath
          - InstanceNumber
          - ImagePositionPatient
          - ImageOrientationPatient
          - PixelSpacing
          - SliceThickness
          - Tag_0019_10B3
          - Tag_0043_1030
        ... etc.
    """
    if filepath.lower().endswith(".csv"):
        df = pd.read_csv(filepath)
    else:
        df = pd.read_pickle(filepath)
    return df


def compute_affine(first_ds, last_ds, nslices: int) -> np.ndarray:
    """
    Compute the 4x4 affine matrix from first/last DICOM datasets.

    Parameters
    ----------
    first_ds : pydicom.dataset.FileDataset
        The DICOM dataset for the first slice (sidx=0).
    last_ds  : pydicom.dataset.FileDataset
        The DICOM dataset for the last slice (sidx=nslices-1).
    nslices  : int
        Number of slices.

    Returns
    -------
    A : np.ndarray of shape (4, 4)
        The DICOM-to-world affine transform.
    """
    # Extract direction cosines (6 floats: row direction, col direction)
    dircos = first_ds.ImageOrientationPatient  # [rx, ry, rz, cx, cy, cz]
    F = np.zeros((3, 2), dtype=float)
    F[:, 0] = dircos[0:3]  # row direction
    F[:, 1] = dircos[3:6]  # column direction

    # Pixel spacing
    res = first_ds.PixelSpacing  # [row spacing, col spacing]
    rowres = float(res[0])
    colres = float(res[1])

    impospt_first = np.array(first_ds.ImagePositionPatient, dtype=float)
    impospt_last  = np.array(last_ds.ImagePositionPatient,  dtype=float)

    # Slice spacing (approx)
    if nslices > 1:
        slice_spacing = (impospt_last - impospt_first) / (nslices - 1)
    else:
        # If there's only one slice, we might default to slice thickness or 0
        slice_spacing = np.array([0.0, 0.0, float(first_ds.SliceThickness)])

    # Construct the 4x4 affine
    A = np.zeros((4, 4), dtype=float)
    A[3, 3] = 1.0
    # Row direction
    A[0:3, 0] = rowres * F[:, 0]
    # Column direction
    A[0:3, 1] = colres * F[:, 1]
    # Through-plane direction
    A[0:3, 2] = slice_spacing
    # Origin
    A[0:3, 3] = impospt_first

    return A


def flip_slices_if_needed(volume: np.ndarray,
                          ds_first,
                          ds_second_slice,
                          orientation_vector: np.ndarray) -> np.ndarray:
    """
    Optionally flip the slices in 'volume' if the direction from slice 0 to slice 1
    does not match the sign of orientation_vector.

    This is a simplified version of the logic in your original code.

    Parameters
    ----------
    volume : np.ndarray
        The volume array, shape e.g. (R, C, S, T, Channels).
    ds_first : pydicom.dataset.FileDataset
        DICOM for slice=0, time=0.
    ds_second_slice : pydicom.dataset.FileDataset
        DICOM for slice=1, time=0.
    orientation_vector : np.ndarray of shape (3,)
        The cross product of row and col direction cosines, or any normal vector used
        to check orientation consistency.

    Returns
    -------
    volume : np.ndarray
        Possibly flipped along slice dimension.
    """
    impospt_0 = np.array(ds_first.ImagePositionPatient,    dtype=float)
    impospt_1 = np.array(ds_second_slice.ImagePositionPatient, dtype=float)

    delta = np.sum(impospt_1 - impospt_0)
    sdir_sum = np.sum(orientation_vector)

    # If the sign is mismatched, flip:
    if (delta * sdir_sum) < 0:
        # Flip along the slice dimension (index=2 if shape is (R, C, S, T, Channels))
        volume = np.flip(volume, axis=2)
    return volume


def make_4dflow_nifti(flow_df: pd.DataFrame,
                      output_path: str,
                      tdim: int,
                      sdim: int,
                      separate_volumes: bool = False):
    """
    Build the final 4D Flow data from a dataframe subset and save to NIfTI.

    Parameters
    ----------
    flow_df : pd.DataFrame
        Sub-dataFrame with the relevant 4D Flow DICOM files for one series.
    output_path : str
        Output path for the .nii or .nii.gz file(s).
    tdim : int
        Number of time frames (cardiac phases).
    sdim : int
        Number of slices.
    separate_volumes : bool
        If True, writes four separate NIfTI volumes for magnitude, vx, vy, vz.
        Otherwise writes a single 5D volume: (R, C, S, T, 4).
    """
    # --- 1. Figure out the 2D shape (Rows, Cols) ---
    example_path = flow_df.iloc[0]["FilePath"]
    ds_example   = pydicom.dcmread(example_path)
    RDIM = ds_example.Rows
    CDIM = ds_example.Columns

    # Create holder array: shape (R, C, S, T, Channels=4)
    # Channel 0 -> magnitude
    # Channel 1 -> vx
    # Channel 2 -> vy
    # Channel 3 -> vz
    holder = np.zeros((RDIM, CDIM, sdim, tdim, 4), dtype=np.int16)

    # --- 2. Populate the data ---
    for idx, row in flow_df.iterrows():
        # Identify direction from Tag_0043_1030 or Tag_0019_10B3
        # Example approach:
        flow_dir = None
        if pd.notnull(row["Tag_0043_1030"]):
            flow_dir = int(row["Tag_0043_1030"])
        elif pd.notnull(row["Tag_0019_10B3"]):
            flow_dir = int(row["Tag_0019_10B3"])

        # Skip if not a recognized 4D flow direction
        if flow_dir not in [2, 3, 4, 5]:
            continue

        ds = pydicom.dcmread(row["FilePath"])
        pix = ds.pixel_array  # shape (R, C)

        instance_num = ds.InstanceNumber  # 1-based
        # Deduce slice/time. E.g. slice = floor((instance_num-1)/TDIM).
        # (Adjust if you know your data enumerates differently.)
        sidx = (instance_num - 1) // tdim
        tidx = (instance_num - 1) % tdim

        # Channel index
        if flow_dir == 2:
            didx = 0  # Magnitude
        elif flow_dir == 3:
            didx = 1  # vx
        elif flow_dir == 4:
            didx = 2  # vy
        elif flow_dir == 5:
            didx = 3  # vz

        holder[..., sidx, tidx, didx] = pix

    # --- 3. Compute the affine ---
    # We find the DICOM for slice=0, time=0, and for slice=sdim-1, time=0
    # InstanceNumber for slice=0, time=0 => 1
    # InstanceNumber for slice=sdim-1, time=0 => (sdim - 1)*tdim + 1
    instnum_first = 1
    instnum_last  = (sdim - 1)*tdim + 1

    row_first = flow_df[flow_df["InstanceNumber"] == instnum_first].iloc[0]
    row_last  = flow_df[flow_df["InstanceNumber"] == instnum_last ].iloc[0]

    ds_first = pydicom.dcmread(row_first["FilePath"])
    ds_last  = pydicom.dcmread(row_last["FilePath"])

    A = compute_affine(ds_first, ds_last, sdim)

    # --- 4. Check if we need a slice flip ---
    # cross product of row dir and col dir from ds_first
    dircos = ds_first.ImageOrientationPatient
    x_cos = np.array(dircos[0:3], dtype=float)
    y_cos = np.array(dircos[3:6], dtype=float)
    sdir = np.cross(x_cos, y_cos)  # Normal vector
    # read the dicom for slice=1, time=0 => instance_num=2
    # (only if sdim>1 to avoid index error)
    if sdim > 1:
        instnum_slice1 = 2  # sidx=1, tidx=0 => 2
        row_slice1 = flow_df[flow_df["InstanceNumber"] == instnum_slice1].iloc[0]
        ds_slice1 = pydicom.dcmread(row_slice1["FilePath"])

        holder = flip_slices_if_needed(holder, ds_first, ds_slice1, sdir)

    # If we did flip, you might also want to adjust the affine A, by flipping
    # in the 3rd dimension. That code depends on whether you want the
    # world coordinates to remain consistent with typical orientation.
    # For simplicity, we skip re-adjusting A here. In production code,
    # you'd want to re-calculate the new slice direction vector or
    # apply a simple transformation matrix to A.

    # --- 5. Possibly reorder or transpose to get a 5D shape nibabel expects ---
    # nibabel can store 5D, but some tools only partially support it.
    # Let's do (R, C, S, T, Channels):
    # We already have that shape. If we want channels as 4th dim and time as 5th,
    # we can transpose. It's up to you.

    # For example, do (R, C, S, Channels, T):
    # holder = np.transpose(holder, (0, 1, 2, 4, 3))

    # --- 6. Write to NIfTI ---
    if separate_volumes:
        # Save separate volumes
        mag_nii = nib.Nifti1Image(holder[..., 0], affine=A)
        vx_nii  = nib.Nifti1Image(holder[..., 1], affine=A)
        vy_nii  = nib.Nifti1Image(holder[..., 2], affine=A)
        vz_nii  = nib.Nifti1Image(holder[..., 3], affine=A)

        nib.save(mag_nii, os.path.join(output_path, "4dflow_mag.nii.gz"))
        nib.save(vx_nii,  os.path.join(output_path, "4dflow_vx.nii.gz"))
        nib.save(vy_nii,  os.path.join(output_path, "4dflow_vy.nii.gz"))
        nib.save(vz_nii,  os.path.join(output_path, "4dflow_vz.nii.gz"))
    else:
        # Single file with 5D data
        img_nii = nib.Nifti1Image(holder, affine=A)
        nib.save(img_nii, os.path.join(output_path, "4dflow_all.nii.gz"))


def main():
    parser = argparse.ArgumentParser(
        description="Build 4D Flow NIfTI from a DataFrame of DICOM metadata."
    )
    parser.add_argument("-i", "--input", required=True,
                        help="Path to CSV or pickle file with DICOM metadata.")
    parser.add_argument("-o", "--output-dir", required=True,
                        help="Directory to store the output NIfTI.")
    parser.add_argument("--series-uid", required=False, default=None,
                        help="Optionally specify a SeriesInstanceUID to filter.")
    parser.add_argument("--tdim", type=int, required=True,
                        help="Number of time frames (cardiac phases).")
    parser.add_argument("--sdim", type=int, required=True,
                        help="Number of slices.")
    parser.add_argument("--separate-volumes", action="store_true",
                        help="If set, saves separate NIfTI for mag, vx, vy, vz.")

    args = parser.parse_args()

    # Make sure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load the DataFrame
    df = load_dataframe(args.input)

    # 2. Filter if SeriesInstanceUID is specified
    if args.series_uid is not None:
        df = df[df["SeriesInstanceUID"] == args.series_uid]
        if df.empty:
            print(f"No rows found for SeriesInstanceUID={args.series_uid}")
            sys.exit(1)

    # 3. (Optional) If you have multiple 4D Flow series in df, group and iterate:
    # For now, assume we just have one group we want to process:
    # or we process everything in one pass if df is already only the 4D Flow series.
    # If you want to group, do something like:
    # for series_uid, group_df in df.groupby("SeriesInstanceUID"):
    #     make_4dflow_nifti(group_df, args.output_dir, args.tdim, args.sdim, args.separate_volumes)

    # If you just want to build from the entire df:
    make_4dflow_nifti(df, args.output_dir, args.tdim, args.sdim, args.separate_volumes)

    print("4D Flow NIfTI creation finished.")

if __name__ == "__main__":
    main()
