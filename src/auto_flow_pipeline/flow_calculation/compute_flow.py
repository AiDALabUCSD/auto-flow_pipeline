import os
import nibabel as nib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline import main_logger
from auto_flow_pipeline.data_io.catalogue_patients import load_patient_catalogue, save_patient_catalogue


def calculate_flow(flow_nii, seg_nii, conversion_factor: float = (0.1)**3 * 1/1000, pixel_area: float = 1.0) -> pd.DataFrame:
    """
    Calculate the flow rate using the segmentation and spline_flow-through NIfTI images.

    Parameters:
        flow_nii (nib.Nifti1Image): Spline flow-through NIfTI image.
        seg_nii (nib.Nifti1Image): Segmentation NIfTI image.
        conversion_factor (float): Conversion factor to convert units from cm^3/s to L/s.
        pixel_area (float): Surface area of each pixel in mm^2.

    Returns:
        pd.DataFrame: DataFrame with rows as slices and columns as times, containing flow rates.
                     Each cell in the DataFrame represents the flow (in L/s) for that slice and time frame.
    """
    # Load NIfTI data and apply unit conversions
    flow_data = flow_nii.get_fdata() * conversion_factor * pixel_area
    seg_data = seg_nii.get_fdata()

    # Mask out values outside the vessel segmentation
    flow_rate_data = flow_data * seg_data

    # Sum across x and y dimensions, leaving (slices, time)
    flow_rate_sums = np.sum(np.sum(flow_rate_data, axis=0), axis=0)

    # Build a DataFrame: rows = slices, columns = time frames
    slices, times = flow_rate_sums.shape
    flow_rate_df = pd.DataFrame(
        flow_rate_sums,
        index=range(slices),
        columns=range(times)
    )
    return flow_rate_df


def compute_volumetric_flow_rate(flow_row: pd.Series, bpm: int) -> float:
    """
    Compute the volumetric flow rate (L/min) for a given slice, by integrating its
    flow values across time (one cardiac cycle) and scaling by BPM.

    Parameters:
        flow_row (pd.Series): A row from the flow rate DataFrame (flow across time for one slice).
        bpm (int): Beats per minute for this patient.

    Returns:
        float: The volumetric flow rate (L/min) for this slice.
    """
    # Duration of one heartbeat in seconds
    sec_per_beat = 60.0 / bpm

    # Build an index to treat columns as time samples
    t = np.arange(len(flow_row))
    # The time spacing between successive frames
    dT = sec_per_beat / len(t)  # sec

    # Integrate the flow (L/s) over one beat => stroke volume (L/beat)
    stroke_volume = 0.0
    for tix in t[:-1]:
        # Trapezoidal rule over consecutive samples
        stroke_volume += dT * (flow_row[tix + 1] + flow_row[tix]) / 2.0

    # Convert liters/beat --> liters/min by multiplying by BPM
    flow_rate_l_per_min = float(stroke_volume * bpm)
    return flow_rate_l_per_min


def append_volumetric_flow_rates(flow_df: pd.DataFrame, bpm: int) -> pd.DataFrame:
    """
    For each slice (each row), compute its volumetric flow rate (L/min)
    by integrating across all time columns in that row.

    Parameters:
        flow_df (pd.DataFrame): Rows = slices, columns = time frames.
        bpm (int): Beats per minute for this patient.

    Returns:
        pd.DataFrame: The original DataFrame plus a new column 'volumetric_flow_rate'
                      containing the L/min for each slice.
    """
    flow_df['volumetric_flow_rate'] = flow_df.apply(
        lambda row: compute_volumetric_flow_rate(row, bpm),
        axis=1
    )
    return flow_df


def plot_flow_data(aortic_flow_df: pd.DataFrame, pulmonary_flow_df: pd.DataFrame, out_path: str):
    """
    Create a figure with the y-axis as flow rates (L/s) and x-axis as time frames.
    Each slice of the aorta is plotted in a different shade of red, each pulmonary slice in a different shade of blue.
    The legend shows each slice's volumetric flow (L/min). The figure is saved to disk.

    Parameters:
        aortic_flow_df (pd.DataFrame): DataFrame of aortic flow. Rows = slices, columns = time frames, plus 'volumetric_flow_rate'.
        pulmonary_flow_df (pd.DataFrame): DataFrame of pulmonary flow. Same format.
        out_path (str): File path where the figure will be saved.
    """
    plt.figure()

    # For convenience, time points are all columns except the last one (volumetric_flow_rate)
    time_aortic = aortic_flow_df.columns[:-1]
    time_pulm = pulmonary_flow_df.columns[:-1]

    # Convert them to numeric arrays (they are integer column names, but let's just ensure np.array)
    time_aortic = np.array(time_aortic, dtype=float)
    time_pulm = np.array(time_pulm, dtype=float)

    # Create color maps for slices
    n_aortic_slices = len(aortic_flow_df)
    n_pulm_slices = len(pulmonary_flow_df)

    aortic_colors = cm.Reds(np.linspace(0.4, 0.8, n_aortic_slices))
    pulm_colors   = cm.Blues(np.linspace(0.4, 0.8, n_pulm_slices))

    # Plot aortic data: connect points with a line
    for i in range(n_aortic_slices):
        row_data = aortic_flow_df.iloc[i, :-1].values  # all time columns except 'volumetric_flow_rate'
        flow_val = aortic_flow_df.iloc[i, -1]          # 'volumetric_flow_rate'
        plt.plot(
            time_aortic,
            row_data,
            color=aortic_colors[i],
            marker='o',
            label=f"Aorta slice {i}: {flow_val:.2f} L/min"
        )

    # Plot pulmonary data: connect points with a line
    for i in range(n_pulm_slices):
        row_data = pulmonary_flow_df.iloc[i, :-1].values
        flow_val = pulmonary_flow_df.iloc[i, -1]
        plt.plot(
            time_pulm,
            row_data,
            color=pulm_colors[i],
            marker='o',
            label=f"Pulm slice {i}: {flow_val:.2f} L/min"
        )

    plt.xlabel("Time frame")
    plt.ylabel("Flow (L/s)")
    plt.title("Aortic and Pulmonary Flow by Slice")
    plt.legend(loc="best")

    # Save and close
    plt.savefig(out_path)
    plt.close()


def process_patient_flow(patient_name: str, base_path: str):
    """
    Process the flow computations for a given patient.

    Parameters:
        patient_name (str): Name/ID of the patient.
        base_path (str): Base path where patient data is stored.
    """
    logger = setup_logger(patient_name, base_path)
    
    logger.info(f"Processing patient: {patient_name}")
    
    # Load the patient catalogue
    logger.info("Loading patient catalogue.")
    patient_catalogue = load_patient_catalogue()
    
    # Read BPM from the patient row
    bpm = int(patient_catalogue.loc[patient_catalogue['patient_id'] == patient_name, 'bpm'].values[0])
    logger.info(f"Beats per minute for patient {patient_name}: {bpm}")
    
    # Load flow-through and segmentation NIfTI images
    logger.info("Loading aortic flow-through and segmentation NIfTI images.")
    aortic_flow_nii = nib.load(os.path.join(base_path, patient_name, 'aortic_spline_flow-through.nii.gz'))
    aortic_seg_nii = nib.load(os.path.join(base_path, patient_name, 'segnet_aorta_segmentation.nii.gz'))
    
    # Calculate and save aortic flow
    logger.info("Calculating and saving aortic flow rates.")
    aortic_flow_df = calculate_flow(aortic_flow_nii, aortic_seg_nii)
    aortic_flow_df = append_volumetric_flow_rates(aortic_flow_df, bpm)
    aortic_flow_csv_path = os.path.join(base_path, patient_name, 'aortic_flow_rates.csv')
    aortic_flow_df.to_csv(aortic_flow_csv_path)
    
    # Same for pulmonary flow
    logger.info("Loading pulmonary flow-through and segmentation NIfTI images.")
    pulmonary_flow_nii = nib.load(os.path.join(base_path, patient_name, 'pulmonary_spline_flow-through.nii.gz'))
    pulmonary_seg_nii = nib.load(os.path.join(base_path, patient_name, 'segnet_pulmonary_segmentation.nii.gz'))
    
    logger.info("Calculating and saving pulmonary flow rates.")
    pulmonary_flow_df = calculate_flow(pulmonary_flow_nii, pulmonary_seg_nii)
    pulmonary_flow_df = append_volumetric_flow_rates(pulmonary_flow_df, bpm)
    pulmonary_flow_csv_path = os.path.join(base_path, patient_name, 'pulmonary_flow_rates.csv')
    pulmonary_flow_df.to_csv(pulmonary_flow_csv_path)

    # Plot and save
    logger.info("Plotting and saving flow data.")
    plot_out_path = os.path.join(base_path, patient_name, f"flow_plot.png")
    plot_flow_data(aortic_flow_df, pulmonary_flow_df, plot_out_path)

    logger.info(f"Plot saved to {plot_out_path}")
    logger.info(f"Completed processing for patient: {patient_name}")


def main():
    patient_name = "Ackoram"  # Update as needed
    base_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    process_patient_flow(patient_name, base_path)


if __name__ == "__main__":
    main()
