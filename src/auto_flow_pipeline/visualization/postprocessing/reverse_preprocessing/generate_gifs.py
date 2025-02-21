import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
from joblib import Parallel, delayed
from tqdm import tqdm
from auto_flow_pipeline.data_io.logging_setup import setup_logger
from auto_flow_pipeline.slice_extraction.extract_from_locnet import _get_MAX  # Import the _get_MAX function

def generate_gif_for_slice(slice_idx, input_array, pred_array, max_locs, timepoint=3):
    """
    Generates a single frame for the GIF.

    Parameters:
    slice_idx (int): The index of the slice to generate the frame for
    input_array (np.ndarray): The input data array
    pred_array (np.ndarray): The prediction data array
    max_locs (list): List of maximum intensity locations for each channel
    timepoint (int): The timepoint to use for the input data

    Returns:
    np.ndarray: The generated frame
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    axes = axes.flatten()
    
    # First subplot: just the input
    axes[0].imshow(input_array[:, :, slice_idx, timepoint], cmap='gray')
    axes[0].set_title('Input Data')

    # Other subplots: input with prediction channels overlaid
    titles = [
        "AV", "Proximal AAo", "Mid AAo", "Full Ao",
        "PV", "Proximal MPA", "MPA", "Full PA with branches"
    ]
    for i in range(8):
        axes[i + 1].imshow(input_array[:, :, slice_idx, timepoint], cmap='gray')
        axes[i + 1].imshow(pred_array[:, :, slice_idx, timepoint, i], cmap='jet', alpha=0.3)
        axes[i + 1].set_title(titles[i])
        
        # Plot the max location as a red dot if it is in the current slice
        max_loc = max_locs[i]
        if max_loc[2] == slice_idx:
            axes[i + 1].plot(max_loc[1], max_loc[0], 'o', markersize=4, markerfacecolor='white')

    for ax in axes:
        ax.axis('off')

    fig.canvas.draw()  # Ensure the figure is drawn before extracting the buffer
    frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8').reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close(fig)
    return frame

def generate_gif(patient_name, base_output_folder, input_array=None, pred_array=None, timepoint=3, n_jobs=-1):
    """
    Generates a 3x3 subplot GIF where the first frame is the input data and the other 8 frames are the input
    with one of the prediction channels overlaid on top. The GIF scrolls down the slice direction and is saved
    as 'pred_from_locnet.gif'.

    Parameters:
    patient_name (str): The name/ID of the patient
    base_output_folder (str): The base folder where patient data is stored
    input_array (np.ndarray, optional): The input data array
    pred_array (np.ndarray, optional): The prediction data array
    timepoint (int): The timepoint to use for the input data
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    logger = setup_logger(patient_name, base_output_folder)
    path_to_patient = os.path.join(base_output_folder, patient_name)

    if input_array is None:
        input_nii_path = os.path.join(path_to_patient, 'mag_4dflow.nii.gz')
        if not os.path.exists(input_nii_path):
            logger.error(f"Could not find {input_nii_path}.")
            return
        input_nii = nib.load(input_nii_path)
        input_array = input_nii.get_fdata()

    if pred_array is None:
        pred_nii_path = os.path.join(path_to_patient, 'locnet_pred_processed.nii.gz')
        if not os.path.exists(pred_nii_path):
            logger.error(f"Could not find {pred_nii_path}.")
            return
        pred_nii = nib.load(pred_nii_path)
        pred_array = pred_nii.get_fdata()

    # Calculate max locations for each channel once
    max_locs = [_get_MAX(pred_array[:, :, :, timepoint, i]) for i in range(8)]

    gif_path = os.path.join(path_to_patient, 'locnet_pred_processed.gif')
    
    logger.info("Generating GIF...")
    num_slices = input_array.shape[2]
    frames = Parallel(n_jobs=n_jobs)(delayed(generate_gif_for_slice)(i, input_array, pred_array, max_locs, timepoint) for i in tqdm(range(num_slices), desc=f"Generating GIF for {patient_name}"))

    imageio.mimsave(gif_path, frames, fps=5)
    logger.info(f"GIF saved at: {gif_path}")

if __name__ == "__main__":
    base_output_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'  # Update this path as needed
    patient_name = 'Ackoram'  # Update this patient name as needed
    generate_gif(patient_name, base_output_folder)
