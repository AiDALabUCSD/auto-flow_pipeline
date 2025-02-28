import os
import numpy as np
from auto_flow_pipeline.visualization.slice_extraction.generate_gifs import generate_combined_gif
import matplotlib.pyplot as plt
import imageio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def generate_gifs_for_slices(array: np.ndarray, output_folder: str, duration: float = 0.5) -> None:
    """
    Generate GIFs for each slice from the given array.
    
    Parameters:
        array (np.ndarray): Input array of shape (time, r, c, slices, 6).
        output_folder (str): Folder to save the generated GIFs.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    time, r, c, slices, channels = array.shape
    assert channels == 6, "Input array must have 6 channels."

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    for s in range(slices):
        mag_images = array[:, :, :, s, :3]  # First three channels for mag images
        through_flow_images = array[:, :, :, s, 3:]  # Last three channels for through_flow velocity

        # Generate GIF for the current slice
        output_path = os.path.join(output_folder, f'slice_{s + 1}.gif')
        generate_combined_gif(mag_images, through_flow_images, output_path, duration, value_range=(-1, 1))

def generate_two_row_gifs_for_slices(
    patient_name: str,
    base_path: str,
    output_name: str,
    array: np.ndarray,
    duration: float = 0.5,
    value_range: tuple = (-1, 1)
) -> None:
    """
    Generate GIFs for each slice from the given array, splitting the channels into two rows:
    - Top row: first 3 channels (mag) shown in grayscale.
    - Bottom row: last 3 channels (through_flow velocity) shown with a 'jet' colormap.

    Parameters:
      patient_name (str): The name of the patient.
      base_path (str): The base path where the output folder will be created.
      output_name (str): The name of the output folder.
      array (np.ndarray): The input array of shape (time, r, c, slices, 6).
      duration (float): Duration (in seconds) for each GIF frame.
      value_range (tuple): (min, max) range for the flow channels. Default is (-1, 1).
    """
    time, r, c, num_slices, channels = array.shape
    assert channels == 6, "Input array must have 6 channels."

    # Construct output folder path
    output_folder = os.path.join(base_path, patient_name, output_name)
    os.makedirs(output_folder, exist_ok=True)

    mag_data = array[..., :3]  # shape: (time, r, c, slices, 3)
    flow_data = array[..., 3:] # shape: (time, r, c, slices, 3)

    # Normalize flow_data to [0..1] within the specified value_range
    vmin, vmax = value_range
    flow_simple_norm = np.clip(flow_data, vmin, vmax)
    flow_simple_norm = (flow_simple_norm - vmin) / (vmax - vmin)

    for s in range(num_slices):
        frames = []
        for t in range(time):
            # Prepare top row (mag) as grayscale
            mag_slice = mag_data[t, :, :, s, :]  # shape: (r, c, 3)
            # Convert each mag channel to grayscale and concatenate horizontally
            # shape after channel combination is (r, c) for each channel => we tile them horizontally
            row_top = []
            for ch in range(3):
                grayscale_img = mag_slice[:, :, ch]
                # scale each channel from [min..max] to [0..255] for display
                min_val, max_val = grayscale_img.min(), grayscale_img.max()
                if max_val - min_val > 0:
                    scaled = 255 * (grayscale_img - min_val) / (max_val - min_val)
                else:
                    scaled = grayscale_img
                row_top.append(scaled.astype(np.uint8))
            row_top = np.hstack(row_top)  # => shape (r, 3*c)

            # Prepare bottom row (flow) as colored
            flow_slice = flow_simple_norm[t, :, :, s, :]  # shape: (r, c, 3)
            # We'll apply the 'jet' colormap to each of the 3 channels
            row_bottom_list = []
            for ch in range(3):
                channel_img = flow_slice[:, :, ch]
                colored = plt.cm.jet(channel_img)[..., :3] * 255  # shape (r, c, 3)
                colored = colored.astype(np.uint8)
                row_bottom_list.append(colored)
            row_bottom_colored = np.hstack(row_bottom_list)  # shape (r, 3*c, 3)

            # Convert row_top to 3-channel grayscale for easy stacking with the colored bottom
            row_top_rgb = np.stack([row_top, row_top, row_top], axis=-1)

            # Stack top (grayscale) and bottom (colored) rows vertically
            frame = np.vstack([row_top_rgb, row_bottom_colored])  # shape ((2*r), (3*c), 3)
            frames.append(frame)

        # Save frames as a GIF for the current slice
        output_path = os.path.join(output_folder, f"spline_{s}.gif")
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
