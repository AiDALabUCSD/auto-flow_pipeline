import os
import numpy as np
from auto_flow_pipeline.visualization.slice_extraction.generate_gifs import generate_combined_gif
import matplotlib.pyplot as plt
import imageio
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

def generate_four_row_gifs_for_slices_w_pred(
    patient_name: str,
    base_path: str,
    output_name: str,
    array: np.ndarray,
    prediction: np.ndarray,
    duration: float = 0.5,
    value_range: tuple = (-1, 1)
) -> None:
    """
    Generate GIFs for each slice from the given array, splitting the channels into two rows:
    - Top row: first 3 channels (mag) shown in grayscale.
    - Bottom row: last 3 channels (through_flow velocity) shown with a 'jet' colormap.
    - Additional rows: top and bottom rows with red segmentation overlay.

    Parameters:
      patient_name (str): The name of the patient.
      base_path (str): The base path where the output folder will be created.
      output_name (str): The name of the output folder.
      array (np.ndarray): The input array of shape (time, r, c, slices, 6).
      prediction (np.ndarray): The segmentation array of shape (time, r, c, slices).
      duration (float): Duration (in seconds) for each GIF frame.
      value_range (tuple): (min, max) range for the flow channels. Default is (-1, 1).
    """
    time, r, c, num_slices, channels = array.shape
    assert channels == 6, "Input array must have 6 channels."
    assert prediction.shape == (time, r, c, num_slices), \
        "Prediction must have shape (time, r, c, slices)."

    # Construct output folder path
    output_folder = os.path.join(base_path, patient_name, output_name)
    os.makedirs(output_folder, exist_ok=True)

    # Separate the first three channels (mag) from the last three (flow)
    mag_data = array[..., :3]   # shape: (time, r, c, slices, 3)
    flow_data = array[..., 3:]  # shape: (time, r, c, slices, 3)

    # Normalize flow_data to [0..1] within the specified value_range
    vmin, vmax = value_range
    flow_simple_norm = np.clip(flow_data, vmin, vmax)
    flow_simple_norm = (flow_simple_norm - vmin) / (vmax - vmin)

    # Loop over each slice
    for s in range(num_slices):
        frames = []
        # Loop over timepoints
        for t in range(time):
            # -------------------------
            # 1) Prepare top row (mag)
            # -------------------------
            mag_slice = mag_data[t, :, :, s, :]  # shape: (r, c, 3)
            
            # Convert each channel to grayscale and concatenate horizontally
            row_top = []
            for ch in range(3):
                grayscale_img = mag_slice[:, :, ch]
                # Scale to 0..255 for display
                min_val, max_val = grayscale_img.min(), grayscale_img.max()
                if max_val - min_val > 0:
                    scaled = 255 * (grayscale_img - min_val) / (max_val - min_val)
                else:
                    scaled = grayscale_img
                row_top.append(scaled.astype(np.uint8))
            row_top = np.hstack(row_top)  # shape: (r, 3*c)

            # ----------------------------
            # 2) Prepare bottom row (flow)
            # ----------------------------
            flow_slice = flow_simple_norm[t, :, :, s, :]  # shape: (r, c, 3)
            row_bottom_list = []
            for ch in range(3):
                channel_img = flow_slice[:, :, ch]
                # Apply the 'jet' colormap (gives shape (r, c, 4), then discard alpha)
                colored = plt.cm.jet(channel_img)[..., :3] * 255
                colored = colored.astype(np.uint8)
                row_bottom_list.append(colored)
            # Concatenate horizontally: shape (r, 3*c, 3)
            row_bottom_colored = np.hstack(row_bottom_list)

            # Convert row_top to 3-channel grayscale for stacking
            row_top_rgb = np.stack([row_top, row_top, row_top], axis=-1)  # (r, 3*c, 3)

            # Stack top (grayscale) and bottom (colored) rows vertically => shape (2*r, 3*c, 3)
            frame_no_seg = np.vstack([row_top_rgb, row_bottom_colored])

            # ---------------------------
            # 3) Overlay segmentation
            # ---------------------------
            row_top_rgb_prediction = row_top_rgb.copy()
            row_bottom_colored_prediction = row_bottom_colored.copy()

            # prediction[t, :, :, s] => shape (r, c)
            predict_slice = prediction[t, :, :, s]

            # Repeat horizontally to match the shape (r, 3*c)
            mask_tiled = np.hstack([predict_slice, predict_slice, predict_slice]) > 0

            # Paint red ([255, 0, 0]) on the overlay
            row_top_rgb_prediction[mask_tiled] = [255, 0, 0]
            row_bottom_colored_prediction[mask_tiled] = [255, 0, 0]

            # Stack those two rows
            frame_with_seg = np.vstack([row_top_rgb_prediction, row_bottom_colored_prediction])

            # Finally, stack the "no seg" and "with seg" vertically
            # => shape (4*r, 3*c, 3)
            frame = np.vstack([frame_no_seg, frame_with_seg])
            frames.append(frame)

        # Save all frames as a GIF
        output_path = os.path.join(output_folder, f"spline_{s}.gif")
        imageio.mimsave(output_path, frames, duration=duration, loop=0)
