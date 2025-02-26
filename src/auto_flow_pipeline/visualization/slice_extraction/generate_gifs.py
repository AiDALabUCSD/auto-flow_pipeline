import imageio
import numpy as np

def generate_gif_from_slices(sliced_array: np.ndarray, output_path: str, duration: float = 0.5):
    """
    Generates a GIF from a 3D array of shape (height, width, num_timepoints).
    
    Parameters:
        sliced_array (np.ndarray): 3D array with shape (height, width, num_timepoints).
        output_path (str): File path to save the generated GIF.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    frames = []
    
    # If the data is not already 8-bit, rescale it to the [0, 255] range.
    arr = sliced_array.copy()
    if arr.dtype != np.uint8:
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.astype(np.uint8)
    
    num_timepoints = arr.shape[2]
    for t in range(num_timepoints):
        # Each frame is a 2D image for timepoint t.
        frame = arr[:, :, t]
        frames.append(frame)
    
    # Save frames as a GIF.
    imageio.mimsave(output_path, frames, duration=duration)