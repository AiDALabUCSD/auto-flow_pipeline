import imageio
import numpy as np

def generate_gif_over_time(sliced_array: np.ndarray, output_path: str, duration: float = 0.5):
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
    imageio.mimsave(output_path, frames, duration=duration, loop=0)

def generate_gif_over_slices_and_time(sliced_array: np.ndarray, output_path: str, duration: float = 0.5):
    """
    Generates a GIF from a 4D array of shape (height, width, num_slices, num_timepoints).
    For each timepoint, the individual slices are tiled horizontally to mimic subplots.
    
    Parameters:
        sliced_array (np.ndarray): 4D array with shape (height, width, num_slices, num_timepoints).
        output_path (str): File path to save the generated GIF.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    frames = []
    
    # If the data is not 8-bit, rescale it to the [0, 255] range.
    arr = sliced_array.copy()
    if arr.dtype != np.uint8:
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.astype(np.uint8)
    
    num_timepoints = arr.shape[3]
    num_slices = arr.shape[2]
    
    for t in range(num_timepoints):
        # For the current time point, extract each slice (2D image).
        slice_images = []
        for s in range(num_slices):
            img = arr[:, :, s, t]  # shape: (256,256)
            slice_images.append(img)
        
        # Tile the slices horizontally to create a composite image.
        # The resulting image will have shape (256, 256 * num_slices)
        composite_image = np.hstack(slice_images)
        frames.append(composite_image)
    
    # Save the frames as a GIF.
    imageio.mimsave(output_path, frames, duration=duration, loop=0)

def generate_gif_velocity_subplots(velocity_array: np.ndarray, output_path: str, duration: float = 0.5):
    """
    Generates a GIF from a 5D array of shape (256,256,5,20,3) where:
      - 256 x 256 is the image resolution.
      - 5 is the number of slices.
      - 20 is the number of timepoints.
      - 3 corresponds to the x, y, and z velocity components.
    
    For each timepoint, the function creates a composite image with 4 rows and 5 columns:
      - Columns: Each column corresponds to a slice.
      - Row 1: Velocity x component.
      - Row 2: Velocity y component.
      - Row 3: Velocity z component.
      - Row 4: Speed calculated from the velocity vector.
    
    Parameters:
        velocity_array (np.ndarray): 5D array with shape (256,256,5,20,3).
        output_path (str): File path to save the generated GIF.
        duration (float): Duration (in seconds) for each frame in the GIF.
    """
    frames = []
    
    # Normalize the entire array to [0, 255] if not already 8-bit.
    arr = velocity_array.copy().astype(np.float32)
    if arr.dtype != np.uint8 or arr.max() > 255:
        arr = 255 * (arr - arr.min()) / (arr.max() - arr.min())
        arr = arr.astype(np.uint8)
    
    height, width, num_slices, num_timepoints, num_channels = arr.shape
    # For each time point build a composite image with 4 rows (velocity components and speed) and 5 columns (slices)
    for t in range(num_timepoints):
        composite_rows = []
        # Process each velocity component (x, y, z)
        for comp in range(3):
            row_images = []
            for s in range(num_slices):
                # Extract the 2D image for a given slice, time, and velocity component.
                img = arr[:, :, s, t, comp]
                row_images.append(img)
            # Tile the images horizontally (5 columns)
            composite_row = np.hstack(row_images)
            composite_rows.append(composite_row)
        
        # Calculate the speed for each slice and tile them.
        speed_row_images = []
        for s in range(num_slices):
            # Get the velocity vector for the given slice and time.
            # Convert to float32 for computation.
            v = arr[:, :, s, t, :].astype(np.float32)
            speed = np.sqrt(np.sum(v**2, axis=-1))
            # Normalize speed to [0,255] per slice.
            sp_min = speed.min()
            sp_max = speed.max()
            if sp_max - sp_min > 0:
                speed_norm = 255 * (speed - sp_min) / (sp_max - sp_min)
            else:
                speed_norm = speed
            speed_norm = speed_norm.astype(np.uint8)
            speed_row_images.append(speed_norm)
        composite_speed_row = np.hstack(speed_row_images)
        composite_rows.append(composite_speed_row)
        
        # Stack all four rows vertically to produce the composite frame.
        composite_frame = np.vstack(composite_rows)
        frames.append(composite_frame)
    
    # Save the frames as a looping GIF.
    imageio.mimsave(output_path, frames, duration=duration, loop=0)