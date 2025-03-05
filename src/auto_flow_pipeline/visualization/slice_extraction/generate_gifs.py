import imageio
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import matplotlib.gridspec as gridspec

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

def generate_gif_with_colormap_and_colorbar(sliced_array: np.ndarray, output_path: str, duration: float = 0.5, value_range: tuple = None):
    """
    Generates a GIF from a 4D array of shape (height, width, num_slices, num_timepoints) with a 'jet' colormap and a colorbar.
    
    Parameters:
        sliced_array (np.ndarray): 4D array with shape (height, width, num_slices, num_timepoints).
        output_path (str): File path to save the generated GIF.
        duration (float): Duration (in seconds) for each frame in the GIF.
        value_range (tuple): Optional tuple (min, max) to set the range for normalization. Values outside this range are clipped.
    """
    frames = []
    
    # Normalize the data to the [0, 1] range.
    arr = sliced_array.copy().astype(np.float32)
    
    if value_range is not None:
        vmin, vmax = value_range
        arr = np.clip(arr, vmin, vmax)
    else:
        vmin, vmax = arr.min(), arr.max()
    
    norm = Normalize(vmin=vmin, vmax=vmax)
    arr = norm(arr)
    
    num_timepoints = arr.shape[3]
    num_slices = arr.shape[2]
    
    for t in range(num_timepoints):
        # For the current time point, extract each slice (2D image).
        slice_images = []
        for s in range(num_slices):
            img = arr[:, :, s, t]
            # Apply the 'jet' colormap.
            img_colored = plt.cm.jet(img)
            # Convert to 8-bit RGB.
            img_colored = (img_colored[:, :, :3] * 255).astype(np.uint8)
            slice_images.append(img_colored)
        
        # Tile the slices horizontally to create a composite image.
        composite_image = np.hstack(slice_images)
        
        # Create a figure with the composite image and a colorbar.
        fig, ax = plt.subplots(figsize=(composite_image.shape[1] / 100, composite_image.shape[0] / 100), dpi=100)
        ax.imshow(composite_image)
        ax.axis('off')
        
        # Add a colorbar.
        sm = ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        
        # Save the figure to a temporary buffer.
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        
        frames.append(frame)
        plt.close(fig)
    
    # Save the frames as a GIF.
    imageio.mimsave(output_path, frames, duration=duration, loop=0)

def generate_combined_gif(array1: np.ndarray,
                          array2: np.ndarray,
                          output_path: str,
                          duration: float = 0.5,
                          value_range: tuple = None):
    """
    Generates a GIF with four rows for each timepoint:
      - Row 1: Tiles the slices of the first array (grayscale).
      - Row 2: Tiles the slices of the second array (normalized and colormapped with 'jet').
      - Row 3: Overlays the first (grayscale) and second (colored) arrays.
      - Row 4: A horizontal colorbar for the second array.
    
    Uses constrained_layout (via fig.add_gridspec) so that the colorbar tick labels are not cropped.
    
    Parameters:
      array1 (np.ndarray): 4D array (H, W, num_slices, num_timepoints) for the base images.
      array2 (np.ndarray): 4D array (H, W, num_slices, num_timepoints) for the overlay images.
      output_path (str): Path to save the generated GIF.
      duration (float): Duration (in seconds) for each frame.
      value_range (tuple): Optional (min, max) for normalization of the second array.
    """
    frames = []
    
    # Process array1: scale to 8-bit if needed.
    arr1 = array1.copy()
    if arr1.dtype != np.uint8:
        arr1 = 255 * (arr1 - arr1.min()) / (arr1.max() - arr1.min())
        arr1 = arr1.astype(np.uint8)
    
    # Process array2: convert to float32 and normalize to [0,1].
    arr2 = array2.copy().astype(np.float32)
    if value_range is not None:
        vmin, vmax = value_range
        arr2 = np.clip(arr2, vmin, vmax)
    else:
        vmin, vmax = arr2.min(), arr2.max()
    norm = Normalize(vmin=vmin, vmax=vmax)
    arr2 = norm(arr2)
    
    H, W, num_slices, num_timepoints = arr1.shape
    
    for t in range(num_timepoints):
        row1_images = []  # Grayscale composite from array1.
        row2_images = []  # Colormapped composite from array2.
        row3_images = []  # Overlay composite.
        
        for s in range(num_slices):
            # Row 1: Grayscale image.
            img1 = arr1[:, :, s, t]
            row1_images.append(img1)
            
            # Row 2: Colormapped image.
            img2 = arr2[:, :, s, t]
            img2_colored = plt.cm.jet(img2)  # returns (H, W, 4)
            img2_colored = (img2_colored[..., :3] * 255).astype(np.uint8)
            row2_images.append(img2_colored)
            
            # Row 3: Overlay composite.
            img1_rgb = np.stack([img1, img1, img1], axis=-1)
            overlay = (0.7 * img1_rgb + 0.3 * img2_colored).astype(np.uint8)
            row3_images.append(overlay)
        
        composite_row1 = np.hstack(row1_images)
        composite_row2 = np.hstack(row2_images)
        composite_row3 = np.hstack(row3_images)
        
        # Create a figure using constrained_layout by using fig.add_gridspec.
        fig = plt.figure(figsize=(composite_row1.shape[1] / 100,
                                    (composite_row1.shape[0]*3 + 40) / 100),
                         dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[3, 0])
        
        ax0.imshow(composite_row1, cmap='gray')
        ax0.axis('off')
        ax1.imshow(composite_row2)
        ax1.axis('off')
        ax2.imshow(composite_row3)
        ax2.axis('off')
        
        # Create a horizontal colorbar in the fourth row.
        sm = ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax3, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        frames.append(frame)
        plt.close(fig)
    
    imageio.mimsave(output_path, frames, duration=duration, loop=0)