import numpy as np
import matplotlib.pyplot as plt
import imageio
import os
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def generate_combined_gif_with_segmentation(
    array1: np.ndarray,
    array2: np.ndarray,
    segmentation: np.ndarray,
    output_path: str,
    duration: float = 0.5,
    value_range: tuple = None
):
    """
    Generates a GIF with four rows for each timepoint, incorporating segmentation overlay:
      - Row 1: Tiles the slices of the first array (grayscale) with segmentation overlay.
      - Row 2: Tiles the slices of the second array (colormapped) with segmentation overlay.
      - Row 3: Overlays the first (grayscale) and second (colored) arrays with segmentation overlay.
      - Row 4: A horizontal colorbar for the second array.
    
    Parameters:
      array1 (np.ndarray): 4D array (H, W, num_slices, num_timepoints) for the base images.
      array2 (np.ndarray): 4D array (H, W, num_slices, num_timepoints) for the overlay images.
      segmentation (np.ndarray): 4D array (H, W, num_slices, num_timepoints) for segmentation masks.
      output_path (str): Path to save the generated GIF.
      duration (float): Duration (in seconds) for each frame.
      value_range (tuple): Optional (min, max) for normalization of the second array.
    """
    frames = []
    
    # Process array1: scale to 8-bit grayscale
    arr1 = array1.copy()
    if arr1.dtype != np.uint8:
        arr1 = 255 * (arr1 - arr1.min()) / (arr1.max() - arr1.min())
        arr1 = arr1.astype(np.uint8)
    
    # Process array2: normalize to [0,1] and colormap
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
        row1_images = []  # Grayscale composite with segmentation.
        row2_images = []  # Colormapped composite with segmentation.
        row3_images = []  # Overlay composite with segmentation.
        
        for s in range(num_slices):
            # Row 1: Grayscale image.
            img1 = arr1[:, :, s, t]
            img1_rgb = np.stack([img1, img1, img1], axis=-1)  # Convert grayscale to RGB
            
            # Row 2: Colormapped image.
            img2 = arr2[:, :, s, t]
            img2_colored = plt.cm.jet(img2)[..., :3] * 255  # Apply colormap and remove alpha
            img2_colored = img2_colored.astype(np.uint8)
            
            # Row 3: Overlay composite.
            overlay = (0.7 * img1_rgb + 0.3 * img2_colored).astype(np.uint8)
            
            # Process segmentation overlay (binary mask)
            seg_mask = (segmentation[:, :, s, t] > 0.5).astype(np.uint8)  # Threshold
            seg_overlay_color = np.array([255, 0, 0], dtype=np.uint8)  # Red color for segmentation
            alpha = 0.4  # Transparency level
            
            def apply_overlay(image, mask, color, alpha):
                overlay = image.copy()
                mask_indices = mask > 0
                overlay[mask_indices] = (
                    (1 - alpha) * image[mask_indices] + alpha * color
                ).astype(np.uint8)
                return overlay
            
            # Apply segmentation to all three images
            row1_images.append(apply_overlay(img1_rgb, seg_mask, seg_overlay_color, alpha))
            row2_images.append(apply_overlay(img2_colored, seg_mask, seg_overlay_color, alpha))
            row3_images.append(apply_overlay(overlay, seg_mask, seg_overlay_color, alpha))
        
        composite_row1 = np.hstack(row1_images)
        composite_row2 = np.hstack(row2_images)
        composite_row3 = np.hstack(row3_images)
        
        # Create figure with constrained layout
        fig = plt.figure(figsize=(composite_row1.shape[1] / 100,
                                  (composite_row1.shape[0] * 3 + 40) / 100),
                         dpi=100, constrained_layout=True)
        gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.2])
        
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[1, 0])
        ax2 = fig.add_subplot(gs[2, 0])
        ax3 = fig.add_subplot(gs[3, 0])
        
        ax0.imshow(composite_row1)
        ax0.axis('off')
        ax1.imshow(composite_row2)
        ax1.axis('off')
        ax2.imshow(composite_row3)
        ax2.axis('off')
        
        # Create a horizontal colorbar
        sm = ScalarMappable(cmap='jet', norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, cax=ax3, orientation='horizontal')
        cbar.ax.tick_params(labelsize=8)
        
        # Convert figure to image frame
        fig.canvas.draw()
        buf = fig.canvas.tostring_rgb()
        ncols, nrows = fig.canvas.get_width_height()
        frame = np.frombuffer(buf, dtype=np.uint8).reshape(nrows, ncols, 3)
        frames.append(frame)
        plt.close(fig)
    
    # Save as GIF
    imageio.mimsave(output_path, frames, duration=duration, loop=0)
