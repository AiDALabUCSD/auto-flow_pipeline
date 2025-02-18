import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
from joblib import Parallel, delayed
from tqdm import tqdm
from auto_flow_pipeline.data_io.logging_setup import setup_logger

def generate_gif_from_preprocessed_nifti(nifti_path, output_path, logger, timepoint=3, n_jobs=-1):
    """
    Generate a GIF scrolling through the preprocessed image stack from top to bottom at a specific timepoint.
    
    Parameters:
    nifti_path (str): Path to the preprocessed NIfTI file.
    output_path (str): Output path for the GIF file.
    logger (logging.Logger): Logger instance.
    timepoint (int): Timepoint to use for generating the GIF (default is 3).
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    try:
        logger.info(f"Loading preprocessed NIfTI file from {nifti_path}")
        nifti_img = nib.load(nifti_path)
        data = nifti_img.get_fdata()
        num_slices = data.shape[3]  # The slices are in the 4th dimension
        
        def process_slice(i):
            fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
            ax.imshow(data[timepoint, :, :, i, 0], cmap='gray')  # Using the specified timepoint
            ax.axis('off')
            plt.subplots_adjust(left=0, right=1, top=1, bottom=0)  # Remove padding
            fig.canvas.draw()
            
            # Extract buffer and reshape correctly (RGBA -> RGB)
            image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))  # (RGBA)
            image = image[..., :3]  # Convert RGBA to RGB
            
            plt.close(fig)
            return image
        
        images = Parallel(n_jobs=n_jobs)(delayed(process_slice)(i) for i in tqdm(range(num_slices), desc="Generating GIF"))
        
        imageio.mimsave(output_path, images, duration=0.1)  # Save as GIF
        logger.info(f"GIF saved to {output_path}")
    except Exception as e:
        logger.error(f"Error generating GIF from preprocessed NIfTI file: {e}")
        raise
