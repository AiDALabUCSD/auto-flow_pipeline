import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import imageio
from joblib import Parallel, delayed
from tqdm import tqdm

def generate_gif_from_nifti(nifti_path, output_path, logger, n_jobs=-1):
    """
    Generate a GIF from a NIfTI file.
    
    Parameters:
    nifti_path (str): Path to the NIfTI file.
    output_path (str): Output path for the GIF file.
    logger (logging.Logger): Logger instance.
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    # Generate a GIF from a NIfTI file
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    num_slices = data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(data[:, :, i, 0], cmap='gray')  # Using the first time point
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

def generate_gif_from_nifti_vel(nifti_path, output_path, logger, vel_dir=2, n_jobs=-1):
    """
    Generate a GIF from a NIfTI file for velocity data.
    
    Parameters:
    nifti_path (str): Path to the NIfTI file.
    output_path (str): Output path for the GIF file.
    logger (logging.Logger): Logger instance.
    vel_dir (int): Velocity direction (default is 2).
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    # Generate a GIF from a NIfTI file for velocity data
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    num_slices = data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(data[:, :, i, 0, vel_dir], cmap='jet')  # Using the first time point
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

def compute_speed_from_velocity_nifti(nifti_path):
    """
    Compute the speed magnitude from velocity NIfTI data.
    
    Parameters:
    nifti_path (str): Path to the NIfTI file.
    
    Returns:
    np.ndarray: Speed magnitude array.
    """
    # Compute the speed magnitude from velocity NIfTI data
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    speed = np.sqrt(np.sum(data ** 2, axis=-1))  # Compute speed magnitude
    return speed

def generate_gif_from_velocity_nifti(nifti_path, output_path, logger, n_jobs=-1):
    """
    Generate a GIF from velocity NIfTI data.
    
    Parameters:
    nifti_path (str): Path to the NIfTI file.
    output_path (str): Output path for the GIF file.
    logger (logging.Logger): Logger instance.
    n_jobs (int): Number of parallel jobs. Use -1 for all available cores.
    """
    # Generate a GIF from velocity NIfTI data
    nifti_img = nib.load(nifti_path)
    data = nifti_img.get_fdata()
    speed_data = np.sqrt(np.sum(data ** 2, axis=-1))  # Compute speed magnitude
    num_slices = speed_data.shape[2]
    
    def process_slice(i):
        fig, ax = plt.subplots(figsize=(5, 5))  # Adjust figure size
        ax.imshow(speed_data[:, :, i, 0], cmap='jet')  # Using the first time point
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