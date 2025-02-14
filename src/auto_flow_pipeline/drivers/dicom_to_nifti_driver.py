import os
import glob
import logging
from tqdm import tqdm
from auto_flow_pipeline.data_io.dicom_to_nifti import (
    load_4dflow_dataframe, create_volume_arrays, fill_volume_arrays,
    process_corrected_velocity_npy, build_affine, check_orientation_and_flip,
    reconstruct_4dflow_nifti, reconstruct_corrected_velocity_nifti,
    generate_gif_from_nifti, generate_gif_from_velocity_nifti, generate_gif_from_nifti_vel
)

def setup_logger(patient_name, output_folder):
    logger = logging.getLogger(patient_name)
    logger.setLevel(logging.INFO)
    log_file = os.path.join(output_folder, f'{patient_name}', f'{patient_name}.log')
    handler = logging.FileHandler(log_file)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def process_patient(dicom_folder, output_folder, velocity_path, csv_path, logger):
    # Load the 4D flow dataframe
    df_4dflow = load_4dflow_dataframe(csv_path)

    # Create and fill the arrays with the 4D flow data pulled from the DICOM files
    mag_4d, vel_5d = create_volume_arrays(df_4dflow)
    fill_volume_arrays(df_4dflow, mag_4d, vel_5d, n_jobs=-1)

    # Create a correctly oriented array filled with the corrected velocity data downloaded from Tempus
    RDIM, CDIM, SDIM, TDIM = vel_5d.shape[:4]
    corrected_vel_5d = process_corrected_velocity_npy(velocity_path, RDIM, CDIM, SDIM, TDIM)

    # Build the affine transformation matrix
    Nslices = len(df_4dflow['slice_index'].unique())
    A, Ainv, rowres, colres, sthick, slice_spacing = build_affine(df_4dflow, Nslices)

    # Check orientation and flip if necessary
    mag_4d, vel_5d, corrected_vel_5d = check_orientation_and_flip(df_4dflow, mag_4d, vel_5d, corrected_vel_5d)

    # Save the 4D flow data as NIfTI files
    mag_path = os.path.join(output_folder, 'mag_4dflow.nii.gz')
    vel_path = os.path.join(output_folder, 'vel-uncorrected_4dflow.nii.gz')
    reconstruct_4dflow_nifti(mag_4d, vel_5d, A, mag_path, vel_path)
    corrected_vel_nifti_path = os.path.join(output_folder, 'vel_corrected_4dflow.nii.gz')
    reconstruct_corrected_velocity_nifti(corrected_vel_5d, A, corrected_vel_nifti_path)

    # Generate GIFs from the NIfTI files
    gif_path = os.path.join(output_folder, 'mag.gif')
    generate_gif_from_nifti(mag_path, gif_path)

    gif_path = os.path.join(output_folder, 'vel-uncorrected.gif')
    generate_gif_from_velocity_nifti(vel_path, gif_path)

    gif_path = os.path.join(output_folder, 'zvel-uncorrected.gif')
    generate_gif_from_nifti_vel(vel_path, gif_path)

    gif_path = os.path.join(output_folder, 'vel-corrected.gif')
    generate_gif_from_velocity_nifti(corrected_vel_nifti_path, gif_path)

    # Log affine matrix details
    logger.info("Affine matrix A:\n%s", A)
    logger.info("Inverse affine matrix Ainv:\n%s", Ainv)
    logger.info("Row resolution: %s", rowres)
    logger.info("Column resolution: %s", colres)
    logger.info("Slice thickness: %s", sthick)
    logger.info("Slice spacing: %s", slice_spacing)

def main():
    dicom_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images'
    output_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients'
    velocity_base_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities'

    # Process each patient folder
    for patient_folder in tqdm(glob.glob(os.path.join(dicom_base_folder, '*')), desc="Processing patients"):
        patient_name = os.path.basename(patient_folder)
        output_folder = os.path.join(output_base_folder, patient_name)
        velocity_path = os.path.join(velocity_base_folder, f'{patient_name}.npy')
        csv_path = os.path.join(output_folder, "flow_info.csv")

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        logger = setup_logger(patient_name, output_folder)
        logger.info(f"Processing patient: {patient_name}")
        process_patient(patient_folder, output_folder, velocity_path, csv_path, logger)

if __name__ == "__main__":
    main()
