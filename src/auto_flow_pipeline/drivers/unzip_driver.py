import os
from auto_flow_pipeline.data_io.unzip import unzip_folder

def main():
    # Define the paths for the input folder containing archive files and the output folder
    zip_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/zipped_images/'
    unzipped_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/'
    overwrite = False

    # Create the output folder if it does not exist
    if not os.path.exists(unzipped_folder):
        os.makedirs(unzipped_folder)

    # Extract all archive files in the input folder to the output folder
    unzip_folder(zip_folder, unzipped_folder, overwrite)

# Example usage
if __name__ == "__main__":
    main()