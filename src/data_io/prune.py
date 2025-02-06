import os
import zipfile
from tqdm import tqdm

def prune_unzipped_folders(zip_folder, unzipped_folder):
    # Get the list of zip files (without extension)
    zip_files = {os.path.splitext(f)[0] for f in os.listdir(zip_folder) if f.endswith('.zip')}
    
    # Get the list of unzipped folders
    unzipped_folders = {f for f in os.listdir(unzipped_folder) if os.path.isdir(os.path.join(unzipped_folder, f))}
    
    # Find unzipped folders that do not have a corresponding zip file
    folders_to_delete = unzipped_folders - zip_files
    
    # Delete the unzipped folders that do not have a corresponding zip file
    for folder in tqdm(folders_to_delete, desc="Deleting folders", unit="folder"):
        folder_path = os.path.join(unzipped_folder, folder)
        for root, dirs, files in os.walk(folder_path, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))
        os.rmdir(folder_path)

# Example usage
if __name__ == "__main__":
    zip_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/zipped_images/'
    unzipped_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/'
    prune_unzipped_folders(zip_folder, unzipped_folder)