import os
import zipfile
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import multiprocessing

def delete_folder(folder_path, progress_bar):
    for root, dirs, files in os.walk(folder_path, topdown=False):
        for name in files:
            os.remove(os.path.join(root, name))
            progress_bar.update(1)
        for name in dirs:
            os.rmdir(os.path.join(root, name))
            progress_bar.update(1)
    os.rmdir(folder_path)
    progress_bar.update(1)

def prune_unzipped_folders(zip_folder, unzipped_folder):
    # Get the list of zip files (without extension)
    zip_files = {os.path.splitext(f)[0] for f in os.listdir(zip_folder) if f.endswith('.zip')}
    
    # Get the list of unzipped folders
    unzipped_folders = {f for f in os.listdir(unzipped_folder) if os.path.isdir(os.path.join(unzipped_folder, f))}
    
    # Find unzipped folders that do not have a corresponding zip file
    folders_to_delete = unzipped_folders - zip_files

    # Print the folders that are going to be deleted
    print("The following folders will be deleted:")
    for folder in folders_to_delete:
        print(folder)
    
    # Calculate the total number of files and directories to delete
    total_items_to_delete = sum(len(files) + len(dirs) + 1 for folder in folders_to_delete for _, dirs, files in os.walk(os.path.join(unzipped_folder, folder)))
    
    # Delete the unzipped folders that do not have a corresponding zip file
    with tqdm(total=total_items_to_delete, desc="Deleting items", unit="item") as progress_bar:
        with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
            futures = []
            for folder in folders_to_delete:
                folder_path = os.path.join(unzipped_folder, folder)
                folder_path = os.path.join(unzipped_folder, folder)
                progress_bar.set_description(f"Deleting folder: {folder}")
                futures.append(executor.submit(delete_folder, folder_path, progress_bar))
            for future in futures:
                future.result()

# Example usage
if __name__ == "__main__":
    zip_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/zipped_images/'
    unzipped_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/'
    prune_unzipped_folders(zip_folder, unzipped_folder)