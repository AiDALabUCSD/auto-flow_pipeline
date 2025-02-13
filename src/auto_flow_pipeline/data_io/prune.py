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

def prune_folders(reference_folder, target_folders):
    # Get the list of zip files (without extension)
    zip_files = {os.path.splitext(f)[0] for f in os.listdir(reference_folder) if f.endswith('.zip')}
    
    for target_folder in target_folders:
        # Get the list of items in the target folder
        target_items = {os.path.splitext(f)[0] if f.endswith('.npy') else f for f in os.listdir(target_folder) if os.path.isdir(os.path.join(target_folder, f)) or f.endswith('.npy')}
        
        # Find items in the target folder that do not have a corresponding zip file
        items_to_delete = target_items - zip_files

        # Find items in the zip folder that do not have a corresponding item in the target folder
        missing_items = zip_files - target_items

        # Print the items that are going to be deleted
        if items_to_delete:
            print(f"\nThe following items in {target_folder} will be deleted: {', '.join(items_to_delete)}")
        else:
            print(f"\nNo items to delete in {target_folder}.")
        
        # Print the items that are in the zip folder but not in the target folder
        if missing_items:
            print(f"\nThe following items are in the zip folder but not in {target_folder}: {', '.join(missing_items)}")
        else:
            print(f"\nAll items in the zip folder are present in {target_folder}.")
        
        # Calculate the total number of files and directories to delete
        total_items_to_delete = sum(len(files) + len(dirs) + 1 for item in items_to_delete for _, dirs, files in os.walk(os.path.join(target_folder, item)))
        
        # Delete the items that do not have a corresponding zip file
        with tqdm(total=total_items_to_delete, desc="Deleting items", unit="item") as progress_bar:
            with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
                futures = []
                for item in items_to_delete:
                    item_path = os.path.join(target_folder, item)
                    if os.path.isdir(item_path):
                        progress_bar.set_description(f"Deleting folder: {item}")
                        futures.append(executor.submit(delete_folder, item_path, progress_bar))
                    elif item_path.endswith('.npy'):
                        os.remove(item_path)
                        progress_bar.update(1)
                for future in futures:
                    future.result()

def main():
    reference_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/zipped_images/'
    target_folders = [
        '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/',
        '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/velocities/'
    ]
    prune_folders(reference_folder, target_folders)

# Example usage
if __name__ == "__main__":
    main()