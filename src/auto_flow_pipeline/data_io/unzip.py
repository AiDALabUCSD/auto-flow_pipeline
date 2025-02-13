import zipfile
import tarfile
import os
import shutil
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

def extract_file(input_file, output_dir, overwrite=False):
    """
    Extracts the contents of an archive file to the specified output directory.
    
    Args:
        input_file (str): Path to the archive file.
        output_dir (str): Directory where the contents will be extracted.
        overwrite (bool): Whether to overwrite existing directories.
    
    Returns:
        str: Status of the extraction ('success', 'ERROR', 'skip').
    """
    try:
        if input_file.endswith('.zip'):
            with zipfile.ZipFile(input_file, 'r') as zip_ref:
                return extract_to_directory(zip_ref, input_file, output_dir, overwrite)
        elif input_file.endswith('.tar') or input_file.endswith('.tar.gz') or input_file.endswith('.tgz'):
            with tarfile.open(input_file, 'r') as tar_ref:
                return extract_to_directory(tar_ref, input_file, output_dir, overwrite)
        else:
            tqdm.write(f"Unsupported file type: {input_file}")
            return 'ERROR'
    except (zipfile.BadZipFile, tarfile.TarError) as e:
        tqdm.write(f"Error: {input_file} is not a valid archive file. {e}")
        return 'ERROR'
    except FileNotFoundError:
        tqdm.write(f"Error: {input_file} does not exist.")
        return 'ERROR'
    except Exception as e:
        tqdm.write(f"An error occurred: {e}")
        return 'ERROR'

def extract_to_directory(archive_ref, input_file, output_dir, overwrite):
    """
    Extracts the contents of an archive reference to a directory.
    
    Args:
        archive_ref: Archive reference (zipfile.ZipFile or tarfile.TarFile).
        input_file (str): Path to the archive file.
        output_dir (str): Directory where the contents will be extracted.
        overwrite (bool): Whether to overwrite existing directories.
    
    Returns:
        str: Status of the extraction ('success', 'fail', 'skip').
    """
    # Extract the archive name from the input file name
    archive_name = os.path.splitext(os.path.basename(input_file))[0]
    # Create a folder with the archive name in the output directory
    archive_dir = os.path.join(output_dir, archive_name)
    
    if not overwrite and os.path.exists(archive_dir):
        tqdm.write(f"Skipping extraction of {input_file} as {archive_dir} already exists.")
        return 'skip'
    
    if os.path.exists(archive_dir):
        shutil.rmtree(archive_dir)
    
    os.makedirs(archive_dir, exist_ok=True)
    
    # Extract the contents of the archive file to the archive folder
    archive_ref.extractall(archive_dir)
    
    # Walk through the extracted files and extract any additional archive files
    for root, dirs, files in os.walk(archive_dir):
        for file in files:
            if file.endswith('.zip') or file.endswith('.tar') or file.endswith('.tar.gz') or file.endswith('.tgz'):
                archive_file = os.path.join(root, file)
                extract_file(archive_file, root)
                os.remove(archive_file)
    
    tqdm.write(f"Successfully extracted {input_file} to {archive_dir}")
    return 'success'

def unzip_folder(folder_path, output_dir, overwrite=False):
    """
    Extracts all archive files in a folder to the specified output directory.
    
    Args:
        folder_path (str): Path to the folder containing archive files.
        output_dir (str): Directory where the contents will be extracted.
        overwrite (bool): Whether to overwrite existing directories.
    """
    archive_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip') or file.endswith('.tar') or file.endswith('.tar.gz') or file.endswith('.tgz'):
                archive_files.append(os.path.join(root, file))
    
    status_dict = {}
    with tqdm(total=len(archive_files), desc="Extracting archives", unit="file") as progress_bar:
        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(extract_file, archive_file, output_dir, overwrite): archive_file for archive_file in archive_files}
            for future in futures:
                archive_file = futures[future]
                status = future.result()
                status_dict[os.path.basename(archive_file)] = status
                progress_bar.update(1)
    
    # Print the status of all the potential unzips
    for archive_name, status in status_dict.items():
        tqdm.write(f"{archive_name}: [{status}]")

# Example usage
if __name__ == "__main__":
    # Define the paths for the input folder containing archive files and the output folder
    zip_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/zipped_images/'
    unzipped_folder = '/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/unzipped_images/'
    overwrite = False

    # Create the output folder if it does not exist
    if not os.path.exists(unzipped_folder):
        os.makedirs(unzipped_folder)

    # Extract all archive files in the input folder to the output folder
    unzip_folder(zip_folder, unzipped_folder, overwrite)