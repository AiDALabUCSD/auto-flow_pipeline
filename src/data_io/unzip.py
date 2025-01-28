import zipfile
import sys
import os
import shutil

def unzip_file(input_file, output_dir, overwrite=False):
    try:
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            # Extract the patient name from the input file name
            patient_name = os.path.splitext(os.path.basename(input_file))[0]
            # Create a folder with the patient name in the output directory
            patient_dir = os.path.join(output_dir, patient_name)
            
            if not overwrite and os.path.exists(patient_dir):
                print(f"Skipping extraction of {input_file} as {patient_dir} already exists.")
                return
            
            if os.path.exists(patient_dir):
                shutil.rmtree(patient_dir)
            
            os.makedirs(patient_dir, exist_ok=True)
            
            # Extract the contents of the zip file to the patient folder
            zip_ref.extractall(patient_dir)
            
            # Walk through the extracted files and extract any additional zip files
            for root, dirs, files in os.walk(patient_dir):
                for file in files:
                    if file.endswith('.zip'):
                        zip_file = os.path.join(root, file)
                        unzip_file(zip_file, root)
                        os.remove(zip_file)
            
            print(f"Successfully extracted {input_file} to {patient_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {input_file} is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def unzip_folder(folder_path, output_dir, overwrite=False):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.zip'):
                zip_file = os.path.join(root, file)
                unzip_file(zip_file, output_dir, overwrite)

def main():
    if len(sys.argv) != 4:
        print("Usage: python unzip.py <input_folder> <output_dir> <true_or_false>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]
    true_or_false = sys.argv[3].lower() == 'true'

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unzip_folder(input_file, output_dir, true_or_false)

if __name__ == "__main__":
    main()