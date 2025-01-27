import zipfile
import sys
import os

def unzip_file(input_file, output_dir):
    try:
        with zipfile.ZipFile(input_file, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print(f"Successfully extracted {input_file} to {output_dir}")
    except zipfile.BadZipFile:
        print(f"Error: {input_file} is not a valid zip file.")
    except FileNotFoundError:
        print(f"Error: {input_file} does not exist.")
    except Exception as e:
        print(f"An error occurred: {e}")

def main():
    if len(sys.argv) != 3:
        print("Usage: python unzip.py <input_file> <output_dir>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    unzip_file(input_file, output_dir)

if __name__ == "__main__":
    main()