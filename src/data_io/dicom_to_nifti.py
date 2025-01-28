import os
import pydicom

def read_dicom(file_path):
	"""
	Reads a DICOM file and returns the dataset.
	
	:param file_path: Path to the DICOM file
	:return: DICOM dataset
	"""
	dataset = pydicom.dcmread(file_path)
	return dataset

def print_dicom_data(dataset):
	"""
	Prints out the data from a DICOM dataset.
	
	:param dataset: DICOM dataset
	"""
	print(dataset)
	
# function that given a high level folder prints out the dicom info from one dicom file in each subfolder
def print_dicom_data_from_folder(folder_path):
    """
    Prints out the data from a DICOM dataset in each subfolder of a given folder.
    
    :param folder_path: Path to the folder containing subfolders with DICOM files
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.dcm'):
                print(file)
                dicom_file_path = os.path.join(root, file)
                dicom_data = read_dicom(dicom_file_path)
                print_dicom_data(dicom_data)
                plot_dicom_image(dicom_data)
                break
            break

# function that prints the image data from a dicom file using matplotlib
def plot_dicom_image(dataset):
    """
    Plots the image data from a DICOM dataset using matplotlib.
    
    :param dataset: DICOM dataset
    """
    import matplotlib.pyplot as plt
    plt.imshow(dataset.pixel_array, cmap=plt.cm.bone)
    plt.show()

# Example usage
if __name__ == "__main__":
	dicom_file_path = "path/to/your/dicom/file.dcm"
	dicom_data = read_dicom(dicom_file_path)
	print_dicom_data(dicom_data)