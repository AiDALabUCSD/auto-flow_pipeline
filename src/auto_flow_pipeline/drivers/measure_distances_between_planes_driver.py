import os
import pandas as pd
from auto_flow_pipeline.data_io.catalogue_patients import get_aortic_distances, get_pulmonary_distances
from auto_flow_pipeline import main_logger

def consolidate_distances(base_folderpath: str, output_aortic_csv: str, output_pulmonary_csv: str):
    """
    Consolidates aortic and pulmonary distances for all patients into separate CSV files.

    Parameters:
        base_folderpath (str): Path to the folder where patient data is stored.
        output_aortic_csv (str): Path to save the consolidated aortic distances CSV file.
        output_pulmonary_csv (str): Path to save the consolidated pulmonary distances CSV file.
    """
    main_logger.info("Starting consolidation of distances between planes.")
    patient_dirs = [
        d for d in os.listdir(base_folderpath)
        if os.path.isdir(os.path.join(base_folderpath, d))
    ]
    main_logger.info(f"Found {len(patient_dirs)} patient directories.")

    aortic_distances = {}
    pulmonary_distances = {}

    aortic_indices = [5, 10, 15, 20, 25]
    pulmonary_indices = [5, 15, 25, 35, 45]

    for patient in patient_dirs:
        try:
            main_logger.info(f"Processing distances for patient: {patient}")
            aortic_distances[patient] = get_aortic_distances(patient, base_folderpath, aortic_indices)
            pulmonary_distances[patient] = get_pulmonary_distances(patient, base_folderpath, pulmonary_indices)
        except Exception as e:
            main_logger.error(f"Error processing distances for patient {patient}: {e}")

    # Save aortic distances to CSV
    aortic_df = pd.DataFrame.from_dict(aortic_distances, orient='index', columns=[f"A{i}" for i in range(1, len(aortic_indices) + 1)])
    aortic_df.to_csv(output_aortic_csv, index_label="patient_id")
    main_logger.info(f"Saved aortic distances to {output_aortic_csv}")

    # Save pulmonary distances to CSV
    pulmonary_df = pd.DataFrame.from_dict(pulmonary_distances, orient='index', columns=[f"P{i}" for i in range(1, len(pulmonary_indices) + 1)])
    pulmonary_df.to_csv(output_pulmonary_csv, index_label="patient_id")
    main_logger.info(f"Saved pulmonary distances to {output_pulmonary_csv}")

def main():
    base_folderpath = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    output_folder = os.path.dirname(base_folderpath)  # Folder above base_folderpath
    output_aortic_csv = os.path.join(output_folder, "distance_between_aortic_measurements.csv")
    output_pulmonary_csv = os.path.join(output_folder, "distance_between_pulmonary_artery_measurements.csv")

    consolidate_distances(base_folderpath, output_aortic_csv, output_pulmonary_csv)

if __name__ == "__main__":
    main()
