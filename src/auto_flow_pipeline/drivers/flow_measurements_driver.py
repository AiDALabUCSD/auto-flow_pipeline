import os
import pandas as pd
from auto_flow_pipeline.data_io.catalogue_patients import get_flow_measurements
from auto_flow_pipeline import main_logger
from concurrent.futures import ProcessPoolExecutor, as_completed

def process_patient_flow_measurements(pid, base_output_folder):
    try:
        flow_measurements = get_flow_measurements(pid, base_output_folder)
        return pid, flow_measurements, None
    except Exception as e:
        return pid, None, e

def consolidate_flow_measurements(base_output_folder: str, output_path: str):
    """
    Consolidates flow measurements for all patients into a single DataFrame and saves it.

    Parameters:
        base_output_folder (str): Path to the folder where patient data is stored.
        output_path (str): Path to save the consolidated flow measurements CSV file.
    """
    try:
        main_logger.info("Starting consolidation of flow measurements.")
        
        patient_dirs = [
            d for d in os.listdir(base_output_folder)
            if os.path.isdir(os.path.join(base_output_folder, d))
        ]
        main_logger.info(f"Found {len(patient_dirs)} patient directories.")

        flow_measurements_list = []

        with ProcessPoolExecutor() as executor:
            futures = {executor.submit(process_patient_flow_measurements, pid, base_output_folder): pid for pid in patient_dirs}
            for future in as_completed(futures):
                pid = futures[future]
                try:
                    pid, flow_measurements, error = future.result()
                    if error:
                        raise error
                    main_logger.info(f"Processing flow measurements for patient: {pid}")
                    flow_measurements_list.append(flow_measurements)
                except Exception as e:
                    main_logger.error(f"Error processing flow measurements for patient {pid}: {e}")

        flow_measurements_df = pd.DataFrame(flow_measurements_list, columns=['patient_id','Ao_auto', 'Ao_auto_std',
                                                                             'PA_auto', 'PA_auto_std', 'Qp/Qs_auto',
                                                                             'Qp/Qs_auto_std', 'A1', 'A2', 'A3','A4',
                                                                             'A5', 'P1', 'P2', 'P3', 'P4', 'P5'])
        flow_measurements_df.to_csv(output_path, index=False)
        main_logger.info(f"Saved consolidated flow measurements to {output_path}")
    except Exception as e:
        main_logger.error(f"Failed to consolidate flow measurements: {e}")

def main():
    base_output_folder = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/patients"
    output_path = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/ge_testing/flow_measurements_autoflow_ge.csv"
    
    consolidate_flow_measurements(base_output_folder, output_path)

if __name__ == "__main__":
    main()
