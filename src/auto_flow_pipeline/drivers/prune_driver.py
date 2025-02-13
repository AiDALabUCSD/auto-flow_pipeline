from auto_flow_pipeline.data_io.prune import prune_folders

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