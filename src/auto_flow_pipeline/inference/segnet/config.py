# segnet/config.py

import os

###############################################################################
# Paths / Model Files
###############################################################################
MODEL_FOLDER = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/final_models"
MODEL_NAME = "segnet.hdf5"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

# MODEL_PATH = "/home/ayeluru/4d-flow-automation/segnet/training_outputs/augmented_dyn-rng_trns_rot_through_def_tr-3e-5/b32_e1000_augmented_dyn-rng_trns_rot_through_def_tr-3e-5_chkpt--1000-0.0926-continue-2.hdf5"

###############################################################################
# Other
###############################################################################
# If you have multiple “modes” (e.g., development vs. production),
# you can keep separate booleans or strings:
MODE = "production"  # or "development"
