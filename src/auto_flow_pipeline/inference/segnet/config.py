# segnet/config.py

import os

###############################################################################
# Paths / Model Files
###############################################################################
MODEL_FOLDER = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/final_models"
MODEL_NAME = "segnet.hdf5"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

###############################################################################
# Other
###############################################################################
# If you have multiple “modes” (e.g., development vs. production),
# you can keep separate booleans or strings:
MODE = "production"  # or "development"
