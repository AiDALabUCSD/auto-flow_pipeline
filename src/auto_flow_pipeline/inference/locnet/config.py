# locnet/config.py

import os

###############################################################################
# Paths / Model Files
###############################################################################
MODEL_FOLDER = "/home/ayeluru/mnt/maxwell/projects/Aorta_pulmonary_artery_localization/final_models"
MODEL_NAME = "locnet.hdf5"
MODEL_PATH = os.path.join(MODEL_FOLDER, MODEL_NAME)

###############################################################################
# Hyperparameters / Inference Settings
###############################################################################
BATCH_SIZE = 8  # example
USE_GPU = True  # for reference; actual GPU usage is handled by conda env + tf.config
THRESHOLD = 0.5 # or any other threshold / scalar you might apply

###############################################################################
# Other
###############################################################################
# If you have multiple “modes” (e.g., development vs. production),
# you can keep separate booleans or strings:
MODE = "production"  # or "development"
