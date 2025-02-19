import os
import tensorflow as tf
from tensorflow.keras.models import load_model

# Import your custom loss function
from .custom_loss import custom_mse
# If you're using Python-based config
from .config import MODEL_PATH
# Memory growth utility
from .utils import set_tf_memory_growth

def load_locnet(checkpoint_path=None):
    """
    Loads the LocNet model from a given checkpoint path.
    If no path is provided, uses the default in MODEL_PATH.
    """

    # Make sure GPU memory growth is enabled before loading the model
    set_tf_memory_growth()

    if checkpoint_path is None:
        checkpoint_path = MODEL_PATH

    # Pass custom_objects so Keras knows about your custom loss function
    model = load_model(
        checkpoint_path,
        custom_objects={"custom_mse": custom_mse}
    )
    print(f"Loaded LocNet model from: {checkpoint_path}")

    return model
