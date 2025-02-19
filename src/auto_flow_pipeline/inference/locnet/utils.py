import tensorflow as tf

def set_tf_memory_growth():
    """
    Enables memory growth for all detected GPUs, so TensorFlow
    doesn't pre-allocate the entire GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        print("GPU(s) detected:", gpus)
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print("Enabled memory growth on:", gpu)
        except RuntimeError as e:
            print("Memory growth must be set at program startup:", e)
    else:
        print("No GPU detected or GPU not available.")
