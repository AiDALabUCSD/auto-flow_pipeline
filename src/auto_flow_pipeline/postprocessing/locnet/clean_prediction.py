import numpy as np

def center_box_clean(prd, b=45):
    """
    Clean the prediction by zeroing out the boundaries of the box.

    Parameters:
    prd (numpy.ndarray): The prediction array to be cleaned.
    b (int, optional): The size of the boundary to be zeroed out. Default is 45.

    Returns:
    numpy.ndarray: The cleaned prediction array.
    """
    # Get the shape of the input prediction
    _, r, c, s, _ = prd.shape
    
    # Create a box mask with ones in the center and zeros at the boundaries
    box = np.ones((r, c, s), dtype=prd.dtype)
    box[:b, :, :] = 0
    box[:, :b, :] = 0
    box[(r-b):, :, :] = 0
    box[:, (c-b):, :] = 0
    
    # Expand the box mask to match the shape of prd
    box = np.expand_dims(box, axis=(0, -1))
    
    # Apply the box mask to the prediction
    cleaned_prd = prd * box
    return cleaned_prd

def zero_out_values(input_array, threshold):
    """
    Zeros out values in the input array that are less than the threshold.
    
    :param input_array: 5D numpy array of shape (R, C, S, T, Channels)
    :param threshold: float value to zero out values less than this
    :return: 5D numpy array of shape (R, C, S, T, Channels)
    """
    return np.where(input_array < threshold, 0, input_array)