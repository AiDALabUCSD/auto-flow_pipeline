import nibabel as nib
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow.experimental import numpy as tnp
# Optional: If you are on TF < 2.8 or without experimental features, you can install a 3D interpolation library

############################################################
# 1. Load NIfTI as a Tensor
############################################################
def load_nifti_tf(nifti_path: str) -> tf.Tensor:
    """
    Loads a NIfTI file and returns the data as a tf.Tensor in float32.
    Shape will be (R, C, S, T).
    """
    nifti_img = nib.load(nifti_path)
    np_data = nifti_img.get_fdata(dtype="float32")  # shape (R, C, S, T)
    return tf.convert_to_tensor(np_data, dtype=tf.float32)

############################################################
# 2. Downsample in 3D using TensorFlow
############################################################
# TensorFlow 2.x doesn't provide a direct built-in 3D resize in tf.image.
# We can either:
#   (A) Use tf.nn.conv3d or a custom approach.
#   (B) Use tf.image.resize on slices/time in a loop or map.
#   (C) Use tf.experimental.image.resize3d (introduced in nightly builds / TF Addons) if available.
# Here, we provide a demonstration with a custom approach.

# NOTE: If you only need to downsample the spatial dims (R, C, S) and keep time as is, we can do a rudimentary approach.
# We'll do a single pass with tf.image.resize for each time slice, though that involves a small Python loop.
# For large performance needs, consider using 3D libraries like MONAI, or a custom 3D interpolation.


def downsample_4d_volume_tf(data: tf.Tensor, target_dims=(192, 192, 64, 20)) -> tf.Tensor:
    """
    Downsamples a 4D volume (R, C, S, T) to target_dims using tf.image.resize.
    The last dimension (time T) is preserved or resized if needed.
    NOTE: This uses a simple approach that loops over time to do 3D resizing.
    """
    # data shape: (R, C, S, T)
    # target_dims = (target_R, target_C, target_S, target_T)
    # We'll separate the time dimension and resize each 3D volume individually.

    r_in, c_in, s_in, t_in = tf.unstack(tf.shape(data))
    r_out, c_out, s_out, t_out = target_dims

    # If we need to preserve the time dimension exactly, we can slice or resize.
    # We assume we want to either preserve or match t_out.
    # We'll handle each time slice.

    resized_volumes = []

    # Split across time dimension
    volumes = tf.split(data, num_or_size_splits=t_in, axis=3)  # list of (R,C,S,1)

    # Resize each 3D volume to (r_out, c_out, s_out)
    for i, vol in enumerate(volumes):
        vol = tf.squeeze(vol, axis=3)  # shape (R, C, S)
        # tf.image.resize is 2D. We can treat S as channels or do a trick.
        # We'll do a simple approach: treat the slice dimension as a batch dimension.
        # Transpose to (S, R, C), then resize each 2D slice, then stack back.
        # For more robust 3D interpolation, consider specialized libs.

        vol_t = tf.transpose(vol, perm=[2, 0, 1])  # (S, R, C)
        # We want to get s_out slices, each (r_out, c_out)
        # We'll do an extra dimension for channel: (S, R, C, 1)
        vol_t = tf.expand_dims(vol_t, axis=-1)

        # Resize along R and C first (2D resize). We keep S as batch.
        vol_t_resized = tf.map_fn(
            lambda x: tf.image.resize(x, (r_out, c_out), method='bilinear'),
            vol_t,
            fn_output_signature=tf.float32
        )  # shape (S, r_out, c_out, 1)

        # Now we must resize along the S dimension from S_in to s_out.
        # We'll transpose back to (r_out, c_out, S, 1) => treat r_out, c_out as batch dims.
        vol_t_resized = tf.transpose(vol_t_resized, perm=[1, 2, 0, 3])  # (r_out, c_out, S, 1)
        # We can now use tf.image.resize again in the 'S' dimension by treating (r_out) as batch, (c_out) as width,
        # which is still 2D. This is tricky.
        # Alternatively, we do a naive approach with a final pass using tf.nn.conv1d or a custom method.

        # For brevity, let's do a naive approach: flatten r_out and c_out into batch, then resize in 'S' dimension.
        vol_flat = tf.reshape(vol_t_resized, [r_out * c_out, s_in, 1])  # combine r_out, c_out -> batch
        vol_flat = tf.transpose(vol_flat, perm=[0, 2, 1])  # (batch, channels, width=s_in)
        # Now we do 1D interpolation from s_in -> s_out with linear
        # We can do that with a custom linear interpolation or tf.image.resize if we reshape to 2D.

        # We'll create an image shape: (batch, s_in, 1) => resize to (batch, s_out)
        vol_flat = tf.reshape(vol_flat, [r_out * c_out, s_in, 1])
        vol_flat_resized = tf.image.resize(vol_flat, (s_out,), method='bilinear')  # (batch, s_out, 1)

        # Now restore shape => (r_out, c_out, s_out, 1)
        vol_s_resized = tf.reshape(vol_flat_resized, [r_out, c_out, s_out, 1])

        resized_volumes.append(vol_s_resized)

    # Stack along time dimension (4)
    # If we want t_out different from t_in, we can do another pass.
    result_4d = tf.stack(resized_volumes, axis=3)  # shape (r_out, c_out, s_out, t_in)

    # If we want t_in != t_out, we can do a final 1D or 2D interpolation along time.
    if t_in != t_out:
        # We do a similar approach along the 3rd dimension for time.
        # For simplicity, let's do a naive approach with tf.image.resize in 1D again.
        result_4d = tf.transpose(result_4d, perm=[0, 1, 2, 3])  # (r_out, c_out, s_out, t_in)
        shape = tf.shape(result_4d)
        r_cur, c_cur, s_cur, t_cur = shape[0], shape[1], shape[2], shape[3]
        result_flat = tf.reshape(result_4d, [r_cur*c_cur*s_cur, t_cur, 1])
        result_t_resized = tf.image.resize(result_flat, (t_out,), method='bilinear')
        result_4d = tf.reshape(result_t_resized, [r_cur, c_cur, s_cur, t_out])

    return result_4d

############################################################
# 3. Reorder to (T, R, C, S, 1)
############################################################
def reorder_4d_volume_tf(data: tf.Tensor) -> tf.Tensor:
    """
    Reorders data from (R, C, S, T) to (T, R, C, S, 1).
    Vectorized with tf.transpose and tf.reshape.
    """
    # data shape: (R, C, S, T)
    data = tf.transpose(data, perm=[3, 0, 1, 2])  # (T, R, C, S)
    data = tf.expand_dims(data, axis=-1)         # (T, R, C, S, 1)
    return data

############################################################
# 4. Min-Max Normalize Each Time-Slice (Vectorized)
############################################################
def min_max_normalize_4d_tf(data: tf.Tensor) -> tf.Tensor:
    """
    Performs min-max normalization on each time-slice in a single pass.
    data shape: (T, R, C, S, 1).
    """
    # Compute min/max along axes (1,2,3,4), per time-slice T
    mins = tf.reduce_min(data, axis=[1,2,3,4], keepdims=True)
    maxs = tf.reduce_max(data, axis=[1,2,3,4], keepdims=True)
    # Avoid division by zero
    data = tf.where(tf.equal(maxs, mins), 0.0, (data - mins) / (maxs - mins))
    return data

############################################################
# 5. Percentile Rescale Each Time-Slice (Vectorized)
############################################################
def percentile_rescale_tf(data: tf.Tensor, p_lower=5.0, p_upper=95.0) -> tf.Tensor:
    """
    Rescales intensities to [p_lower, p_upper] percentile range per time-slice.
    data shape: (T, R, C, S, 1).
    Uses TensorFlow Probability for percentile.
    """
    import tensorflow_probability as tfp

    # Compute p5, p95 along axes=[1,2,3,4], keepdims=True, for each T
    p_low_vals = tfp.stats.percentile(data, p_lower, axis=[1,2,3,4], keepdims=True)
    p_high_vals = tfp.stats.percentile(data, p_upper, axis=[1,2,3,4], keepdims=True)

    # Rescale each T-slice
    # (data - p_low_vals)/(p_high_vals - p_low_vals), clipped to [0,1]
    rng = p_high_vals - p_low_vals
    data = tf.clip_by_value((data - p_low_vals) / rng, 0.0, 1.0)

    return data

############################################################
# 6. Full Preprocessing Pipeline
############################################################
def preprocess_nifti_for_inference_tf(nifti_path: str,
                                      target_dims=(192,192,64,20)) -> tf.Tensor:
    """
    1. Load a 4D NIfTI (R, C, S, T)
    2. Downsample to (192,192,64,T) or custom target_dims
    3. Reorder to (T, R, C, S, 1)
    4. Min-max normalize each time-slice (vectorized)
    5. Rescale intensities [5,95] percentile (vectorized)
    6. Returns a tf.Tensor ready for GPU inference.
    """
    # Step 1: Load as tf.Tensor
    data_4d = load_nifti_tf(nifti_path)  # shape (R, C, S, T)

    # Step 2: Downsample in 3D (CPU or GPU). If you want GPU, ensure
    #         this function runs in a tf.function context or in a GPU session.
    #         The custom code is fairly verbose, but demonstrates the concept.
    downsampled = downsample_4d_volume_tf(data_4d, target_dims)

    # Step 3: Reorder => (T, R, C, S, 1)
    reordered = reorder_4d_volume_tf(downsampled)

    # Step 4: Min-max normalization (per time-slice)
    normalized = min_max_normalize_4d_tf(reordered)

    # Step 5: Percentile rescale to [5,95]
    rescaled = percentile_rescale_tf(normalized, 5.0, 95.0)

    return rescaled

############################################################
# Usage Example:
# preprocessed = preprocess_nifti_for_inference_tf("example.nii.gz")
# model = ... # Some TF model
# predictions = model.predict(preprocessed)  # If shape matches model input
