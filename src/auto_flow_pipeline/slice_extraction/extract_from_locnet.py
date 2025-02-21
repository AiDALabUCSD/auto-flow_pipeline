import numpy as np

def _get_MAX(prd):
    return np.array(np.unravel_index(np.argmax(prd), prd.shape))

def extract_straight_line_splines(heatmap):
    # Extract points using _get_MAX for each relevant channel
    av_point = _get_MAX(heatmap[..., 0])  # AV
    mid_aao_point = _get_MAX(heatmap[..., 2])  # Mid AAo
    pv_point = _get_MAX(heatmap[..., 4])  # PV
    mpa_point = _get_MAX(heatmap[..., 6])  # MPA

    # Create straight-line splines using the extracted points
    aorta_points = np.array([av_point, mid_aao_point])
    pa_points = np.array([pv_point, mpa_point])

    # Interpolate straight lines
    def interpolate_line(start, end, num_points=5):
        return np.linspace(start, end, num_points)

    aorta_spline = interpolate_line(aorta_points[0], aorta_points[1])
    pa_spline = interpolate_line(pa_points[0], pa_points[1])

    return aorta_spline, pa_spline