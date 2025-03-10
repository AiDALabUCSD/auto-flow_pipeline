import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

def plot_flow_data(aortic_flow_df: pd.DataFrame, pulmonary_flow_df: pd.DataFrame, out_path: str):
    """
    Create a figure with the y-axis as flow rates (L/s) and x-axis as time frames.
    Each slice of the aorta is plotted in a different shade of red, each pulmonary slice in a different shade of blue.
    The legend shows each slice's volumetric flow (L/min). The figure is saved to disk.

    Parameters:
        aortic_flow_df (pd.DataFrame): DataFrame of aortic flow. Rows = slices, columns = time frames, plus 'volumetric_flow_rate'.
        pulmonary_flow_df (pd.DataFrame): DataFrame of pulmonary flow. Same format.
        out_path (str): File path where the figure will be saved.
    """
    plt.figure()

    # For convenience, time points are all columns except the last one (volumetric_flow_rate)
    time_aortic = aortic_flow_df.columns[:-1]
    time_pulm = pulmonary_flow_df.columns[:-1]

    # Convert them to numeric arrays (they are integer column names, but let's just ensure np.array)
    time_aortic = np.array(time_aortic, dtype=float)
    time_pulm = np.array(time_pulm, dtype=float)

    # Create color maps for slices
    n_aortic_slices = len(aortic_flow_df)
    n_pulm_slices = len(pulmonary_flow_df)

    aortic_colors = cm.Reds(np.linspace(0.4, 0.8, n_aortic_slices))
    pulm_colors   = cm.Blues(np.linspace(0.4, 0.8, n_pulm_slices))

    # Plot aortic data: connect points with a line
    for i in range(n_aortic_slices):
        row_data = aortic_flow_df.iloc[i, :-1].values  # all time columns except 'volumetric_flow_rate'
        flow_val = aortic_flow_df.iloc[i, -1]          # 'volumetric_flow_rate'
        plt.plot(
            time_aortic,
            row_data,
            color=aortic_colors[i],
            marker='o',
            label=f"Aorta slice {i}: {flow_val:.2f} L/min"
        )

    # Plot pulmonary data: connect points with a line
    for i in range(n_pulm_slices):
        row_data = pulmonary_flow_df.iloc[i, :-1].values
        flow_val = pulmonary_flow_df.iloc[i, -1]
        plt.plot(
            time_pulm,
            row_data,
            color=pulm_colors[i],
            marker='o',
            label=f"Pulm slice {i}: {flow_val:.2f} L/min"
        )

    plt.xlabel("Time frame")
    plt.ylabel("Flow (L/s)")
    plt.title("Aortic and Pulmonary Flow by Slice")
    plt.legend(loc="best")

    # Save and close
    plt.savefig(out_path)
    plt.close()
