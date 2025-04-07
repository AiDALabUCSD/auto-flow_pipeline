import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter
from matplotlib.ticker import FuncFormatter
from scipy.stats import pearsonr
import pandas as pd
import pingouin as pg

def plot_scatter_basic(
    df,
    x_col,
    y_col,
    ax,
    color='blue',
    label=None,
    point_size=5,
    alpha=1,
    edgecolors='black',
    linewidths=0.5
):
    """
    Plots a simple scatter plot on the provided axes, with control over the point size.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis values.
        y_col (str): Column name for y-axis values.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        color (str): Color for data points.
        label (str): Label for the legend.
        point_size (int or float): Size of the points.
        alpha (float): Transparency of the markers.
        edgecolors (str): Color of marker edges.
        linewidths (float): Width of marker edges.
    """
    ax.scatter(
        df[x_col],
        df[y_col],
        color=color,
        alpha=alpha,
        label=label,
        s=point_size,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    ax.grid(False)


def plot_scatter_with_error(
    df,
    x_col,
    y_col,
    std_col,
    ax,
    threshold,
    alpha=1,
    color='blue',
    high_std_color='red',
    high_std_marker='^',
    max_error_size=None,
    label_normal='Normal',
    label_high_std='High Std Dev',
    point_size=5,
    edgecolors='black',
    linewidths=0.5
):
    """
    Plots a scatter plot with error bars and highlights points exceeding a standard deviation threshold,
    with control over the point size, alpha, and optional marker edges.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis values.
        y_col (str): Column name for y-axis values.
        std_col (str): Column name for standard deviation values.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        threshold (float): Standard deviation threshold for highlighting.
        alpha (float): Transparency of markers.
        color (str): Color for normal data points.
        high_std_color (str): Color for data points exceeding the threshold.
        high_std_marker (str): Marker style for data points exceeding the threshold.
        max_error_size (float, optional): Maximum size for error bars.
        label_normal (str): Label for normal data points in the legend.
        label_high_std (str): Label for high standard deviation points in the legend.
        point_size (int or float): Size of the points.
        edgecolors (str): Edge color for markers.
        linewidths (float): Edge line width for markers.
    """
    errors = df[std_col].clip(upper=max_error_size) if max_error_size else df[std_col]
    normal_points = df[std_col] <= threshold
    high_std_points = df[std_col] > threshold

    # Plot error bars (no markers)
    ax.errorbar(
        df[x_col][normal_points],
        df[y_col][normal_points],
        yerr=errors[normal_points],
        fmt='none',
        ecolor='black',
        alpha=alpha,
        label=label_normal,
        elinewidth=1
    )
    # Plot normal points as scatter
    ax.scatter(
        df[x_col][normal_points],
        df[y_col][normal_points],
        color=color,
        alpha=alpha,
        label=label_normal,
        s=point_size,
        edgecolors=edgecolors,
        linewidths=linewidths
    )

    # Plot error bars (no markers) for high STD points
    ax.errorbar(
        df[x_col][high_std_points],
        df[y_col][high_std_points],
        yerr=errors[high_std_points],
        fmt='none',
        ecolor='black',
        alpha=alpha,
        label=label_high_std,
        elinewidth=1
    )
    # Plot high STD points as scatter
    ax.scatter(
        df[x_col][high_std_points],
        df[y_col][high_std_points],
        color=high_std_color,
        alpha=alpha,
        label=label_high_std,
        s=point_size,
        marker=high_std_marker,
        edgecolors=edgecolors,
        linewidths=linewidths
    )

    ax.grid(False)


def customize_plot(
    ax,
    xlabel=None,
    ylabel=None,
    title=None,
    xlim=None,
    ylim=None,
    show_legend=True,
    legend_fontsize=12,
    tick_label_fontsize=12
):
    """
    Customizes the aesthetics of a plot, including optional axis labels, title, and optional legend.
    Adds control over the font sizes of tick labels and formats ticks to show integer labels.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object to customize.
        xlabel (str, optional): Label for the x-axis.
        ylabel (str, optional): Label for the y-axis.
        title (str, optional): Title for the plot.
        xlim (tuple, optional): Limits for the x-axis (xmin, xmax).
        ylim (tuple, optional): Limits for the y-axis (ymin, ymax).
        show_legend (bool): Whether to display the legend.
        legend_fontsize (int): Font size for the legend.
        tick_label_fontsize (int): Font size for the tick labels.
    """
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=15)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=15)
    if title:
        ax.set_title(title, fontsize=18)

    ax.set_facecolor('white')
    
    if xlim:
        ax.set_xlim(xlim)
    if ylim:
        ax.set_ylim(ylim)

    # Set integer locator and formatter for the x-axis
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Set integer locator and formatter for the y-axis
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%d'))

    # Set tick label font sizes
    ax.tick_params(axis='x', labelsize=tick_label_fontsize)
    ax.tick_params(axis='y', labelsize=tick_label_fontsize)

    if show_legend and ax.get_legend_handles_labels()[1]:
        ax.legend(fontsize=legend_fontsize, frameon=False)
        
    for spine in ax.spines.values():
        spine.set_visible(True)  # Make the spine visible
        spine.set_color('black')  # Set the color of the spine to black
        spine.set_linewidth(1.5) 

    ax.grid(False)  # Disable grid


def add_identity_line(ax, line_style='k--', alpha=0.75):
    """
    Adds an identity line (y = x) to the plot with customizable style.
    """
    lims = [
        np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
        np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ]
    ax.plot(lims, lims, line_style, alpha=alpha, zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)


def add_horizontal_line(ax, y, line_style='k:', alpha=0.75):
    """
    Adds a horizontal dotted line at a specified y-level.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object where the line is added.
        y (float): The y-coordinate of the horizontal line.
        line_style (str): Style of the line (color and line type).
        alpha (float): Opacity of the line.
    """
    xlim = ax.get_xlim()
    ax.plot(xlim, [y, y], line_style, alpha=alpha)
    ax.set_xlim(xlim)


def add_vertical_line(ax, x, line_style='k:', alpha=0.75):
    """
    Adds a vertical dotted line at a specified x-level.

    Parameters:
        ax (matplotlib.axes.Axes): Axes object where the line is added.
        x (float): The x-coordinate of the vertical line.
        line_style (str): Style of the line (color and line type).
        alpha (float): Opacity of the line.
    """
    ylim = ax.get_ylim()
    ax.plot([x, x], ylim, line_style, alpha=alpha)
    ax.set_ylim(ylim)


def add_correlation_text_with_markers(ax, labels, r_values, p_values, colors, 
                                      position=(0.05, 0.05), fontsize=12, box_alpha=0.75):
    """
    Adds a text box with colored markers, Pearson correlation coefficients, and p-values to the plot.
    
    Parameters:
        ax (matplotlib.axes.Axes): Axes object where the text box will be displayed.
        labels (list of str): List of labels corresponding to each dataset/reader.
        r_values (list of float): List of pre-calculated Pearson correlation coefficients.
        p_values (list of float): List of pre-calculated p-values.
        colors (list of str): List of colors corresponding to each dataset/reader.
        position (tuple): (x, y) position of the text box in axes fraction coordinates (default: (0.05, 0.05)).
        fontsize (int): Font size of the text (default: 12).
        box_alpha (float): Transparency level of the text box background (default: 0.75).
    """
    # Prepare text lines with markers
    text_lines = []
    for label, r, p, color in zip(labels, r_values, p_values, colors):
        text_lines.append(f"$\\bullet$ {label}: r = {r:.2f}, p = {p:.2g}")
    
    # Combine all lines into a single string
    textstr = '\n'.join(text_lines)
    
    # Use LaTeX for bullet points and set colors
    # Create a dictionary for custom colors in LaTeX
    from matplotlib import rcParams
    rcParams['text.usetex'] = True
    rcParams['text.latex.preamble'] = [r'\\usepackage{xcolor}']
    
    # Modify text to include color commands
    colored_text_lines = []
    for line, color in zip(text_lines, colors):
        colored_line = r'\\textcolor{' + color + '}{' + line + '}'
        colored_text_lines.append(colored_line)
    colored_textstr = '\n'.join(colored_text_lines)
    
    # Add the text box to the plot
    ax.text(
        position[0],
        position[1],
        colored_textstr,
        transform=ax.transAxes,
        fontsize=fontsize,
        verticalalignment='bottom',
        horizontalalignment='left',
        bbox=dict(facecolor='white', alpha=box_alpha, boxstyle='round,pad=0.5')
    )


def plot_scatter_with_threshold(
    df,
    x_col,
    y_col,
    threshold_cols,
    threshold,
    ax,
    alpha=1,
    color='blue',
    high_value_color='red',
    high_value_marker='^',
    label_normal='Normal',
    label_high_value='High Value',
    point_size=5,
    edgecolors='black',
    linewidths=0.5
):
    """
    Plots a scatter plot and highlights points where any specified columns exceed a threshold.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis values.
        y_col (str): Column name for y-axis values.
        threshold_cols (list of str): List of column names to apply the threshold.
        threshold (float): Threshold value for highlighting.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        alpha (float): Transparency of markers.
        color (str): Color for normal data points.
        high_value_color (str): Color for data points exceeding the threshold.
        high_value_marker (str): Marker style for data points exceeding the threshold.
        label_normal (str): Label for normal data points in the legend.
        label_high_value (str): Label for high value points in the legend.
        point_size (int or float): Size of the points.
        edgecolors (str): Edge color for markers.
        linewidths (float): Edge line width for markers.
    """
    # Create a boolean mask for rows where any of the threshold_cols exceed the threshold
    high_value_points = df[threshold_cols].gt(threshold).any(axis=1)
    normal_points = ~high_value_points

    # Plot normal points
    ax.scatter(
        df.loc[normal_points, x_col],
        df.loc[normal_points, y_col],
        c=color,
        label=label_normal,
        alpha=alpha,
        s=point_size,
        edgecolors=edgecolors,
        linewidths=linewidths
    )

    # Plot high value points
    ax.scatter(
        df.loc[high_value_points, x_col],
        df.loc[high_value_points, y_col],
        c=high_value_color,
        marker=high_value_marker,
        label=label_high_value,
        alpha=alpha,
        s=point_size,
        edgecolors=edgecolors,
        linewidths=linewidths
    )

    ax.grid(False)


def create_two_by_three_plot(df_og, alpha=1, edgecolors='black', linewidths=0.5):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

    # --- Top row (Human: Ao, PA, Qp/Qs) ---
    # 1) Ao (Human)
    ax = axes[0, 0]
    channel = 'Ao'
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_PR',
        ax,
        color='green',
        label='Reader 1',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_LS',
        ax,
        color='purple',
        label='Reader 2',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-1, 12), ylim=(-1, 12),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 2) PA (Human)
    ax = axes[0, 1]
    channel = 'PA'
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_PR',
        ax,
        color='green',
        label='Reader 1',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_LS',
        ax,
        color='purple',
        label='Reader 2',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-1, 16), ylim=(-1, 16),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 3) Qp/Qs (Human)
    ax = axes[0, 2]
    channel = 'Qp/Qs'
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_PR',
        ax,
        color='green',
        label='Reader 1',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    plot_scatter_basic(
        df_og,
        channel + '_AH',
        channel + '_LS',
        ax,
        color='purple',
        label='Reader 2',
        point_size=100,
        alpha=alpha,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-0.5, 4), ylim=(-0.5, 4),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)
    add_horizontal_line(ax, y=1.5, line_style='k:', alpha=0.5)
    add_vertical_line(ax, x=1.5, line_style='k:', alpha=0.5)

    # --- Bottom row (Auto: Ao, PA, Qp/Qs) ---
    # 4) Ao (auto)
    ax = axes[1, 0]
    channel = 'Ao'
    plot_scatter_with_error(
        df=df_og,
        x_col=channel + '_AH',
        y_col=channel + '_auto',
        std_col=channel + '_auto_std',
        alpha=alpha,
        ax=ax,
        threshold=0.5,
        color='blue',
        high_std_color='red',
        high_std_marker='^',
        max_error_size=0.5,
        label_normal='AutoFlow',
        label_high_std='AutoFlow High Std Dev',
        point_size=100,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-1, 12), ylim=(-1, 12),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 5) PA (auto)
    ax = axes[1, 1]
    channel = 'PA'
    plot_scatter_with_error(
        df=df_og,
        x_col=channel + '_AH',
        y_col=channel + '_auto',
        std_col=channel + '_auto_std',
        ax=ax,
        alpha=alpha,
        threshold=0.5,
        color='blue',
        high_std_color='red',
        high_std_marker='^',
        max_error_size=0.5,
        label_normal='AutoFlow',
        label_high_std='AutoFlow High Std Dev',
        point_size=100,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-1, 16), ylim=(-1, 16),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 6) Qp/Qs (auto)
    ax = axes[1, 2]
    channel = 'Qp/Qs'
    plot_scatter_with_threshold(
        df=df_og,
        x_col=channel + '_AH',
        y_col=channel + '_auto',
        threshold_cols=['Ao_auto_std', 'PA_auto_std'],  # Adjust as needed
        threshold=0.5,  # Example threshold
        ax=ax,
        alpha=alpha,
        color='blue',
        high_value_color='red',
        high_value_marker='^',
        label_normal='AutoFlow',
        label_high_value='AutoFlow High Std Dev',
        point_size=100,
        edgecolors=edgecolors,
        linewidths=linewidths
    )
    customize_plot(ax, xlim=(-0.5, 4), ylim=(-0.5, 4),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)
    add_horizontal_line(ax, y=1.5, line_style='k:', alpha=0.5)
    add_vertical_line(ax, x=1.5, line_style='k:', alpha=0.5)

    plt.tight_layout()
    return fig, axes

def filter_dataframe(dataframe, column, threshold):
    return dataframe[dataframe[column] <= threshold]

def bland_altman_plot(data1, data2, ax=None, title='Bland-Altman Plot', ylim=None, *args, **kwargs):
    """
    Generate a Bland-Altman plot with absolute differences, including annotations for mean difference
    and standard deviation limits, with customizable y-axis limits.

    Parameters:
        data1 (array-like): Data from the first measurement method.
        data2 (array-like): Data from the second measurement method.
        ax (matplotlib.axes.Axes, optional): The axes upon which to plot the figure. If None, will create a new figure.
        title (str, optional): Title of the plot.
        ylim (tuple, optional): Tuple of (min, max) for y-axis limits.
        *args: Additional arguments for plt.scatter.
        **kwargs: Additional keyword arguments for plt.scatter, e.g., 'marker'.

    Returns:
        The axes with the Bland-Altman plot.
    """
    if ax is None:
        fig, ax = plt.subplots()

    means = np.mean([data1, data2], axis=0)
    diffs = data1 - data2
    mean_diff = np.mean(diffs)
    std_diff = np.std(diffs)

    # Limits of agreement
    loa1 = mean_diff - 1.96 * std_diff
    loa2 = mean_diff + 1.96 * std_diff

    # Plotting
    ax.scatter(means, diffs, color='blue', *args, **kwargs)
    ax.axhline(mean_diff, color='red', linestyle='--', label=f'Mean diff: {mean_diff:.2f}')
    ax.axhline(loa1, color='grey', linestyle='--', label=f'-1.96 SD: {loa1:.2f}')
    ax.axhline(loa2, color='grey', linestyle='--', label=f'+1.96 SD: {loa2:.2f}')

#     # Annotations
#     ax.annotate(f'Mean diff: {mean_diff:.2f}', xy=(0.05, mean_diff), xycoords=('axes fraction', 'data'),
#                 xytext=(0, 10), textcoords='offset points', ha='left', va='bottom', color='red')
#     ax.annotate(f'-1.96 SD: {loa1:.2f}', xy=(0.05, loa1), xycoords=('axes fraction', 'data'),
#                 xytext=(0, -10), textcoords='offset points', ha='left', va='top', color='grey')
#     ax.annotate(f'+1.96 SD: {loa2:.2f}', xy=(0.95, loa2), xycoords=('axes fraction', 'data'),
#                 xytext=(0, 10), textcoords='offset points', ha='right', va='bottom', color='grey')

    ax.set_title(title)
    ax.set_xlabel('Mean')
    ax.set_ylabel('Difference')
    ax.grid(False)
    ax.legend()

    if ylim:
        ax.set_ylim(ylim)

    return ax

def create_bland_altman_2x3(
    df_og,
    comp_suffix,
    bland_altman_plot_func,
    filter_dataframe_func,
    threshold=0.5
):
    """
    Creates a 2x3 figure with Bland–Altman plots for Ao, PA, Qp/Qs:
      - Top row = unfiltered data
      - Bottom row = filtered data (with your custom logic)
    
    Parameters
    ----------
    df_og : pd.DataFrame
        The original unfiltered dataframe.
    bland_altman_plot_func : function
        A function for creating Bland–Altman plots. Signature like:
          bland_altman_plot_func(data1, data2, ax=..., title=..., ylim=...).
    filter_dataframe_func : function
        Your filter function, e.g. filter_dataframe(df, column, threshold)
    threshold : float, default=0.5
        The threshold to apply for filtering.

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : np.ndarray of shape (2, 3)
    """
    
    if comp_suffix == "_auto":
        comp_name = "AutoFlow"
    elif comp_suffix == "_PR":
        comp_name = "Reader 1"
    elif comp_suffix == "_LS":
        comp_name = "Reader 2"
    else:
        # Fallback if you have some other suffix
        comp_name = f"Comparison {comp_suffix}"

    # -----------------------
    # 1) Filtering Logic
    # -----------------------
    # filtered Ao by Ao_auto_std <= threshold
    filtered_df_Ao = filter_dataframe_func(df_og, 'Ao_auto_std', threshold)
    # filtered PA by PA_auto_std <= threshold
    filtered_df_PA = filter_dataframe_func(df_og, 'PA_auto_std', threshold)
    # filtered Qp/Qs based on already Ao-filtered, then filtering by PA_auto_std <= threshold
    filtered_df_QpQs = filter_dataframe_func(filtered_df_Ao, 'PA_auto_std', threshold)

    # -----------------------
    # 2) Setup Figure & Axes
    # -----------------------
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 8))  # adjust size if needed

    # We'll handle each channel in a loop
    all_channels = ['Ao', 'PA', 'Qp/Qs']
    # This list matches each filtered df
    filtered_list = [filtered_df_Ao, filtered_df_PA, filtered_df_QpQs]

    for i, channel in enumerate(all_channels):
        # Decide y-limits (like your code)
        if channel in ['Ao', 'PA']:
            ylim = (-8, 8)
        else:
            ylim = (-2, 2)

        # Unfiltered data (top row)
        ax_top = axes[0, i]
        CNN = channel + comp_suffix
        GT = channel + '_AH'
        data1_unf = df_og[GT]
        data2_unf = df_og[CNN]
        bland_altman_plot_func(
            data1_unf,
            data2_unf,
            ax=ax_top,
            title=f"{comp_name} vs Ground Truth: {channel}",
            ylim=ylim
        )

        # Filtered data (bottom row)
        ax_bottom = axes[1, i]
        df_filt = filtered_list[i]
        data1_filt = df_filt[GT]
        data2_filt = df_filt[CNN]
        bland_altman_plot_func(
            data1_filt,
            data2_filt,
            ax=ax_bottom,
            title=f"Thresholded {comp_name} vs Ground Truth: {channel}",
            ylim=ylim
        )

    plt.tight_layout()
    return fig, axes

def analyze_performance_pearsonR(df, landmark, threshold):
    # Filter the DataFrame for standard deviation below the threshold
    std_col = f'{landmark}_auto_std'
    if landmark != 'Qp/Qs':
        filtered_df = df[df[std_col] < threshold]
    else:
        # For Qp/Qs, we need to filter based on both Ao_auto_std and PA_auto_std
        # filtered Ao by Ao_auto_std <= threshold
        filtered_df_Ao = filter_dataframe(df, 'Ao_auto_std', threshold)
        # filtered PA by PA_auto_std <= threshold
        filtered_df_PA = filter_dataframe(df, 'PA_auto_std', threshold)
        # filtered Qp/Qs based on already Ao-filtered, then filtering by PA_auto_std <= threshold
        filtered_df = filter_dataframe(filtered_df_Ao, 'PA_auto_std', threshold)
        
    print(len(filtered_df))

    # Ground truth column
    gt_col = landmark + '_AH'

    # CNN measurements
    cnn_col = f'{landmark}_auto'

    # Reader 1 measurements
    reader1_col = f'{landmark}_PR'

    # Reader 2 measurements
    reader2_col = f'{landmark}_LS'

    # Calculate Pearson correlation coefficient and p-value
    cnn_corr, cnn_p = pearsonr(filtered_df[gt_col], filtered_df[cnn_col])
    reader1_corr, reader1_p = pearsonr(filtered_df[gt_col], filtered_df[reader1_col])
    reader2_corr, reader2_p = pearsonr(filtered_df[gt_col], filtered_df[reader2_col])

    return {
        'CNN': {'Correlation': cnn_corr, 'P-value': cnn_p},
        'Reader 1': {'Correlation': reader1_corr, 'P-value': reader1_p},
        'Reader 2': {'Correlation': reader2_corr, 'P-value': reader2_p}
    }

def print_results_pearsonR(landmark,results):
    print(landmark)
    for key, values in results.items():
        print(f"{key} Results:")
        print(f"  Correlation Coefficient: {values['Correlation']:.3f}")
        print(f"  P-value: {values['P-value']:.3e}\n")

def reshape_for_icc_all_measurements(df_wide):
    """
    Reshape a wide DataFrame (df_wide) into a long form suitable
    for ICC analysis across ALL measurements (Ao, PA, Qp/Qs) together.

    The returned DataFrame has columns:
      - 'Phonetic'  (patient ID)
      - 'Measurement' (e.g. "Ao", "PA", or "Qp/Qs")
      - 'Rater'       (e.g. "AH", "PR", "LS", "auto")
      - 'Rating'      (numeric value)
      - 'UniqueID'    (a combo of Phonetic + Measurement)
    """

    # Ensure 'Phonetic' is a normal column (not the DataFrame index).
    if 'Phonetic' not in df_wide.columns:
        df_wide = df_wide.reset_index(drop=False)

    measurements = ["Ao", "PA"]#, "Qp/Qs"]
    raters = ["AH", "PR", "LS", "auto"]  # suffixes in wide columns

    long_frames = []

    for meas in measurements:
        for rater in raters:
            # e.g. "Ao_AH", "Ao_PR", "Ao_auto", etc.
            col_name = f"{meas}_{rater}"
            if col_name not in df_wide.columns:
                continue  # skip if not present

            # Create the partial long DataFrame
            tmp = pd.DataFrame({
                "Phonetic": df_wide["Phonetic"],
                "Measurement": meas,
                "Rater": rater,
                "Rating": df_wide[col_name]
            })
            long_frames.append(tmp)

    # Concatenate them
    df_long = pd.concat(long_frames, ignore_index=True)

    # Create a combined ID for "Phonetic + Measurement"
    # so each (patient, measurement) pair is treated as the "target" for ICC
    df_long["UniqueID"] = df_long["Phonetic"].astype(str) + "_" + df_long["Measurement"]

    return df_long


def filter_auto_std_wide(df_wide):
    """
    For each row (patient):
      - If Ao_auto_std >= 0.5, set Ao columns (and Ao_auto_std) to NaN.
      - If PA_auto_std >= 0.5, set PA columns (and PA_auto_std) to NaN.
      - If Ao_auto_std >= 0.5 OR PA_auto_std >= 0.5, set Qp/Qs columns (and Qp/Qs_auto_std) to NaN.
    Returns a copy of df_wide with only the 'qualified' columns retained for each measurement.
    """

    # Make a copy to modify
    df_filtered = df_wide.copy()

    # -------------------------------------------------------------
    # 1) Compute all masks from the ORIGINAL df_wide (not df_filtered!)
    # -------------------------------------------------------------
    # Ao fails if Ao_auto_std >= 0.5
    mask_ao_fail = (df_wide["Ao_auto_std"] >= 0.5)

    # PA fails if PA_auto_std >= 0.5
    mask_pa_fail = (df_wide["PA_auto_std"] >= 0.5)

    # Qp/Qs depends on BOTH Ao and PA (fail if either fails)
    mask_qpqs_fail = mask_ao_fail | mask_pa_fail

    # -------------------------------------------------------------
    # 2) Apply the masks in df_filtered
    # -------------------------------------------------------------
    # Ao fail => set Ao columns to NaN
    df_filtered.loc[mask_ao_fail, ["Ao_AH", "Ao_PR", "Ao_LS", "Ao_auto", "Ao_auto_std"]] = float('nan')

    # PA fail => set PA columns to NaN
    df_filtered.loc[mask_pa_fail, ["PA_AH", "PA_PR", "PA_LS", "PA_auto", "PA_auto_std"]] = float('nan')

    # Qp/Qs fail => set Qp/Qs columns to NaN
    df_filtered.loc[mask_qpqs_fail, ["Qp/Qs_AH", "Qp/Qs_PR", "Qp/Qs_LS", "Qp/Qs_auto", "Qp/Qs_auto_std"]] = float('nan')

    return df_filtered


def perform_pairwise_icc_all(df_long, rater_pairs):
    """
    Perform pairwise ICC across ALL measurements (Ao, PA, Qp/Qs) collectively,
    using 'UniqueID' = (Phonetic + Measurement) as the target.

    df_long is the output of reshape_for_icc_all_measurements.
    rater_pairs is something like: [("AH","PR"), ("AH","LS"), ("AH","auto")]

    Returns a dict { (r1,r2): icc_df } with pingouin result DataFrames.
    """
    results_dict = {}

    for (r1, r2) in rater_pairs:
        # Subset to only these two raters
        df_sub = df_long[df_long["Rater"].isin([r1, r2])].copy()

        # If everything is NaN or no rows, skip
        if df_sub["Rating"].dropna().empty:
            print(f"No valid rows for {r1} vs {r2}. Skipping.")
            continue

        # Pingouin ICC: "targets" is UniqueID (patient+measurement)
        icc = pg.intraclass_corr(
            data=df_sub,
            targets="UniqueID",
            raters="Rater",
            ratings="Rating"
        ).set_index("Type")

        print(f"ICC across Ao, PA, Qp/Qs combined: {r1} vs {r2}")
        display(icc)

        results_dict[(r1, r2)] = icc

    return results_dict

def get_low_cert_complement(df_original, df_filtered):
    """
    Creates a 'low-cert' complement of the df_filtered output from filter_auto_std_wide.
    For each measurement column, if df_filtered has NaN, that means it was flagged
    as low-cert. We keep the original data in that location.
    Otherwise, we set it to NaN.
    
    Parameters
    ----------
    df_original : pd.DataFrame
        The original unfiltered DataFrame (wide form).
    df_filtered : pd.DataFrame
        The output from filter_auto_std_wide(df_original).
    
    Returns
    -------
    df_low : pd.DataFrame
        The complement dataset: columns that are NaN in df_filtered are kept
        from df_original (low-cert), while columns that survived in df_filtered
        become NaN here.
    """
    df_low = df_original.copy()
    # For every column in df_filtered, if it's NOT NaN, it survived => set df_low to NaN
    # if it's NaN in df_filtered, that means it failed => keep original in df_low
    
    for col in df_filtered.columns:
        # Where df_filtered[col] is non-null => that measurement was high-cert
        # so in df_low we set it to NaN
        mask_survived = df_filtered[col].notna()
        df_low.loc[mask_survived, col] = float('nan')
    
    return df_low

