import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MaxNLocator, FormatStrFormatter

def plot_scatter_basic(df, x_col, y_col, ax, color='blue', label=None, point_size=5):
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
    """
    ax.scatter(df[x_col], df[y_col], color=color, label=label, s=point_size)
    ax.grid(False)


def plot_scatter_with_error(df, x_col, y_col, std_col, ax, threshold, color='blue', high_std_color='red', high_std_marker='^', max_error_size=None, label_normal='Normal', label_high_std='High Std Dev', point_size=5):
    """
    Plots a scatter plot with error bars and highlights points exceeding a standard deviation threshold, with control over the point size.

    Parameters:
        df (pd.DataFrame): DataFrame containing the data.
        x_col (str): Column name for x-axis values.
        y_col (str): Column name for y-axis values.
        std_col (str): Column name for standard deviation values.
        ax (matplotlib.axes.Axes): Axes object to plot on.
        threshold (float): Standard deviation threshold for highlighting.
        color (str): Color for normal data points.
        high_std_color (str): Color for data points exceeding the threshold.
        high_std_marker (str): Marker style for data points exceeding the threshold.
        max_error_size (float, optional): Maximum size for error bars.
        label_normal (str): Label for normal data points in the legend.
        label_high_std (str): Label for high standard deviation points in the legend.
        point_size (int or float): Size of the points.
    """
    errors = df[std_col].clip(upper=max_error_size) if max_error_size else df[std_col]
    normal_points = df[std_col] <= threshold
    high_std_points = df[std_col] > threshold

    ax.errorbar(df[x_col][normal_points], df[y_col][normal_points], yerr=errors[normal_points], fmt='o', color=color, ecolor='black', label=label_normal, elinewidth=1, markersize=np.sqrt(point_size))
    ax.errorbar(df[x_col][high_std_points], df[y_col][high_std_points], yerr=errors[high_std_points], fmt=high_std_marker, color=high_std_color, ecolor='black', label=label_high_std, elinewidth=1, markersize=np.sqrt(point_size))

    ax.grid(False)


def customize_plot(ax, xlabel=None, ylabel=None, title=None, xlim=None, ylim=None, show_legend=True, legend_fontsize=12, tick_label_fontsize=12):
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
    rcParams['text.latex.preamble'] = [r'\usepackage{xcolor}']
    
    # Modify text to include color commands
    colored_text_lines = []
    for line, color in zip(text_lines, colors):
        colored_line = r'\textcolor{' + color + '}{' + line + '}'
        colored_text_lines.append(colored_line)
    colored_textstr = '\n'.join(colored_text_lines)
    
    # Add the text box to the plot
    ax.text(position[0], position[1], colored_textstr, transform=ax.transAxes, fontsize=fontsize,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=box_alpha, boxstyle='round,pad=0.5'))


def create_two_by_three_plot(df_og):
    fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 12))

    # --- Top row (Human: Ao, PA, Qp/Qs) ---
    # 1) Ao
    ax = axes[0, 0]
    channel = 'Ao'
    plot_scatter_basic(df_og, channel + '_AH', channel + '_PR', ax,
                       color='green', label='Reader 1', point_size=100)
    plot_scatter_basic(df_og, channel + '_AH', channel + '_LS', ax,
                       color='purple', label='Reader 2', point_size=100)
    customize_plot(ax, xlim=(-1, 12), ylim=(-1, 12),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 2) PA
    ax = axes[0, 1]
    channel = 'PA'
    plot_scatter_basic(df_og, channel + '_AH', channel + '_PR', ax,
                       color='green', label='Reader 1', point_size=100)
    plot_scatter_basic(df_og, channel + '_AH', channel + '_LS', ax,
                       color='purple', label='Reader 2', point_size=100)
    customize_plot(ax, xlim=(-1, 16), ylim=(-1, 16),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 3) Qp/Qs
    ax = axes[0, 2]
    channel = 'Qp/Qs'
    plot_scatter_basic(df_og, channel + '_AH', channel + '_PR', ax,
                       color='green', label='Reader 1', point_size=100)
    plot_scatter_basic(df_og, channel + '_AH', channel + '_LS', ax,
                       color='purple', label='Reader 2', point_size=100)
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
        ax=ax,
        threshold=0.5,
        color='blue',
        high_std_color='red',
        high_std_marker='^',
        max_error_size=0.5,
        label_normal='AutoFlow',
        label_high_std='AutoFlow High Std Dev',
        point_size=100
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
        threshold=0.5,
        color='blue',
        high_std_color='red',
        high_std_marker='^',
        max_error_size=0.5,
        label_normal='AutoFlow',
        label_high_std='AutoFlow High Std Dev',
        point_size=100
    )
    customize_plot(ax, xlim=(-1, 16), ylim=(-1, 16),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)

    # 6) Qp/Qs (auto)
    ax = axes[1, 2]
    channel = 'Qp/Qs'
    plot_scatter_with_error(
        df=df_og,
        x_col=channel + '_AH',
        y_col=channel + '_auto',
        std_col=channel + '_auto_std',
        ax=ax,
        threshold=0.15,
        color='blue',
        high_std_color='red',
        high_std_marker='^',
        max_error_size=0.15,
        label_normal='AutoFlow',
        label_high_std='AutoFlow High Std Dev',
        point_size=100
    )
    customize_plot(ax, xlim=(-0.5, 4), ylim=(-0.5, 4),
                   show_legend=False, legend_fontsize=20, tick_label_fontsize=20)
    add_identity_line(ax, line_style='k--', alpha=0.75)
    add_horizontal_line(ax, y=1.5, line_style='k:', alpha=0.5)
    add_vertical_line(ax, x=1.5, line_style='k:', alpha=0.5)

    plt.tight_layout()
    return fig, axes