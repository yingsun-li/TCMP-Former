# Define shared fonts, colors, and figure sizes.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BLUE = '#165e7c'
ORANGE = '#D97929'
GRAY = '#64717a'
METRICS = [BLUE, '#b65e35', '#397b70']
CMAP = LinearSegmentedColormap.from_list(
    'tcmp_blue', ['#ffffff', '#dceaf1', '#77aabc', BLUE])
SIZES = {
    'Fig2_binary_performance': (7.0, 2.1393744532),
    'Fig3_detection_timing': (7.0, 5.8333333333),
    'Fig4_typing_performance': (6.5, 5.1025),
    'Fig5_typing_confusion': (6.44859798775, 5.82612860892),
    'FigS2_matched_interictal': (3.5, 2.65),
}


def apply_style():
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 8, 'axes.labelsize': 8, 'axes.titlesize': 9,
        'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5, 'legend.fontsize': 7,
        'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False,
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none',
        'savefig.facecolor': 'white', 'figure.facecolor': 'white',
        'axes.unicode_minus': False,
    })


def panel_headings(fig, axes, titles):
    




    fw, fh = fig.get_size_inches()
    for letter, ax, title in zip('abcdefghijklmnopqrstuvwxyz', axes, titles):
        p = ax.get_position()
        y = p.y1 + .105 / fh
        fig.text(p.x0 - .23 / fw, y, letter, fontsize=11,
                 weight='bold', ha='left', va='baseline')
        fig.text(p.x0, y, title, fontsize=9, weight='normal',
                 ha='left', va='baseline')


def light_grid(ax, axis):
    ax.grid(axis=axis, color='#e5e8ea', linewidth=.45)
    ax.set_axisbelow(True)
