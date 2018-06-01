from glob import iglob
import matplotlib
import ternary
import pandas as pd
import sys
import numpy as np
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
plt.style.use('ggplot')


def plot_ternary(av_df, labels_ordered):

    scale = 1
    fig, ax = plt.subplots()
    figure, tax = ternary.figure(ax=ax, scale=scale)
    # figure.set_size_inches(10, 10)
    # Draw Boundary and Gridlines
    tax.boundary(linewidth=2.0)
    tax.gridlines(color="blue", multiple=0.1)

    # Set Axis labels and Title
    fontsize = 20
    # tax.set_title("Various Lines", fontsize=20)
    tax.left_axis_label(labels_ordered[1], fontsize=fontsize)
    tax.right_axis_label(labels_ordered[2], fontsize=fontsize)
    tax.bottom_axis_label(labels_ordered[0], fontsize=fontsize)

    # Draw lines parallel to the axes
    # tax.horizontal_line(16)
    # tax.left_parallel_line(10, linewidth=2., color='red', linestyle="--")
    # tax.right_parallel_line(20, linewidth=3., color='blue')
    # Draw an arbitrary line, ternary will project the points for you
    p1 = (0.2, 0.35, 0.45)
    p2 = (0, 0, 1)
    tax.line(p1, p2, linewidth=3., marker='s', color='green', linestyle=":")
    ticks = list(np.arange(0, 1.1, 0.1))
    tax.ticks(ticks=ticks, axis='brl', multiple=0.1, tick_formats="%0.1f")
    tax.clear_matplotlib_ticks()
    plt.tight_layout()
    plt.show()


files = iglob('run*/trio_df.tsv')
df = pd.concat([pd.read_table(fn, sep='\t') for fn in files]).groupby(
    ['EUR',  'ASN', 'AFR'], as_index=False).agg('mean')

import ternary

## Boundary and Gridlines
scale = 100
figure, tax = ternary.figure(scale=scale)

# Draw Boundary and Gridlines
tax.boundary(linewidth=2.0)
#tax.gridlines(color="black", multiple=10)
#tax.gridlines(color="blue", multiple=1, linewidth=0.5)

# Set Axis labels and Title
fontsize = 20
tax.set_title("Simplex Boundary and Gridlines", fontsize=fontsize)
tax.left_axis_label("Left label $\\alpha^2$", fontsize=fontsize)
tax.right_axis_label("Right label $\\beta^2$", fontsize=fontsize)
tax.bottom_axis_label("Bottom label $\\Gamma - \\Omega$", fontsize=fontsize)
tax.scatter(points, c=colors)
# Set ticks
tax.ticks(axis='lbr', linewidth=1, multiple=10)

# Remove default Matplotlib Axes
tax.clear_matplotlib_ticks()

ternary.plt.show()