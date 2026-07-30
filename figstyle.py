"""Figure style for the paper. One import, one call, and every figure matches.

Three decisions are locked here and should not be made again per figure.

Typography. The manuscript sets Latin Modern. Matplotlib ships cmr10, Latin
Modern's ancestor, so figure text and body text are the same letterforms; the
math is Computer Modern for the same reason. Figures are built at exactly
``\\textwidth`` = 6.5in so ``\\includegraphics[width=\\textwidth]`` rescales
nothing and the point sizes below are the sizes that reach the page.

Colour by job, not by taste. The paper has exactly one nominal contrast, raw
against residual, and it owns the two hues. Model size is ordinal, so it takes a
single-hue ramp light to dark: the reader sees the ordering in the ink, and
lightness survives grayscale printing and every kind of colour blindness. Both
sets were checked with a CVD validator rather than by eye; the nominal pair
clears ΔE 12.4 under protanopia and 26.1 under normal vision with contrast above
3:1, and every ramp step clears 3:1 on white.

Identity is never carried by colour alone. ``label_ends`` writes each series name
at the end of its own line in the line's own colour, which removes the legend
box, removes the dependence on hue discrimination, and reads better.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ----------------------------------------------------------------- the palette
RAW = "#D6536A"     # rose, the raw measurement
RES = "#2E7EBB"     # blue, the residual after the projection
RAW_FILL = "#F3CAD2"
RES_FILL = "#C3DAEE"

SIZE = ["#5E9AC9", "#2A72AC", "#10416B"]   # ordinal, light to dark: 160M, 410M, 1.4B
OTHER = "#C9762B"                          # one extra family, off the red-green axis

INK = "#2B2B2B"
MUTED = "#6E6E6E"
GRID = "#E4E4E4"
RULE = "#BFBFBF"

TEXTWIDTH = 6.5      # tmlr.sty: \textwidth 6.5 true in
COLWIDTH = 3.15

_RC = {
    "font.family": "serif",
    "font.serif": ["cmr10", "CMU Serif", "Latin Modern Roman", "STIX Two Text"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,          # cmr10 has no unicode minus glyph

    "font.size": 8.5,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,

    "text.color": INK,
    "axes.labelcolor": INK,
    "axes.titlecolor": INK,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": MUTED,
    "ytick.labelcolor": MUTED,

    "axes.edgecolor": RULE,
    "axes.linewidth": 0.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "xtick.minor.size": 1.4,
    "ytick.minor.size": 1.4,
    "xtick.direction": "out",
    "ytick.direction": "out",

    "axes.grid": False,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "axes.axisbelow": True,

    "lines.linewidth": 1.2,
    "lines.markersize": 3.0,
    "lines.markeredgewidth": 0.0,
    "lines.solid_capstyle": "round",

    "legend.frameon": False,
    "legend.handlelength": 1.4,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.0,
    "legend.labelspacing": 0.3,

    "axes.titlelocation": "left",
    "axes.titlepad": 4.0,
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
    "savefig.transparent": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def use():
    """Install the style. Call once before building any figure."""
    mpl.rcParams.update(_RC)


def panel(ax, title=None, ygrid=False):
    """Finish one axes: a left-aligned muted title and an optional hairline y-grid."""
    if title:
        ax.set_title(title, color=MUTED, fontsize=8.5)
    if ygrid:
        ax.grid(axis="y", color=GRID, linewidth=0.5)
    ax.tick_params(length=2.5, pad=2)
    return ax


def label_ends(ax, series, xpad=1.03, fontsize=7.5, va_nudge=None):
    """Write each series name at the end of its own line, in its own colour.

    ``series`` is a list of ``(text, x, y, colour)``. This is what makes the
    figures legible in grayscale and to a colour-blind reader: the name sits on
    the line, so hue is decoration rather than the encoding.
    """
    for i, (text, x, y, colour) in enumerate(series):
        dy = 0.0 if va_nudge is None else va_nudge[i]
        ax.annotate(text, xy=(x, y), xycoords="data",
                    xytext=(3.0, dy), textcoords="offset points",
                    color=colour, fontsize=fontsize, va="center", ha="left",
                    annotation_clip=False)


def label_at(ax, x, y, text, colour, dx=0.0, dy=0.0, fontsize=7.0, ha="left"):
    """Name a curve at a point where it is separated from its neighbours.

    Use this instead of ``label_ends`` when several curves converge at the right
    edge, which is where end labels would collide.
    """
    ax.annotate(text, xy=(x, y), xycoords="data", xytext=(dx, dy),
                textcoords="offset points", color=colour, fontsize=fontsize,
                va="center", ha=ha, annotation_clip=False)


def label_curves(ax, curves, xrange=None, fontsize=6.8, pad=7.5):
    """Name each curve at the x where it has the most room, on the roomier side.

    ``curves`` is a list of ``(text, x, y, colour)``; the x grids need not agree,
    since the curves are resampled onto a common one to compare them. Putting
    every label at one x is what makes them collide when the curves run close
    together, so this picks each label its own x and its own side.
    """
    xs = np.linspace(max(np.min(c[1]) for c in curves),
                     min(np.max(c[1]) for c in curves), 400)
    ys = [np.interp(xs, np.asarray(c[1], float), np.asarray(c[2], float))
          for c in curves]
    lo, hi = ax.get_ylim()
    ok = (np.ones(len(xs), bool) if xrange is None
          else (xs >= xrange[0]) & (xs <= xrange[1]))
    mid = 0.5 * (xs[ok].min() + xs[ok].max())

    for k, (text, _, _, colour) in enumerate(curves):
        yk = ys[k]
        others = [ys[j] for j in range(len(ys)) if j != k]
        above = (np.minimum.reduce([np.where(o > yk, o, hi) for o in others])
                 if others else np.full(len(xs), hi))
        below = (np.maximum.reduce([np.where(o < yk, o, lo) for o in others])
                 if others else np.full(len(xs), lo))
        up, dn = np.minimum(above, hi) - yk, yk - np.maximum(below, lo)
        room = np.maximum(up, dn)
        room[~ok] = -np.inf
        # Among the positions that are near-best, take the most central one:
        # ties are common on a plateau, and the edges are where labels overflow.
        near = np.flatnonzero(room >= 0.97 * room.max())
        i = int(near[np.argmin(np.abs(xs[near] - mid))])
        top = up[i] >= dn[i]
        ax.annotate(text, xy=(xs[i], yk[i]), xycoords="data",
                    xytext=(0, pad if top else -pad), textcoords="offset points",
                    color=colour, fontsize=fontsize, ha="center",
                    va="bottom" if top else "top", annotation_clip=False)


def hero(ax, text, sub=None, loc=(0.97, 0.06)):
    """A number that is itself the finding, set large and quiet in the corner."""
    ax.text(loc[0], loc[1], text, transform=ax.transAxes, ha="right", va="bottom",
            fontsize=11, color=INK)
    if sub:
        ax.text(loc[0], loc[1], sub, transform=ax.transAxes, ha="right", va="top",
                fontsize=7, color=MUTED)


def band(ax, x, lo, hi, colour, alpha=0.9):
    """A spread band drawn as a light companion of the line colour, no edge."""
    ax.fill_between(x, lo, hi, color=colour, alpha=alpha, linewidth=0, zorder=1)


def shade(ax, x0, x1, label=None, y=0.97):
    """Mark an interval on the x axis without drawing a box around it."""
    ax.axvspan(x0, x1, color="#F2F1EE", zorder=0, linewidth=0)
    if label:
        ax.text(np.sqrt(x0 * x1), y, label, transform=ax.get_xaxis_transform(),
                ha="center", va="top", fontsize=6.8, color=MUTED)


def save(fig, path, w=TEXTWIDTH, h=None):
    """Set the exact printed width, then write the PDF."""
    if h is not None:
        fig.set_size_inches(w, h)
    fig.savefig(path)
    plt.close(fig)
