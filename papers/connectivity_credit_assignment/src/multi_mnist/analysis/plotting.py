"""Figure styling and the plot types the paper uses.

Two plot types carry every figure: a learning curve with a confidence band
over seeds, and an asymptotic-metric-versus-x summary. Both propagate NaN, so
a diverged configuration leaves a visible gap instead of being averaged away.

Every figure is saved twice, with and without its legend (``_nl`` suffix). The
paper's figures label series inline, so the legend-free version is what goes
into the manuscript while the legended one stays readable on its own.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy.stats import t as t_dist


COLOR_PALETTE = [
    '#0086DA', '#CA3720', '#2BAF2A', '#9923DC',
    '#e67e22', '#34495e', '#1abc9c',
]

# Color-blind-safe variant (deuteranopia / protanopia / tritanopia). Same slot
# order as COLOR_PALETTE, so a fixed slot keeps its meaning across both.
COLOR_PALETTE_CB = [
    '#0077BB', '#CC3311', '#009988', '#EE3377',
    '#EE7733', '#33BBEE', '#BBBBBB',
]

# Pinned slots, so a method is the same color in every figure it appears in.
DEFAULT_COLOR_IDS = {
    'block-sparse': 0,       # blue
    'dense': 1,              # red
    'dense transition': 3,   # purple
    'set': 2,                # green
    'deep-r': 4,             # orange
    'random-sparse': 5,
}


def get_color_palette(classes=None, n=None, cb=False):
    """The default palette, ``n`` of its colors, or a class-to-color mapping.

    Classes listed in ``DEFAULT_COLOR_IDS`` keep their pinned slot; anything
    else is assigned alphabetically from what remains, so adding a method does
    not recolor the existing ones.
    """
    palette = COLOR_PALETTE_CB if cb else COLOR_PALETTE

    if classes is not None:
        if len(classes) > len(palette):
            raise ValueError(
                f'Only {len(palette)} colors available, {len(classes)} requested.')
        color_map = {c: DEFAULT_COLOR_IDS[c.lower()]
                     for c in classes if c.lower() in DEFAULT_COLOR_IDS}
        remaining = [i for i in range(len(palette)) if i not in color_map.values()]
        for c in sorted(classes, key=str.lower):
            if c not in color_map:
                color_map[c] = remaining.pop(0)
        return {c: palette[i] for c, i in color_map.items()}

    if n is not None:
        if n > len(palette):
            raise ValueError(f'Only {len(palette)} colors available, {n} requested.')
        return palette[:n]

    return palette


#: Text sizes are set by how many panels sit side by side on a page: a figure
#: that will be printed at one quarter width needs much larger type than one
#: printed at half width.
_STYLES = {
    'default': dict(figsize=(5, 3), label=13, title=14, pad=10, tick=11,
                    legend=11, legend_title=12, lw=1.5),
    '2-col': dict(figsize=(7, 4.25), label=20, title=22, pad=18, tick=16,
                  legend=15, legend_title=16, lw=2.3),
    '3-col': dict(figsize=(6, 4.3), label=21, title=23, pad=19, tick=17,
                  legend=16, legend_title=17, lw=2.9),
    '4-col': dict(figsize=(5, 4), label=23, title=25, pad=21, tick=18,
                  legend=17, legend_title=18, lw=2.7),
}


def set_style(style='default'):
    """Apply one of the named figure styles. See ``_STYLES``."""
    if style not in _STYLES:
        raise ValueError(f'Unknown style {style!r}. Known: {sorted(_STYLES)}')
    s = _STYLES[style]
    plt.rcParams.update({
        'axes.spines.top': False,
        'axes.spines.right': False,
        'figure.figsize': s['figsize'],
        'axes.labelsize': s['label'],
        'axes.titlesize': s['title'],
        'axes.titlepad': s['pad'],
        'xtick.labelsize': s['tick'],
        'ytick.labelsize': s['tick'],
        'legend.fontsize': s['legend'],
        'legend.title_fontsize': s['legend_title'],
        'lines.linewidth': s['lw'],
    })


def save_fig(name, fig_dir='../figures', formats=('pdf', 'png'), dpi=400):
    """Save the current figure with and without its legend.

    Handles both axes-level legends and figure-level ones; the latter is what
    multi-panel figures use, and missing it is why a ``_nl`` file can come out
    with a legend still in it.
    """
    fig = plt.gcf()
    fig_legends = list(fig.legends)
    ax_legend = plt.gca().get_legend()

    def _set_visible(visible):
        for lg in fig_legends:
            lg.set_visible(visible)
        if ax_legend is not None:
            ax_legend.set_visible(visible)

    for fmt in formats:
        out_dir = Path(fig_dir) / fmt
        out_dir.mkdir(parents=True, exist_ok=True)
        _set_visible(False)
        plt.savefig(out_dir / f'{name}_nl.{fmt}', bbox_inches='tight', dpi=dpi)
        _set_visible(True)
        plt.savefig(out_dir / f'{name}.{fmt}', bbox_inches='tight', dpi=dpi)


# ---------------------------------------------------------------------------
# Plot types
# ---------------------------------------------------------------------------

def plot_ci_curve(steps, arr, label=None, color=None, alpha=0.20, ax=None,
                  confidence=0.95):
    """Mean and t-confidence band from a ``(n_periods, n_seeds)`` array.

    NaN-propagating on purpose: one diverged seed nulls the mean from its
    divergence onward, so the curve visibly stops rather than continuing on
    whichever seeds happened to survive.
    """
    ax = ax or plt.gca()
    mean = arr.mean(axis=1)
    sem = arr.std(axis=1, ddof=1) / np.sqrt(arr.shape[1])
    half_width = t_dist.ppf(0.5 + confidence / 2, df=arr.shape[1] - 1) * sem
    ax.plot(steps, mean, label=label, color=color)
    ax.fill_between(steps, mean - half_width, mean + half_width,
                    alpha=alpha, color=color, linewidth=0)
    return ax


def plot_sensitivity(asym, ax=None, x_col='optimizer.learning_rate', y_col='_metric',
                     hue_col=None, palette=None, title='', xlabel='Step-size',
                     ylabel='Accuracy', ylim=None, legend_title=None, log_base=2):
    """Asymptotic metric against a swept parameter, one line per hue value.

    With ``log_base=2`` the x ticks are rendered as powers of two, which is how
    every step-size axis in the paper reads.
    """
    ax = ax or plt.gca()

    if hue_col is None:
        sub = asym.sort_values(x_col)
        ax.plot(sub[x_col], sub[y_col], '-o', color=COLOR_PALETTE[0])
    else:
        try:
            hue_vals = sorted(asym[hue_col].dropna().unique(), key=float)
        except (TypeError, ValueError):
            hue_vals = sorted(asym[hue_col].dropna().unique())
        colors = (list(palette) if palette is not None
                  else sns.color_palette('viridis', n_colors=len(hue_vals)))
        for hue_val, color in zip(hue_vals, colors):
            sub = asym[asym[hue_col] == hue_val].sort_values(x_col)
            try:
                label = f'{int(hue_val):,}' if float(hue_val).is_integer() else f'{hue_val}'
            except (TypeError, ValueError):
                label = str(hue_val)
            ax.plot(sub[x_col], sub[y_col], '-o', color=color, label=label)
        ax.legend(title=legend_title or hue_col, loc='best')

    if log_base:
        ax.set_xscale('log', base=log_base)
        ticks = sorted(asym[x_col].dropna().unique())
        ax.set_xticks(ticks)
        ax.set_xticklabels(
            [f'$2^{{{int(np.log2(v))}}}$' for v in ticks] if log_base == 2
            else [str(v) for v in ticks])

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.grid(True, alpha=0.4)
    return ax


def format_step_axis(ax, scale=1e5):
    """Render a step axis with a shared power-of-ten offset."""
    import matplotlib.ticker as mticker
    ax.xaxis.set_major_formatter(mticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(axis='x', style='sci', scilimits=(0, 0), useMathText=True)
    return ax


# ---------------------------------------------------------------------------
# Inline series labels
# ---------------------------------------------------------------------------

#: The paper carries no legends: every series names itself in the empty space
#: beside its own curve, in that curve's color. These are the two numbers that
#: style does not get to vary per figure.
INLINE_LABEL_SIZE = 18            # above the tick labels (16), below the axis labels (20)
INLINE_LABEL_REF = (5.06, 15)     # the clearance that read right, and the size it was set at

INLINE_LABEL_FONT = dict(fontfamily='DejaVu Sans', fontstyle='italic',
                         fontstretch='condensed')


def label_curves_inline(ax, specs, size=INLINE_LABEL_SIZE, gap_pt=None,
                        sub_scale=0.72):
    """Name each series inline, clear of its own curve, and drop the legend.

    ``specs`` is a sequence of ``(text, x, side, color, curve)``, where ``side``
    is ``'above'`` or ``'below'`` and ``curve`` is the ``(x, y)`` the label has
    to clear. A sixth element adds a smaller parenthetical after the name, set
    on the same baseline.

    Two things make the spacing awkward enough to be worth centralising. The
    clearance has to be to the extreme of the curve spanned by the *whole*
    label, because a rising curve passes under its own name and closes a gap
    set at the anchor alone. And it is not knowable until the text has been
    laid out, so this places, draws, measures, and shifts by the shortfall --
    one pass is exact, since moving a label vertically does not change its
    width. Every label ends up the same number of points clear.
    """
    ref_gap, ref_size = INLINE_LABEL_REF
    gap_pt = ref_gap * size / ref_size if gap_pt is None else gap_pt
    fig = ax.figure
    px_per_pt = fig.dpi / 72
    y0 = float(np.mean(ax.get_ylim()))

    placed = []
    for text, x, side, color, curve, *rest in specs:
        main = ax.text(x, y0, text, color=color, ha='left', va='baseline',
                       fontsize=size, **INLINE_LABEL_FONT)
        placed.append([main, None, x, side, color, curve, rest[0] if rest else None])

    # The parenthetical starts where the name ends, so the name has to be
    # measured before it can be positioned.
    fig.canvas.draw()
    inv = ax.transData.inverted()
    for p in placed:
        if p[6] is None:
            continue
        bb = p[0].get_window_extent()
        x_sub = inv.transform((bb.x1 + 0.25 * size * px_per_pt, bb.y0))[0]
        p[1] = ax.text(x_sub, y0, p[6], color=p[4], ha='left', va='baseline',
                       fontsize=size * sub_scale, **INLINE_LABEL_FONT)
        # The pair is positioned and measured as one word.
        p[0]._inline_sub = p[1]

    # A name wider than the space to its right would otherwise stick out past
    # the axes, and the figure would be saved wider to fit it -- which reads as
    # the data having a longer x range than it does. Pull it back inside first,
    # before any height is worked out, since moving it changes what it spans.
    fig.canvas.draw()
    right = ax.get_window_extent().x1
    for p in placed:
        bb = p[0].get_window_extent()
        if p[1] is not None:
            bb = bb.union([bb, p[1].get_window_extent()])
        if bb.x1 <= right:
            continue
        for artist in (p[0], p[1]):
            if artist is None:
                continue
            ax_x, ax_y = artist.get_position()
            dx, dy = ax.transData.transform((ax_x, ax_y))
            artist.set_x(inv.transform((dx - (bb.x1 - right), dy))[0])

    fig.canvas.draw()
    for main, sub, x, side, _, (cx, cy), _ in placed:
        bb = main.get_window_extent()
        if sub is not None:
            bb = bb.union([bb, sub.get_window_extent()])
        disp = ax.transData.transform(np.column_stack([np.asarray(cx), np.asarray(cy)]))
        under = disp[(disp[:, 0] >= bb.x0) & (disp[:, 0] <= bb.x1), 1]
        if not len(under):
            continue
        sign = 1 if side == 'above' else -1
        gap = sign * (bb.y0 - under.max() if side == 'above' else under.min() - bb.y1)
        shift = sign * gap_pt * px_per_pt - gap
        for artist in (main, sub):
            if artist is None:
                continue
            ax_x, ax_y = artist.get_position()
            dx, dy = ax.transData.transform((ax_x, ax_y))
            artist.set_y(inv.transform((dx, dy + shift))[1])

    legend = ax.get_legend()
    if legend is not None:
        legend.remove()
    for lg in list(fig.legends):
        lg.remove()
    return [p[0] for p in placed]


def measure_inline_labels(ax, specs, texts):
    """Clearance of each inline label from its curve, in points. A check."""
    ax.figure.canvas.draw()
    px_per_pt = ax.figure.dpi / 72
    out = []
    for (text, x, side, color, curve, *_), artist in zip(specs, texts):
        bb = artist.get_window_extent()
        sub = getattr(artist, '_inline_sub', None)
        if sub is not None:
            bb = bb.union([bb, sub.get_window_extent()])
        cx, cy = curve
        disp = ax.transData.transform(np.column_stack([np.asarray(cx), np.asarray(cy)]))
        under = disp[(disp[:, 0] >= bb.x0) & (disp[:, 0] <= bb.x1), 1]
        gap = (bb.y0 - under.max()) if side == 'above' else (under.min() - bb.y1)
        out.append((text, round(gap / px_per_pt, 3)))
    return out
