"""Resize the inline labels of a figure that can no longer be regenerated.

Figures 1 and 2 were labelled by hand in a vector editor, and the Comet project
their curves came from (``paper-weight-pruning-scaling-best``) no longer
exists, so they cannot be rebuilt through ``label_curves_inline`` the way
figures 3 and 5 are. This edits the PDFs instead: it scales each label to
``INLINE_LABEL_SIZE`` and re-seats it the standard clearance from its own
curve, leaving every other object untouched.

The labels are ordinary text objects, so the scale is exact -- glyphs and
advances both -- and the result stays vector. The clearance is measured off a
high-resolution render of the original: the curve is the only ink of its color
outside the label's own box, which is what makes the measurement possible
without the data behind it.

    python analysis/resize_inline_labels.py ../figures/paper_edited/figure_1_1.pdf
"""

import argparse
import re
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from multi_mnist.analysis.plotting import INLINE_LABEL_REF, INLINE_LABEL_SIZE  # noqa: E402

SVG_NS = 'http://www.w3.org/2000/svg'
ET.register_namespace('', SVG_NS)
INLINE_FACE = 'DejaVuSansCondensed-Oblique'
RENDER_DPI = 600


def _inkscape(*args):
    subprocess.run(['inkscape', *args], check=True, capture_output=True)


def _render(pdf, out_root):
    subprocess.run(['pdftoppm', '-r', str(RENDER_DPI), '-png', str(pdf), str(out_root)],
                   check=True, capture_output=True)
    return np.asarray(Image.open(f'{out_root}-1.png').convert('RGB'), dtype=int)


def _word_boxes(pdf):
    """Every word's box, from the PDF's own text, in points from the top left."""
    out = subprocess.run(['pdftotext', '-bbox', str(pdf), '-'],
                         check=True, capture_output=True, text=True).stdout
    return [(m.group(5), *(float(m.group(i)) for i in range(1, 5)))
            for m in re.finditer(
                r'<word xMin="([\d.]+)" yMin="([\d.]+)" xMax="([\d.]+)" '
                r'yMax="([\d.]+)">([^<]*)</word>', out)]


def _labels(svg_root):
    """The inline labels: the only text set in the condensed oblique face."""
    for text in svg_root.iter(f'{{{SVG_NS}}}text'):
        style = text.get('style', '')
        if INLINE_FACE not in style:
            continue
        size = float(re.search(r'font-size:([\d.]+)px', style).group(1))
        color = re.search(r'fill:(#[0-9a-fA-F]{6})', style).group(1)
        a, _, _, _, e, f = [float(v) for v in re.search(
            r'matrix\(([^)]*)\)', text.get('transform')).group(1).split(',')]
        tspan = text.find(f'{{{SVG_NS}}}tspan')
        yield text, tspan, dict(size=size, color=color, a=a, e=e, f=f,
                                content=''.join(tspan.itertext()))


def _label_box(words, lab):
    """The label's box, unioned over its words -- a name can be several."""
    n = len(lab['content'].split())
    line = sorted((w for w in words if abs(w[2] - lab['f']) < 20
                   and w[1] >= lab['e'] - 0.5), key=lambda w: w[1])[:n]
    if len(line) != n:
        raise RuntimeError(f"could not locate {lab['content']!r} in the PDF text")
    return (min(w[1] for w in line), min(w[2] for w in line),
            max(w[3] for w in line), max(w[4] for w in line))


def _curve_extreme(img, color, box, x_range, exclude, side):
    """Where the label's own curve runs, in points, under the label's width.

    The label shares its curve's color, so its own box is cut out first.
    """
    rgb = np.array([int(color[i:i + 2], 16) for i in (1, 3, 5)])
    scale = RENDER_DPI / 72
    mask = (np.abs(img - rgb).sum(axis=2) < 60)
    ys, xs = np.nonzero(mask)
    keep = (xs >= x_range[0] * scale) & (xs <= x_range[1] * scale)
    ex0, ey0, ex1, ey1 = exclude
    keep &= ~((xs >= (ex0 - 2) * scale) & (xs <= (ex1 + 2) * scale)
              & (ys >= (ey0 - 2) * scale) & (ys <= (ey1 + 2) * scale))
    if not keep.any():
        raise RuntimeError(f'no {color} ink under the label at {x_range}')
    ys = ys[keep] / scale
    return ys.min() if side == 'above' else ys.max()


def resize(pdf, size=INLINE_LABEL_SIZE, side='above'):
    ref_gap, ref_size = INLINE_LABEL_REF
    gap_pt = ref_gap * size / ref_size
    pdf = Path(pdf)
    words = _word_boxes(pdf)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        svg = tmp / 'fig.svg'
        _inkscape(str(pdf), f'--export-plain-svg={svg}')
        img = _render(pdf, tmp / 'orig')

        tree = ET.parse(svg)
        root = tree.getroot()
        report = []
        for text, tspan, lab in _labels(root):
            k = size / lab['size']
            xs = [float(v) for v in tspan.get('x').split()]
            # The box the label occupies now, and the box it will occupy once
            # scaled: a pure scale about the baseline origin.
            box = _label_box(words, lab)
            x0, y0, x1, y1 = box
            new = (lab['e'], lab['f'] + (y0 - lab['f']) * k,
                   lab['e'] + (x1 - lab['e']) * k, lab['f'] + (y1 - lab['f']) * k)

            curve_y = _curve_extreme(img, lab['color'], box, (new[0], new[2]), box, side)
            shift = ((curve_y - gap_pt) - new[3] if side == 'above'
                     else (curve_y + gap_pt) - new[1])

            text.set('style', re.sub(r'font-size:[\d.]+px', f'font-size:{size}px',
                                     text.get('style')))
            tspan.set('x', ' '.join(f'{v * k:.6f}' for v in xs))
            text.set('transform',
                     f"matrix({lab['a']},0,0,1,{lab['e']},{lab['f'] + shift})")
            report.append((lab['content'], lab['size'], size, round(shift, 2)))

        tree.write(svg)
        out = tmp / 'out.pdf'
        _inkscape(str(svg), f'--export-pdf={out}')
        pdf.write_bytes(out.read_bytes())

    for content, old_size, new_size, shift in report:
        print(f'{pdf.name}: {content!r} {old_size:.1f} -> {new_size} pt, '
              f'moved {shift:+.2f} pt')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('pdf', nargs='+')
    ap.add_argument('--size', type=float, default=INLINE_LABEL_SIZE)
    ap.add_argument('--side', default='above', choices=('above', 'below'),
                    help='which side of its curve every label in these files sits')
    args = ap.parse_args()
    for path in args.pdf:
        resize(path, size=args.size, side=args.side)
