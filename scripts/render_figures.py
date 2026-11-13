# Validate figure inputs, check layouts, and export figures.
import argparse
import hashlib
import json
import os
from pathlib import Path
import warnings
import tempfile
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'tcmp_public_mpl_cache'))
from PIL import Image
from style import apply_style
import matplotlib.pyplot as plt
from matplotlib.text import Text
import plots
from plots import BUILDERS

ROOT = Path(__file__).resolve().parents[1]


def audit(fig):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.canvas.draw()
    issues = [str(w.message) for w in caught if 'Glyph' in str(w.message)]
    renderer = fig.canvas.get_renderer()
    bounds = fig.bbox
    for t in fig.findobj(Text):
        if not t.get_visible() or not t.get_text().strip():
            continue
        b = t.get_tightbbox(renderer)
        if b.x0 < -2 or b.y0 < -2 or b.x1 > bounds.x1+2 or b.y1 > bounds.y1+2:
            issues.append('Text outside canvas: '+t.get_text())
    for ax in fig.axes:
        for tickset in [ax.get_xticklabels(), ax.get_yticklabels()]:
            
            
            if any(t.get_rotation() % 180 for t in tickset):
                continue
            boxes = [t.get_tightbbox(renderer) for t in tickset if t.get_visible() and t.get_text()]
            for i, a in enumerate(boxes):
                for b in boxes[i+1:]:
                    if min(a.x1,b.x1)-max(a.x0,b.x0)>1 and min(a.y1,b.y1)-max(a.y0,b.y0)>1:
                        issues.append('Overlapping tick labels')
    return sorted(set(issues))


def verify_frozen_inputs(data_dir):
    manifest_path = data_dir / 'manifest.json'
    if not manifest_path.exists():
        raise FileNotFoundError('Place plot inputs and manifest.json in --data-dir; see data/README.md')
    manifest = json.loads(manifest_path.read_text())
    for name, item in manifest['inputs'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Input manifest must use paths within --data-dir')
        digest = hashlib.sha256((data_dir / relative).read_bytes()).hexdigest()
        if digest != item['sha256']:
            raise ValueError('Frozen input changed: '+name)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--final', action='store_true', help='Export PDF, editable SVG and 600 dpi PNG')
    p.add_argument('--figure', choices=list(BUILDERS), action='append', help='Default: all data figures')
    p.add_argument('--data-dir', type=Path, default=ROOT/'data'/'plot_inputs')
    p.add_argument('--output-dir', type=Path, default=ROOT/'outputs'/'plots')
    a = p.parse_args()
    plots.DATA = a.data_dir.resolve()
    output = a.output_dir.resolve()
    verify_frozen_inputs(plots.DATA)
    apply_style()
    (output/'qa').mkdir(parents=True, exist_ok=True)
    (output/'figures').mkdir(parents=True, exist_ok=True)
    report = {}
    for name in a.figure or BUILDERS:
        fig = BUILDERS[name]()
        issues = audit(fig)
        report[name] = {'layout_issues': issues, 'size_inches': list(fig.get_size_inches())}
        fig.savefig(output/'qa'/f'{name}_preview.png', dpi=200)
        if issues:
            print(name, issues)
        else:
            print(name, 'PASS')
        if a.final:
            if issues:
                raise RuntimeError('Fix layout issues before final export: '+name)
            for ext in ['png', 'pdf', 'svg']:
                fig.savefig(output/'figures'/f'{name}.{ext}', dpi=600)
            Image.open(output/'figures'/f'{name}.png').convert('L').save(
                output/'qa'/f'{name}_grayscale.png', dpi=(600, 600))
        plt.close(fig)
    (output/'qa'/'layout_audit.json').write_text(json.dumps(report, indent=2)+'\n')


if __name__ == '__main__':
    main()
