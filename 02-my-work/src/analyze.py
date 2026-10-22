"""Collect metrics.json files -> summary table (CSV/markdown) + report figures."""
import json, os, glob, sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

RES = sys.argv[1] if len(sys.argv) > 1 else 'results'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'figures'
os.makedirs(OUT, exist_ok=True)
CLASSES = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# palette (validated, light mode) + text tokens
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4'
INK, INK2, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#8a8984', '#e4e3df', '#ffffff'
plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 7.5, 'axes.titlesize': 8, 'axes.labelsize': 7.5,
    'axes.edgecolor': GRID, 'axes.linewidth': 0.8, 'axes.labelcolor': INK2, 'xtick.color': INK2,
    'ytick.color': INK2, 'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
    'figure.dpi': 300, 'savefig.dpi': 300, 'axes.titlecolor': INK, 'axes.titleweight': 'bold',
    'axes.titlelocation': 'left'})

runs = {}
for f in glob.glob(os.path.join(RES, '*', 'metrics.json')):
    r = json.load(open(f))
    if 'infer_ms_per_img' in r:          # finished runs only
        runs[r['name']] = r


def row(r):
    b = r['best']; e = b['epoch'] - 1; L = r['log']
    return {'name': r['name'], 'params': r['params'], 'macs_M': r['macs'] / 1e6,
            'epoch_s': float(np.mean(L['epoch_time_s'])), 'best_ep': b['epoch'],
            'train_acc': L['train_acc'][e], 'val_acc': b['val_acc'], 'test_acc': b['acc'],
            'macro_f1': b['macro_f1'], 'gap': L['train_acc'][e] - b['val_acc'],
            'test_ep1': L['test_acc'][0], 'test_last': L['test_acc'][-1],
            'infer_ms': r['infer_ms_per_img']}


rows = sorted([row(r) for r in runs.values()], key=lambda d: -d['test_acc'])
with open(os.path.join(OUT, 'summary.csv'), 'w') as f:
    keys = list(rows[0].keys())
    f.write(','.join(keys) + '\n')
    for d in rows:
        f.write(','.join(f'{d[k]:.4f}' if isinstance(d[k], float) else str(d[k]) for k in keys) + '\n')
print(f"{'name':14s} {'params':>8s} {'MACs':>6s} {'s/ep':>5s} {'bestep':>6s} {'train':>6s} {'val':>6s} {'test':>6s} {'F1':>6s} {'gap':>6s} {'ep1':>6s} {'ms':>5s}")
for d in rows:
    print(f"{d['name']:14s} {d['params']:8d} {d['macs_M']:6.2f} {d['epoch_s']:5.0f} {d['best_ep']:6d} {d['train_acc']:6.3f} "
          f"{d['val_acc']:6.3f} {d['test_acc']:6.3f} {d['macro_f1']:6.3f} {d['gap']:6.3f} {d['test_ep1']:6.3f} {d['infer_ms']:5.2f}")
json.dump(rows, open(os.path.join(OUT, 'summary.json'), 'w'), indent=1)

# ------------------------------------------------------------------ figures
from matplotlib import font_manager
for f in ['Carlito-Regular.ttf', 'Carlito-Bold.ttf']:
    if os.path.exists('/usr/share/fonts/truetype/crosextra/' + f): font_manager.fontManager.addfont('/usr/share/fonts/truetype/crosextra/' + f)
plt.rcParams['font.family'] = 'Carlito' if os.path.exists('/usr/share/fonts/truetype/crosextra/Carlito-Regular.ttf') else 'DejaVu Sans'
plt.rcParams['font.size'] = 8.5
plt.rcParams['axes.titlesize'] = 9
plt.rcParams['axes.labelsize'] = 8.5
plt.rcParams.update({'xtick.major.size': 0, 'ytick.major.size': 0, 'axes.axisbelow': True,
                     'xtick.major.pad': 2.5, 'ytick.major.pad': 2.5})

LABEL = {
    'orig_rgb': 'Original ConvNet (sigmoid+BCE)', 'ref_softmax': 'Reference (softmax+CE)',
    'ref_gray': 'Reference, red-only input (bug)', 'width_x2': 'Width ×2 (32-64-128)',
    'width_x4': 'Width ×4 (64-128-256)', 'depth_2': 'Depth −1 (2 conv blocks)',
    'depth_4_pad': 'Depth +1 (4 blocks, pad 1)', 'pad_same': "Padding 1 ('same')",
    'kernel_5': 'Kernel 5×5 (pad 2)', 'vgg_2conv': '2 convs per block (pad 1)',
    'batchnorm': '+ BatchNorm', 'dropout': '+ Dropout 0.3 (FC)', 'flatten_head': 'Flatten head (no GAP)',
    'sigmoid_bn': 'Original + BatchNorm (sigmoid)', 'combo_A': 'Combined: width×2 + pad 1 + BN',
    'avg_pool': 'Average pooling', 'stride_conv': 'Strided conv (no pooling)',
    'vgg_2conv_bn': '2 convs per block + BN', 'depth_4_pad_bn': 'Depth +1 + BN',
}
GREY = '#9b9a96'
XLIM, YLIM = (0.9, 30), (45, 73)
FIG1_H = 2.95
CURVE_YMAX = 72
BEST_COL = ORANGE
LEG2_Y, TITLE2_PAD = 1.13, 14
CM_TICKS = CLASSES
CM_TITLE = '(b) Confusion matrix in row % \u2013 {}'
LEG_Y = -0.2


def lab(n):
    return LABEL.get(n, n)


def fig_curves_and_bars(curve_runs, path, ref='ref_softmax'):
    fig, ax = plt.subplots(1, 2, figsize=(7.25, 2.75), gridspec_kw={'width_ratios': [1, 1.25]})
    cols = [GREY, BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    a = ax[0]
    for n, c in zip(curve_runs, cols):
        L = runs[n]['log']
        a.plot(L['epoch'], [v * 100 for v in L['val_acc']], color=c, lw=1.6, solid_capstyle='round',
               marker='o', ms=2.6, label=lab(n))
    a.set_xlabel('epoch'); a.set_ylabel('validation accuracy (%)')
    a.set_xticks(range(1, 11)); a.set_ylim(0, None)
    a.axhline(10, color=MUTED, lw=0.6); a.text(10, 11, 'chance', ha='right', va='bottom', color=MUTED, fontsize=7.5)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, fontsize=7.2, handlelength=1.4,
             columnspacing=1.0, borderaxespad=0.0)
    a.set_title('(a) Learning curves (validation set)')
    # bars
    b = ax[1]
    rs = sorted(rows, key=lambda d: d['test_acc'])
    y = np.arange(len(rs))
    def bar_col(n):
        if n in ('orig_rgb', 'sigmoid_bn', 'ref_gray'):
            return '#c9c8c3'
        if n == ref:
            return BLUE
        if n.startswith('combo') or n.startswith('aug'):
            return ORANGE
        return '#8fb8e8'
    colr = [bar_col(d['name']) for d in rs]
    b.barh(y, [d['test_acc'] * 100 for d in rs], height=0.62, color=colr)
    for i, d in enumerate(rs):
        b.text(d['test_acc'] * 100 + 0.6, i, f"{d['test_acc'] * 100:.1f}", va='center', fontsize=7.2, color=INK)
    b.set_yticks(y); b.set_yticklabels([lab(d['name']) for d in rs], fontsize=7.4)
    refv = runs[ref]['best']['acc'] * 100
    b.axvline(refv, color=BLUE, lw=0.8)
    b.set_xlabel('test accuracy (%)'); b.set_xlim(0, max(d['test_acc'] for d in rs) * 100 + 7)
    b.grid(axis='y', visible=False)
    b.set_title('(b) Test accuracy of every model (best-val checkpoint)')
    fig.tight_layout(w_pad=1.2)
    fig.savefig(path, bbox_inches='tight', facecolor='white'); plt.close(fig)


def fig_classes(ref, best, path):
    fig, ax = plt.subplots(1, 2, figsize=(7.25, 2.85), gridspec_kw={'width_ratios': [1.1, 1]})
    a = ax[0]
    pr = np.array(runs[ref]['best']['per_class_acc']) * 100
    pb = np.array(runs[best]['best']['per_class_acc']) * 100
    x = np.arange(10); w = 0.38
    a.bar(x - w / 2 - 0.02, pr, w, color=BLUE, label=lab(ref))
    a.bar(x + w / 2 + 0.02, pb, w, color=BEST_COL, label=lab(best))
    a.set_xticks(x); a.set_xticklabels(CLASSES, rotation=35, ha='right', fontsize=7.4)
    a.set_ylabel('test accuracy per class (%)'); a.set_ylim(0, 100)
    a.grid(axis='x', visible=False)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, LEG2_Y), ncol=2, fontsize=7.2, handlelength=1.2)
    a.set_title('(a) Per-class accuracy', pad=TITLE2_PAD)
    b = ax[1]
    cm = np.array(runs[best]['best']['confusion'], dtype=float)
    cmn = cm / cm.sum(1, keepdims=True) * 100
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list('b', ['#ffffff', '#cde2fb', '#86b6ef', '#3987e5', '#1c5cab', '#0d366b'])
    b.imshow(cmn, cmap=cmap, vmin=0, vmax=100)
    for i in range(10):
        for j in range(10):
            v = cmn[i, j]
            if v >= 4.5:
                b.text(j, i, f'{v:.0f}', ha='center', va='center', fontsize=7.0,
                       color='white' if v > 55 else INK)
    b.set_xticks(range(10)); b.set_yticks(range(10))
    b.set_xticklabels(CM_TICKS, rotation=55, ha='right', fontsize=7.2); b.set_yticklabels(CM_TICKS, fontsize=7.2)
    b.set_xlabel('predicted', labelpad=1); b.set_ylabel('true', labelpad=1)
    b.grid(False)
    for s in b.spines.values(): s.set_visible(False)
    b.set_title(CM_TITLE.format(lab(best)), fontsize=8.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(path, bbox_inches='tight', facecolor='white'); plt.close(fig)


if __name__ == '__main__' and len(sys.argv) > 3:
    curve = sys.argv[3].split(',')
    best = sys.argv[4]
    fig_curves_and_bars(curve, os.path.join(OUT, 'fig1.png'))
    fig_classes('ref_softmax', best, os.path.join(OUT, 'fig2.png'))
    print('figures written')


SHORT = {'ref_softmax': 'reference', 'width_x2': 'width ×2', 'width_x4': 'width ×4', 'depth_2': 'depth −1',
         'depth_4_pad': 'depth +1', 'pad_same': 'padding', 'kernel_5': 'kernel 5×5', 'vgg_2conv': '2 convs/block',
         'batchnorm': 'BatchNorm', 'dropout': 'dropout', 'flatten_head': 'flatten head', 'combo_A': 'combined',
         'aug_combo_A': 'combined + aug.', 'avg_pool': 'avg. pooling', 'stride_conv': 'stride 2',
         'vgg_2conv_bn': '2 convs + BN', 'depth_4_pad_bn': 'depth +1 + BN',
         'aug_depth_4_pad_bn': 'depth +1 + BN + aug.'}


def fig_cost(names, path, bench=None, offsets=None):
    """Test accuracy vs compute (MACs, log scale) for the softmax variants."""
    offsets = offsets or {}
    fig, a = plt.subplots(figsize=(3.55, 2.6))
    for n in names:
        r = runs[n]; x = r['macs'] / 1e6; y = r['best']['acc'] * 100
        c = INK2 if n == 'ref_softmax' else (ORANGE if n.startswith('combo') or n.startswith('aug') else BLUE)
        a.scatter([x], [y], s=26, color=c, edgecolors='white', linewidths=1.2, zorder=3)
        dx, dy = offsets.get(n, (5, 0))
        a.annotate(SHORT.get(n, n), (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.2,
                   va='center', ha='left' if dx >= 0 else 'right', color=INK)
    a.set_xscale('log'); a.set_xlabel('compute per image (MMACs, log scale)'); a.set_ylabel('test accuracy (%)')
    a.set_title('Accuracy vs. compute')
    fig.tight_layout(); fig.savefig(path, bbox_inches='tight', facecolor='white'); plt.close(fig)


def fig_curves_and_cost(curve_runs, cost_runs, path, extra=None, offsets=None):
    """(a) validation curves, (b) test accuracy vs compute (log MACs)."""
    offsets = offsets or {}
    fig, ax = plt.subplots(1, 2, figsize=(7.25, FIG1_H), gridspec_kw={'width_ratios': [1, 1.12]})
    cols = [GREY, BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
    a = ax[0]
    for n, c in zip(curve_runs, cols):
        L = runs[n]['log']
        a.plot(L['epoch'], [v * 100 for v in L['val_acc']], color=c, lw=1.6, solid_capstyle='round',
               marker='o', ms=2.6, label=lab(n))
    a.set_xlabel('epoch', labelpad=1); a.set_ylabel('validation accuracy (%)')
    a.set_xticks(range(1, 11)); a.set_ylim(0, CURVE_YMAX)
    a.axhline(10, color=MUTED, lw=0.6); a.text(10, 11, 'chance', ha='right', va='bottom', color=MUTED, fontsize=7.5)
    a.legend(loc='upper center', bbox_to_anchor=(0.5, LEG_Y), ncol=2, fontsize=7.2, handlelength=1.4,
             columnspacing=1.0, borderaxespad=0.0)
    a.set_title('(a) Learning curves (validation set)')
    b = ax[1]
    pts = [(n, runs[n]['macs'] / 1e6, runs[n]['best']['acc'] * 100) for n in cost_runs]
    if extra:
        pts += [(n, extra[n]['macs'] / 1e6, extra[n]['best']['acc'] * 100) for n in extra]
    for n, x, y in pts:
        if n == 'ref_softmax':
            c = BLUE
        elif n.startswith('combo') or n == 'aug_combo_A':
            c = ORANGE
        else:
            c = '#6f9fd8'
        hollow = n.startswith('aug')
        b.scatter([x], [y], s=30, facecolors='white' if hollow else c, edgecolors=c if hollow else 'white',
                  linewidths=1.4 if hollow else 1.0, zorder=3)
        dx, dy = offsets.get(n, (5, 0))
        b.annotate(SHORT.get(n, n), (x, y), xytext=(dx, dy), textcoords='offset points', fontsize=7.2,
                   va='center', ha='center' if dx == 0 else ('left' if dx > 0 else 'right'), color=INK)
    b.set_xscale('log')
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
    b.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 5, 10, 20])); b.xaxis.set_major_formatter(FixedFormatter(['0.5', '1', '2', '5', '10', '20']))
    b.xaxis.set_minor_locator(NullLocator())
    b.set_xlim(XLIM); b.set_ylim(YLIM)
    b.set_xlabel('compute per image (million MACs, log scale)'); b.set_ylabel('test accuracy (%)')
    b.set_title('(b) Test accuracy vs. compute (softmax models)')
    fig.tight_layout(w_pad=1.4)
    fig.savefig(path, bbox_inches='tight', facecolor='white'); plt.close(fig)
