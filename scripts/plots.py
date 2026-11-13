# Draw evaluation figures from prepared summary data.
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from style import BLUE, ORANGE, GRAY, METRICS, CMAP, SIZES, panel_headings, light_grid

DATA = Path(__file__).resolve().parents[1] / 'data' / 'plot_inputs'


def load_json(name):
    return json.loads((DATA / name).read_text())


def binary_performance():
    r = pd.read_csv(DATA / 'binary_roc.csv')
    p = pd.read_csv(DATA / 'binary_pr.csv')
    s = load_json('binary_summary.json')
    fig = plt.figure(figsize=SIZES['Fig2_binary_performance'])
    aa = fig.add_axes([.065, .235, .205, .585])
    bb = fig.add_axes([.365, .235, .205, .585])
    cc = fig.add_axes([.700, .235, .179, .585])
    op = s['operating_point']
    aa.plot(r.false_positive_rate, r.sensitivity, color=BLUE, lw=1.4)
    aa.plot(op['FPR'], op['sensitivity'], 'o', color=ORANGE, ms=4.5,
            mec='white', mew=.45, zorder=5)
    aa.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, .5, 1],
           yticks=[0, .25, .5, .75, 1], xlabel='False-positive rate', ylabel='Sensitivity')
    aa.text(.96, .06, f"AUROC {s['AUROC']:.3f}", transform=aa.transAxes,
            ha='right', va='bottom', fontsize=7)
    bb.plot(p.recall, p.precision, color=BLUE, lw=1.4)
    bb.axhline(s['positive_prevalence'], color=GRAY, ls='--', lw=.8)
    bb.plot(op['sensitivity'], op['precision'], 'o', color=ORANGE, ms=4.5,
            mec='white', mew=.45, zorder=5)
    bb.set(xlim=(0, 1), ylim=(0, 1.02), xticks=[0, .5, 1],
           yticks=[0, .25, .5, .75, 1], xlabel='Sensitivity (recall)', ylabel='Precision')
    bb.text(.04, .06, f"AP {s['average_precision']:.3f}\nBaseline {s['positive_prevalence']:.1%}",
            transform=bb.transAxes, ha='left', va='bottom', fontsize=7)
    cm = np.asarray(s['confusion_matrix_true_rows_predicted_columns'])
    share = cm / cm.sum(axis=1, keepdims=True)
    im = cc.pcolormesh(np.arange(3)-.5, np.arange(3)-.5, share, cmap=CMAP,
                       vmin=0, vmax=1, shading='flat', edgecolors='white', linewidth=.8)
    cc.set_xlim(-.5, 1.5)
    cc.set_ylim(1.5, -.5)
    cc.set(xticks=[0, 1], yticks=[0, 1], xticklabels=['Non-SOP', 'SOP'],
           yticklabels=['Non-SOP', 'SOP'], xlabel='Predicted label', ylabel='Original label')
    cc.tick_params(length=0, pad=3)
    for i in range(2):
        for j in range(2):
            cc.text(j, i, f'{cm[i,j]:,}\n{share[i,j]:.1%}', ha='center', va='center',
                    fontsize=7.5, color='white' if share[i,j] > .6 else '#25343d')
    cc.set_xticks([-.5, .5, 1.5], minor=True)
    cc.set_yticks([-.5, .5, 1.5], minor=True)
    cc.grid(which='minor', color='white', lw=.8)
    cc.tick_params(which='minor', bottom=False, left=False)
    cax = fig.add_axes([.902, .235, .012, .585])
    cb = fig.colorbar(im, cax=cax, ticks=[0, .5, 1])
    cb.solids.set_rasterized(False)
    cb.set_label('Within-class fraction', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(.5)
    for ax in (aa, bb):
        light_grid(ax, 'y')
    panel_headings(fig, [aa, bb, cc], ['ROC curve', 'Precision–recall curve', 'Confusion matrix'])
    return fig


def detection_timing():
    b = load_json('timing_bundle.json')
    fig = plt.figure(figsize=SIZES['Fig3_detection_timing'])
    aa = fig.add_axes([.185, .645, .285, .275])
    bb = fig.add_axes([.685, .645, .285, .275])
    cc = fig.add_axes([.185, .115, .285, .285])
    dd = fig.add_axes([.685, .115, .285, .285])
    im = aa.imshow(np.asarray(b['heat']), aspect='auto', interpolation='nearest',
                   extent=[-38, 30, 131.5, .5], cmap=CMAP, vmin=0, vmax=1, rasterized=True)
    aa.axvline(0, color='white', ls='--', lw=1.5)
    aa.axvline(0, color='.25', ls='--', lw=.7)
    aa.set(xticks=[-30, -15, 0, 15, 30], yticks=[1, 40, 80, 131],
           ylabel='Seizure event', xlabel='Time from annotated onset (s)')
    cax = fig.add_axes([.185, .525, .285, .012])
    cb = fig.colorbar(im, cax=cax, orientation='horizontal', ticks=[0, .5, 1])
    cb.set_label('TCMP-Former probability', fontsize=7, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2)
    cb.outline.set_linewidth(.5)
    tr = b['trajectory']
    t = np.asarray(tr['sample_time_s'])
    bb.fill_between(t, tr['ci_low'], tr['ci_high'], color=BLUE, alpha=.18, lw=0)
    bb.plot(t, tr['patient_mean'], color=BLUE, lw=1.4)
    bb.axvline(0, color='.35', ls='--', lw=.7)
    bb.set(xlim=(-38, 30), ylim=(0, 1.02), xticks=[-30, -15, 0, 15, 30],
           yticks=[0, .25, .5, .75, 1], xlabel='Time from annotated onset (s)',
           ylabel='TCMP-Former probability')
    order = [('eegnet', 'EEGNet'), ('bilstm', 'BiLSTM'),
             ('transformer', 'Transformer'), ('resnet1d', 'ResNet1D')]
    for i, (key, label) in enumerate(order):
        q = b['models']['sop_comparisons'][key]['mean_lead_s']
        v, (lo, hi) = q['estimate'], q['ci95']
        cc.errorbar(v, len(order)-1-i, xerr=[[v-lo], [hi-v]], fmt='o',
                    color=BLUE, ms=4.5, capsize=3, lw=1.1,
                    mfc='white' if key == 'resnet1d' else BLUE)
    cc.axvline(0, color='.45', ls='--', lw=.7)
    cc.set(xlim=(-.5, 10), ylim=(-.65, len(order)-.35), xticks=[0, 2, 4, 6, 8, 10],
           yticks=list(reversed(range(len(order)))), yticklabels=[r[1] for r in order],
           xlabel='Mean paired lead of TCMP-Former (s)')
    cc.tick_params(axis='y', length=0, pad=6)
    cc.spines['left'].set_visible(False)
    gains = [r for r in b['hops']['within_model_gains'] if r['model'] == 'sop']
    dd.plot([1, .5, .25], [0] + [r['mean_gain_s']['estimate'] for r in gains],
            'o-', color=BLUE, lw=1.2, ms=4.5)
    for r in gains:
        q = r['mean_gain_s']; v, (lo, hi) = q['estimate'], q['ci95']
        dd.errorbar(r['hop_s'], v, yerr=[[v-lo], [hi-v]], fmt='none',
                    ecolor=BLUE, capsize=3, lw=1.1)
    dd.set(xlim=(1.08, .17), ylim=(-.02, .52), xticks=[1, .5, .25],
           xticklabels=['1.0', '0.5', '0.25'], yticks=[0, .1, .2, .3, .4, .5],
           xlabel='Inference hop (s)', ylabel='Mean localization gain (s)')
    panel_headings(fig, [aa, bb, cc, dd], ['Event-level model output', 'Patient-level trajectory',
                                         'Paired detection timing', 'Inference hop sensitivity'])
    return fig


def typing_performance():
    pc = pd.read_csv(DATA / 'typing_per_class.csv')
    folds = pd.read_csv(DATA / 'typing_folds.csv')
    fig = plt.figure(figsize=SIZES['Fig4_typing_performance'])
    aa = fig.add_axes([.18, .59, .30, .26])
    bb = fig.add_axes([.685, .59, .30, .26])
    cc = fig.add_axes([.18, .115, .30, .26])
    dd = fig.add_axes([.685, .115, .30, .26])
    y = np.arange(len(pc))
    for metric, label, color, off, marker in zip(
            ['precision', 'recall', 'f1'], ['Precision', 'Recall', 'F1'],
            METRICS, [-.17, 0, .17], ['o', 's', 'D']):
        aa.scatter(pc[metric], y+off, s=19, marker=marker, color=color,
                   edgecolor='white', linewidth=.35, label=label, zorder=3)
    aa.set(yticks=y, yticklabels=pc.abbrev, xlim=(.48, 1.01),
           xticks=np.arange(.5, 1.01, .1), xlabel='Score')
    aa.invert_yaxis()
    fig.legend(*aa.get_legend_handles_labels(), frameon=False, loc='upper left',
               bbox_to_anchor=(.18, .905), ncol=3, fontsize=6.8,
               handletextpad=.3, columnspacing=.8, borderaxespad=0)
    light_grid(aa, 'x')
    bars = bb.barh(y, pc.support, color=['#c7dfe8', BLUE] + ['#c7dfe8']*6,
                   edgecolor='#52616a', linewidth=.4)
    bb.set(yticks=y, yticklabels=pc.abbrev, xlabel='Test samples (n)', xlim=(0, 3050),
           xticks=[0, 1000, 2000, 3000])
    bb.invert_yaxis()
    for bar, val in zip(bars, pc.support):
        bb.text(val+35, bar.get_y()+bar.get_height()/2, f'{val:,}',
                va='center', ha='left', fontsize=6.5)
    light_grid(bb, 'x')
    x = folds.fold.to_numpy()
    for k, (metric, label, color, marker) in enumerate(zip(
            ['accuracy', 'macro_f1', 'balanced_accuracy'],
            ['Accuracy', 'Macro-F1', 'Balanced accuracy'], METRICS, ['o', 's', 'D'])):
        cc.scatter(x+(k-1)*.12, folds[metric], s=23, color=color, marker=marker,
                   edgecolor='white', linewidth=.4, label=label, zorder=3)
    cc.set(xticks=x, xticklabels=[str(i) for i in x], ylim=(.65, .96),
           xlabel='Fold', ylabel='Score')
    fig.legend(*cc.get_legend_handles_labels(), frameon=False, loc='upper left',
               bbox_to_anchor=(.18, .43), ncol=3, fontsize=6.8,
               handletextpad=.3, columnspacing=.8, borderaxespad=0)
    light_grid(cc, 'y')
    pooled = load_json('typing_plot_constants.json')['pooled_calibrated_ece']
    dd.scatter(x, folds.ece_calibrated, s=27, marker='D', color=METRICS[2],
               edgecolor='white', linewidth=.4, zorder=3)
    dd.axhline(pooled, color='#444444', ls='--', lw=.8)
    dd.set(xticks=x, xticklabels=[str(i) for i in x], xlim=(.7, 5.3),
           ylim=(.025, .105), yticks=np.arange(.03, .101, .01),
           ylabel='Expected calibration error', xlabel='Fold')
    dd.text(.82, pooled+.002, f'Pooled calibrated ECE = {pooled:.3f}',
            color='#333333', va='bottom', ha='left', fontsize=6.5)
    light_grid(dd, 'y')
    for letter, ax, title, ytitle in zip('abcd', [aa, bb, cc, dd],
            ['Class-wise discrimination', 'Class distribution',
             'Five-fold test stability', 'Probability calibration'], [.94, .94, .465, .465]):
        xleft = ax.get_position().x0
        fig.text(xleft-.23/fig.get_figwidth(), ytitle, letter, fontsize=11,
                 weight='bold', ha='left', va='baseline')
        fig.text(xleft, ytitle, title, fontsize=9, ha='left', va='baseline')
    return fig


def typing_confusion():
    df = pd.read_csv(DATA / 'typing_confusion_counts.csv', index_col=0)
    counts = df.to_numpy(dtype=int)
    pct = counts/counts.sum(axis=1, keepdims=True)*100
    names = df.columns.tolist()
    fig = plt.figure(figsize=SIZES['Fig5_typing_confusion'])
    ax = fig.add_axes([.205, .245, .650, .7194])
    im = ax.pcolormesh(np.arange(9)-.5, np.arange(9)-.5, pct, cmap=CMAP,
                       vmin=0, vmax=100, edgecolors='white', linewidth=.6, shading='flat')
    ax.set_aspect('equal')
    ax.set(xlim=(-.5, 7.5), ylim=(7.5, -.5), xticks=range(8), yticks=range(8),
           xticklabels=names, yticklabels=names, xlabel='Predicted SOP class', ylabel='True SOP class')
    plt.setp(ax.get_xticklabels(), rotation=40, ha='right', rotation_mode='anchor')
    ax.tick_params(length=0, pad=6)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(8):
        for j in range(8):
            color = 'white' if pct[i,j] >= 60 else '#25343d'
            ax.text(j, i-.115, f'{pct[i,j]:.1f}%', ha='center', va='center', fontsize=8, color=color)
            ax.text(j, i+.18, f'n={counts[i,j]}', ha='center', va='center', fontsize=6.5, color=color)
    pos = ax.get_position()
    cax = fig.add_axes([.888, pos.y0, .019, pos.height])
    cb = fig.colorbar(im, cax=cax, ticks=[0, 20, 40, 60, 80, 100])
    cb.solids.set_rasterized(False)
    cb.set_label('Row-normalized proportion (%)', fontsize=8, labelpad=6)
    cb.outline.set_linewidth(.5)
    cb.ax.tick_params(length=3, width=.5, labelsize=7.5)
    return fig


def matched_interictal():
    d = pd.read_csv(DATA / 'interictal_summary.csv')
    
    labels = [str((Decimal(str(round(v, 14)))*100).quantize(Decimal('.01'), rounding=ROUND_HALF_UP))+'%'
              for v in d['mean']]
    fig = plt.figure(figsize=SIZES['FigS2_matched_interictal'])
    ax = fig.add_axes([.19, .255, .765, .69])
    for i, row in d.iterrows():
        v, lo, hi = np.array([row['mean'], row.ci_lower, row.ci_upper])*100
        color = BLUE if i == 0 else GRAY
        ax.errorbar(i, v, yerr=[[v-lo], [hi-v]], fmt='o' if i == 0 else 's',
                    ms=5.5, color=color, mfc=color if i == 0 else 'white',
                    mew=1, elinewidth=1.2, capsize=4, capthick=1)
        ax.text(i, hi+1.1, labels[i], ha='center', va='bottom', fontsize=8)
    ax.set(xlim=(-.6, 1.6), ylim=(0, 30), xticks=[0, 1],
           xticklabels=['Seizure-adjacent', 'Matched interictal'],
           yticks=[0, 5, 10, 15, 20, 25, 30], ylabel='Positive segments (%)')
    ax.tick_params(axis='x', length=0, pad=8)
    return fig


BUILDERS = {
    'Fig2_binary_performance': binary_performance,
    'Fig3_detection_timing': detection_timing,
    'Fig4_typing_performance': typing_performance,
    'Fig5_typing_confusion': typing_confusion,
    'FigS2_matched_interictal': matched_interictal,
}
