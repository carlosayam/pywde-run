import os
import sys
import itertools as itt
import numpy as np
import pandas as pd

from pathlib import Path
from dist_codes import dist_from_code


def do_compare_algos(directory):
    plt, sns = _init_()
    directory = Path(directory)
    df = _load_data(directory)
    #df.to_csv('exp01.csv', index=False)
    #raise RuntimeError('')
    nums = [100, 500, 1000, 1500, 2500, 3500, 5000]
    plans = [
        ('mix8', ['db4', 'sym6']),
        ('tmx4', ('sym6', 'bior2.8')),
        ('mix9', ('db4', 'sym6')),
        ('mix6', ('sym6', 'bior3.9')),
    ]
    # dist_code, wave_name, num_obvs, sample_no
    # -> best_j per HD
    resp = {}
    os.makedirs(str(directory / 'plots2'), exist_ok=True)
    lines = []
    lines.append(_HTML_HEAD)
    for a_plan in plans:
        dist_code, wavelets = a_plan
        print(dist_code)
        lines.append('<h2>%s</h2>' % dist_code)
        lines.append('<div>\n')
        for wave_name in wavelets:
            fig, ax1 = plt.subplots(1, 1, sharey=True, sharex=True, figsize=(5.7, 4.0))
            series = {'num_obvs': [], 'algorithm': [], 'deltaj': []}
            for num_obvs, bestj_algo in itt.product(nums, ['normed', 'diff']):
                df1 = df[((df.dist_code == dist_code) & (df.num_obvs == num_obvs)
                          & (df.wave_name == wave_name) & (df.bestj_algo == bestj_algo))]
                # sample_no, j
                j_hd = df1.loc[df1.groupby(['sample_no'])['hd'].idxmin()]['j']
                j_algo = df1.loc[df1.groupby(['sample_no'])['b_hat_j'].idxmax()]['j']
                #print(j_hd.values)
                #print(j_algo.values)
                series['num_obvs'].append('%4d' % num_obvs)
                series['algorithm'].append(bestj_algo)
                series['deltaj'].append((j_algo.values - j_hd.values).mean())
            data = pd.DataFrame(series)
            sns.lineplot(x='num_obvs', y='deltaj', hue='algorithm', data=data, ax=ax1, dashes=True)
            leg = ax1.get_legend()
            # new_title = 'Method'
            # leg.set_title(new_title)
            # print(leg.texts, len(leg.texts))
            new_labels = ['', '${}_1\,\hat{J}_n$', '${}_2\,\hat{J}_n$']
            for t, l in zip(leg.texts, new_labels): t.set_text(l)
            leg.texts = leg.texts[1:]
            # ax1.set_title('Wavelet %s' % wave_name)
            ax1.set(xlabel='Sample size', ylabel='$\hat{J}_n - J^{*}_n$')
            ax1.grid(True)
            ax1.set_ylim((-1, 0.1))
            figname = 'plot-%s-%s.png' % (dist_code, wave_name)
            full_figname = str(directory / 'plots2' / figname)
            fig.savefig(full_figname, dpi=300)
            plt.close(fig)
            line = '<img src="%s" alt="%s" style="width:35%%"/>\n' % (figname, dist_code)
            lines.append(line)
        fig = _plot_true(plt, dist_code)
        figname = 'plot-%s-true.png' % (dist_code,)
        full_figname = str(directory / 'plots2' / figname)
        fig.savefig(full_figname, dpi=300)
        alt_tit = '%s, true' % (dist_code,)
        line = '<img src="%s" alt="%s" style="width:20%%"/>\n' % (figname, alt_tit)
        lines.append(line)
        lines.append('</div>\n')
        plt.close(fig)
    lines.append(_HTML_END)
    with open(str( directory / 'plots2' / 'plots3.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots3.html')



def do_plot_exp01(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"

    plt, sns = _init_()
    directory = Path(directory)
    df = _load_data(directory)
    nums = [100, 500, 1000, 1500, 2500, 3500, 5000]
    plans = [
        ('mix8', ['db4', 'sym6']),
        ('tmx4', ('sym6', 'bior2.8')),
        ('mix9', ('db4', 'sym6')),
        ('mix6', ('sym6', 'bior3.9')),
    ]
    os.makedirs(str(directory / 'plots1'), exist_ok=True)
    lines = []
    lines.append(_HTML_HEAD)
    for a_plan in plans:
        dist_code, wavelets = a_plan
        for wave_name in wavelets:
            lines.append('<h1>%s - %s</h1>' % (dist_code, wave_name))
            col = 0
            for num_obvs in nums:
                fig = _plot_fig(plt, sns, dist_code, num_obvs, wave_name, df)
                figname = 'plot-%s-%s-%d.png' % (dist_code, wave_name, num_obvs)
                full_figname = str(directory / 'plots1' / figname)
                fig.savefig(full_figname, dpi=300)
                plt.close(fig)
                print('>', figname)
                alt_tit = '%s, %d, %s' % (dist_code, num_obvs, wave_name)
                line = 'n = %d<br/><img src="%s" alt="%s" style="width:30%%"/>\n' % (num_obvs, figname, alt_tit)
                if col == 0:
                    lines.append('<div>')
                lines.append(line)
                if col == 2:
                    lines.append('</div>\n')
                col = (col + 1) % 3
            fig = _plot_true(plt, dist_code)
            figname = 'plot-%s-true.png' % (dist_code,)
            full_figname = str(directory / 'plots1' / figname)
            fig.savefig(full_figname, dpi=300)
            plt.close(fig)
            print('>', figname)
            alt_tit = '%s, true' % (dist_code,)
            line = '<img src="%s" alt="%s" style="width:45%%"/>\n' % (figname, alt_tit)
            lines.append(line)
            lines.append('</div>\n')
            lines.append('<footer />\n')
    lines.append(_HTML_END)
    with open(str( directory / 'plots1' / 'plots.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots.html')


def _plot_fig(plt, sns, dist_code, num_obvs, wave_name, df):
    fig, (ax1, ax2) = plt.subplots(1, 2, sharey=False, sharex=True, figsize=(5.2,5.5))
    df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs) & (df.wave_name == wave_name)]
    sns.lineplot(x='j', y='b_hat_j', hue='bestj_algo', data=df1, ax=ax1, ci='sd', err_style='band', estimator='mean')
    ## ax1.set_title('%s, N=%d, %s\n$\hat{\mathcal{B}}_j$ ' % (dist_code, num_obvs, wave_name))
    #ax1.legend_.set_title('Algorithm')
    sns.lineplot(x='j', y='hd', hue='bestj_algo', data=df1, ax=ax2, ci='sd', err_style='band', estimator='mean')
    #ax2.set_title('$HD^2$')
    #ax1.legend_.set_title('Algorithm')
    df2 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs) & (df.method == 'KDE')]
    ax2.plot(range(7), (df2.hd.mean(),) * 7, 'k:')
    ax2.fill_between(range(7), (df2.hd.mean()-2*df2.hd.std(),) * 7, (df2.hd.mean()+2*df2.hd.std(),) * 7, alpha=0.3, color='k')
    ax1.set(ylabel='${}_p\widehat{\mathcal{B}}_J$')
    ax2.set(ylabel='$HD({}_p\hat{g}_J,f)$')
    ax2.set(xlabel='$J$')
    for ax in [ax1, ax2]:
        leg = ax.get_legend()
        new_labels = ['', '${}_1\,\hat{J}_n$', '${}_2\,\hat{J}_n$']
        for t, l in zip(leg.texts, new_labels): t.set_text(l)
        leg.texts = leg.texts[1:]
    return fig


def _plot_true(plt, dist_code):
    dist = dist_from_code(dist_code)
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    max_v = (zz / zz_sum).max()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap='BuGn')
    # ax.set_title(dist.code)
    ax.set_zlim(0, 1.1 * max_v)
    return fig


def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)



def _init_(is_agg=True):
    import matplotlib
    if is_agg:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    from mpl_toolkits.mplot3d import Axes3D
    return plt, sns


def _load_data(directory):
    df = None
    for fname in directory.glob('*.tab'):
        sa = pd.read_csv(fname, delimiter='\t', header=None, names=['dist_code', 'num_obvs', 'sample_no',
                                                                    'method', 'wave_name', 'bestj_algo',
                                                                    'j', 'is_best', 'b_hat_j', 'hd', 'elapsed_time'])
        if df is None:
            df = sa
        else:
            df = pd.concat((df, sa), sort=False)
    return df


_HTML_HEAD = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8" />
<title>Plots</title>
<style>
@media print {
  h1 {page-break-before: always;}
}
</style>
</head>

<body>
"""

_HTML_END = """ 
</body>
</html>
"""
