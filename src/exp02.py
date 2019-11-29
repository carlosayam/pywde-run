import os
import sys
import itertools as itt
import numpy as np
import pandas as pd

from pathlib import Path
from dist_codes import dist_from_code
from pywde.spwde import SPWDE

_EX = {
    'ex01': 'Mix 1',
    'ex02': 'Mix 2',
    'ex03': 'Comb 1',
    'ex04': 'Comb 2',
}

def do_compare_algos(directory):
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
    # dist_code, wave_name, num_obvs, sample_no
    # -> best_j per HD
    resp = {}
    os.makedirs(str(directory / 'plots2'), exist_ok=True)
    lines = []
    lines.append(_HTML_HEAD)
    for a_plan in plans:
        dist_code, wavelets = a_plan
        print(dist_code)
        fig, axs = plt.subplots(1, 2, sharey=True, sharex=True, figsize=(9, 2))
        for wave_name, ax1 in zip(wavelets, axs):
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
            ax1.set_title('%s' % wave_name)
            ax1.set(xlabel='Sample size', ylabel='$\hat{j} - J$')
            ax1.grid(True)
        lines.append('<h2>%s</h2>' % dist_code)
        figname = 'plot-%s.png' % dist_code
        full_figname = str(directory / 'plots2' / figname)
        fig.savefig(full_figname)
        plt.close(fig)
        line = '<img src="%s" alt="%s" style="width:70%%"/>\n' % (figname, dist_code)
        lines.append('<div>\n')
        lines.append(line)
        fig = _plot_true(plt, dist_code)
        figname = 'plot-%s-true.png' % (dist_code,)
        full_figname = str(directory / 'plots2' / figname)
        fig.savefig(full_figname)
        alt_tit = '%s, true' % (dist_code,)
        line = '<img src="%s" alt="%s" style="width:20%%"/>\n' % (figname, alt_tit)
        lines.append(line)
        lines.append('</div>\n')
        plt.close(fig)
    lines.append(_HTML_END)
    with open(str( directory / 'plots2' / 'plots.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots.html')


def do_plot_exp02(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"

    plt, sns = _init_()
    directory = Path(directory)
    df = _load_data(directory)
    print(sorted(set([(str(row[0]), str(row[1])) for _, row in df[['dist_code', 'wave_name']].iterrows()])))
    nums = [250, 500, 1000, 1500, 2000, 3000, 4000]
    plans = [
        ('ex01', ['sym3', 'sym4', 'db4']),
        ('ex02', ['sym3', 'sym4', 'db4']),
        ('ex03', ['sym3', 'sym4', 'db4']),
        ('ex04', ['sym3', 'sym4', 'db4']),
    ]
    os.makedirs(str(directory / 'plots'), exist_ok=True)
    lines = []
    lines.append(_HTML_HEAD)
    for planx, a_plan in enumerate(plans):
        dist_code, wavelets = a_plan
        lines.append('<h1>%s</h1>' % (_EX[dist_code],))
        fig = _plot_true(plt, dist_code)
        figname = 'plot-%s-true.png' % (dist_code,)
        full_figname = str(directory / 'plots' / figname)
        fig.savefig(full_figname)
        #fig.close()
        plt.close(fig)
        print('>', figname)
        alt_tit = '%s, true' % (dist_code,)
        line = '<img src="%s" alt="%s" style="width:45%%"/>\n' % (figname, alt_tit)
        lines.append('<div>')
        lines.append(line)
        col = 1
        for num_obvs in nums:
            ##lines.append('<h2>N = %d</h2>' % (num_obvs,))
            fig = _plot_fig(plt, sns, dist_code, num_obvs, wavelets, df)
            figname = 'plot-%s-%d.png' % (dist_code, num_obvs)
            full_figname = str(directory / 'plots' / figname)
            fig.savefig(full_figname)
            plt.close(fig)
            print('>', figname)
            alt_tit = '%s, %d' % (dist_code, num_obvs)
            line = '<img src="%s" alt="%s" style="width:45%%"/>\n' % (figname, alt_tit)
            if col == 0:
                lines.append('<div>')
            lines.append(line)
            if col == 1:
                lines.append('</div>\n')
            col = (col + 1) % 2
        if col == 1:
            lines.append('</div>\n')
        # lines.append('</div>\n')
        if planx < 3:
            lines.append('<footer />\n')
    lines.append(_HTML_END)
    with open(str( directory / 'plots' / 'plots.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots.html')

import matplotlib.lines as mlines

def _plot_fig(plt, sns, dist_code, num_obvs, waves, df):
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(10, 9))
    # wave_name -> mode -> th_mode -> delta_j
    df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs)]
    max_hd = np.percentile(df1.hd, 95)
    kde_hd = {}
    for ix, row in df[df.algorithm == 'KDE'].iterrows():
        kde_hd[row.sample_no] = row.hd
    _HD = {'normed': '$HD_1$', 'diff': '$HD_2$'}
    _TH = {SPWDE.TH_CLASSIC: '$C$',
           SPWDE.TH_ADJUSTED: '$C\sqrt{\Delta j}$',
           SPWDE.TH_EMP_STD: '$C\sigma^B$'}
    _Y = '$HD_i$'
    _DELTA_J = '$\Delta J$'
    for px, wave_name in enumerate(waves):
        title = wave_name
        # print(list(df))
        # print(df.dist_code.unique())
        # print(df.num_obvs.unique())
        # print(df.wave_name.unique())
        # print(dist_code, num_obvs, wave_name)
        df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs) & (df.wave_name == wave_name) &
                 (df.best_j == df.start_j + df.delta_j)].copy()
        df1.reset_index()
        func = lambda row: '%s\n%s' % (_HD[row.opt_target], _TH[row.treshold_mode])
        #print(list(df1))
        # print(df1.shape)
        df1['mode'] = df1.apply(func, axis=1)
        # copy one
        df2 = df1[df1.delta_j == 1].copy()
        df2['hd'] = df2.apply(lambda row: kde_hd[row.sample_no], axis=1)
        df2['delta_j'] = 'kde'
        df2.reset_index()

        df1 = pd.concat([df1, df2])
        # axs[px].scatter(df1.modes, df1.hd, alpha=0.2)
        axs[px].set_title(title)
        ##axs[px].set_legend(loc='bottom right')
        # import code
        # code.interact('**', local=locals())
        # print('delta_j', df1.delta_j.unique())

        sns.boxplot(x='mode', y='hd', hue='delta_j', data=df1, ax=axs[px],
                    hue_order=['kde', 1, 2, 3])
        handles, labels = axs[px].get_legend_handles_labels()
        axs[px].legend(handles, labels, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title=_DELTA_J)
        # ylim = axs[px].get_ylim()
        # ylim = (0, ylim[1])
        axs[px].set(ylim=(0, max_hd))
        #ax1.legend_.set_title('Algorithm')
    xmin, xmax = axs[len(waves) - 1].get_xbound()
    for px in range(len(waves)):
        l = mlines.Line2D([xmin, xmax], [0, 0], linestyle='--', color='r')
        axs[px].add_line(l)
        if px < len(waves) - 1:
            axs[px].set_xlabel('')
    fig.suptitle('N=%d' % (num_obvs,))
    ##fig.legend(loc='upper right')
    ##plt.show()
    return fig


def _plot_true(plt, dist_code):
    dist = dist_from_code(dist_code)
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    max_v = (zz / zz_sum).max()
    fig = plt.figure(figsize=(4.5, 4.5))
    ax = fig.gca(projection='3d')
    elev = 15
    azim = -60
    ax.view_init(elev, azim)
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap='BuGn')
    ax.set_title(_EX[dist.code])
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
    df, numf = None, 0
    for fname in directory.glob('*.tab'):
        sa = pd.read_csv(fname, delimiter='\t')
        numf += 1
        if df is None:
            df = sa
        else:
            df = pd.concat((df, sa), sort=False)
    print('Read %d files' % numf)
    return df


def exp02_repl(directory):
    df = _load_data(Path(directory))
    plt, sns = _init_(is_agg=False)
    from code import interact
    import pandas as pd
    import numpy as np
    interact('** df, plt, sns, pd, np', local=locals())



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
