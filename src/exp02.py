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


def do_geyser_plots(directory):
    plt, sns = _init_(is_agg=True)
    directory = Path(directory)
    # orthogonal cases
    # db3 ok, generate curve and plot, 2 levels
    # - plot-geyser-curve-db3.png
    # - plot-geyser-density-db3.png
    # sym4 ok, generate curve and plot
    # - plot-geyser-curve-sym4.png
    # - plot-geyser-density-sym4.png
    # db4 fails, generate curve and plot, 2 levels
    # sym6 fails, generate curve and plot, 2 levels
    # sym4 fails, generate curve and plot, 1 level
    # sym4 "fails", generate curve and plot, 3 levels
    # biorthogonal
    # bior2.4 ok, generate curve and plot
    # bior4.6



def do_tex_table_exp02(directory):
    import pystache as ps
    with open('src/ch4.tex', 'rt') as fh:
        tpl = ps.parse("\n".join(fh.readlines()), delimiters=('<<','>>'))
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    directory = Path(directory)
    df = _load_data(directory)
    print('>>', list(df))
    nums = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000]
    dist_codes = [('ex01', '(a)', 'db4'), ('ex02', '(b)', 'db4'), ('ex03', '(c)', 'sym3'), ('ex04', '(d)', 'sym3')]
    os.makedirs(str(directory / 'plots2'), exist_ok=True)
    _HD = [('normed', '${}_1\widehat{\mathcal{B}}_J$'), ('diff', '${}_2\widehat{\mathcal{B}}_J$')]
    _TH = [
        (SPWDE.TH_CLASSIC, 'TH1'),
        (SPWDE.TH_ADJUSTED, 'TH2'),
        (SPWDE.TH_EMP_STD, 'TH3')
    ]
    _DENSITY_NAMES = {
        'ex01': 'Kurtotic Mixture 1',
        'ex02': 'Mixture 2',
        'ex03': '2D Comb 1 (claw)',
        'ex04': '2D Smooth comb'
    }
    _WAVE_NAMES = {
        'db4': 'Daubechie 4',
        'sym3': 'Symlet 3'
    }
    for dist_code, fig_letter, wave_name in dist_codes:
        with open(str(directory / 'plots2' / 'ch4-th-table-%s.tex') % dist_code, 'wt') as fh:
            fh.write('%% auto generated %%\n')
            ctx = {
                'DensityName': _DENSITY_NAMES[dist_code],
                'DensityCode': dist_code,
                'FigureLetter': fig_letter,
                'WaveName': _WAVE_NAMES[wave_name]
            }
            fh.write('%% dist_code = %s\n' % dist_code)
            items = []
            linnum = 0
            for idx_n, num_obvs in enumerate(nums):
                first_num = True
                df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs)
                         & (df.algorithm == 'KDE')].copy()
                kde_q1, kde_hd, kde_q3 = df1.hd.quantile([0.25, 0.50, 0.75])
                ## kde_hd = df1.hd.mean()
                kde_hd = '%.4f' % kde_hd
                for idx_p, hd_desc in enumerate(_HD):
                    first_hd = True
                    for j_level in [1,2,3]:
                        item = {'Num': num_obvs, 'HD': hd_desc[1], 'JLevel': j_level, 'KDE': kde_hd}
                        for th_desc in _TH:
                            th_algo, th_var = th_desc
                            df1 = df[ (df.dist_code == dist_code) & (df.num_obvs == num_obvs)
                                      & (df.wave_name == wave_name) & (df.delta_j == j_level)
                                      & (df.best_j == df.start_j + df.delta_j) & (df.algorithm == 'WDE')
                                      & (df.opt_target == hd_desc[0]) & (df.treshold_mode == th_algo)].copy()
                            q1, avg, q3 = df1.hd.quantile([0.25, 0.50, 0.75])
                            ## avg = df1.hd.mean()
                            q1 = '%.4f' % q1
                            q3 = '%.4f' % q3
                            avg = '%.4f' % avg
                            item['%sQ1' % th_var] = q1
                            item['%sOk' % th_var] = avg <= kde_hd
                            item['%sAvg' % th_var] = avg
                            item['%sQ3' % th_var] = q3
                            item['%sNumCoeffs' % th_var] = ':'.join(['%.1f' % v for v in
                                                                     df1.num_coeffs.quantile([0.25, 0.50, 0.75])])
                        item['FirstNum'] = first_num
                        item['FirstHD'] = first_hd
                        if linnum % 6 == 5:
                            item['Border'] = '\hline'
                        elif linnum % 3 == 2:
                            item['Border'] = '\cline{2-12}'
                        else:
                            item['Border'] = '\cline{3-12}'
                        item['LinNum'] = linnum
                        first_num = False
                        first_hd = False
                        linnum += 1
                        ## print(item)
                        items.append(item)
            ctx['Items'] = items
            for line in ps.Renderer().render(tpl, **ctx).splitlines():
                line = line.rstrip()
                if line:
                    fh.write(line)
                    fh.write('\n')


def do_tex_table2_exp02(directory):
    import pystache as ps
    with open('src/ch4-params.tex', 'rt') as fh:
        tpl = ps.parse("\n".join(fh.readlines()), delimiters=('<<','>>'))
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    directory = Path(directory)
    df = _load_data(directory)
    print('>>', list(df))
    nums = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000]
    dist_codes = [('ex01', '(a)', 'db4'), ('ex02', '(b)', 'db4'), ('ex03', '(c)', 'sym3'), ('ex04', '(d)', 'sym3')]
    os.makedirs(str(directory / 'plots2'), exist_ok=True)
    _HD = [('normed', '${}_1\widehat{\mathcal{B}}_J$'), ('diff', '${}_2\widehat{\mathcal{B}}_J$')]
    _TH = [
        (SPWDE.TH_CLASSIC, 'TH1'),
        (SPWDE.TH_ADJUSTED, 'TH2'),
        (SPWDE.TH_EMP_STD, 'TH3')
    ]
    _DENSITY_NAMES = {
        'ex01': 'Kurtotic Mixture 1',
        'ex02': 'Mixture 2',
        'ex03': '2D Comb 1 (claw)',
        'ex04': '2D Smooth comb'
    }
    _WAVE_NAMES = {
        'db4': 'Daubechies 4',
        'sym3': 'Symlet 3'
    }
    for dist_code, fig_letter, wave_name in dist_codes:
        with open(str(directory / 'plots2' / 'ch4-th-table2-%s.tex') % dist_code, 'wt') as fh:
            fh.write('%% auto generated %%\n')
            ctx = {
                'DensityName': _DENSITY_NAMES[dist_code],
                'DensityCode': dist_code,
                'FigureLetter': fig_letter,
                'WaveName': _WAVE_NAMES[wave_name]
            }
            fh.write('%% dist_code = %s\n' % dist_code)
            items = []
            linnum = 0
            for idx_n, num_obvs in enumerate(nums):
                first_num = True
                df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs)
                         & (df.algorithm == 'KDE')].copy()
                kde_q1, kde_hd, kde_q3 = df1.hd.quantile([0.25, 0.50, 0.75])
                ## kde_hd = df1.hd.mean()
                kde_hd = '%.4f' % kde_hd
                for idx_p, hd_desc in enumerate(_HD):
                    first_hd = True
                    for j_level in [1,2,3]:
                        item = {'Num': num_obvs, 'HD': hd_desc[1], 'JLevel': j_level}
                        for th_desc in _TH:
                            th_algo, th_var = th_desc
                            df1 = df[ (df.dist_code == dist_code) & (df.num_obvs == num_obvs)
                                      & (df.wave_name == wave_name) & (df.delta_j == j_level)
                                      & (df.best_j == df.start_j + df.delta_j) & (df.algorithm == 'WDE')
                                      & (df.opt_target == hd_desc[0]) & (df.treshold_mode == th_algo)].copy()
                            q1, avg, q3 = df1.num_coeffs.quantile([0.25, 0.50, 0.75])
                            ## avg = df1.hd.mean()
                            q1 = '%.1f' % q1
                            q3 = '%.1f' % q3
                            avg = '%.1f' % avg
                            _, avg_hd, _ = df1.hd.quantile([0.25, 0.50, 0.75])
                            avg_hd = '%.4f' % avg_hd
                            item['%sQ1' % th_var] = q1
                            item['%sOk' % th_var] = avg_hd <= kde_hd
                            item['%sAvg' % th_var] = avg
                            item['%sQ3' % th_var] = q3
                            item['%sNumCoeffs' % th_var] = ':'.join(['%.1f' % v for v in
                                                                     df1.num_coeffs.quantile([0.25, 0.50, 0.75])])
                        item['FirstNum'] = first_num
                        item['FirstHD'] = first_hd
                        if linnum % 6 == 5:
                            item['Border'] = '\hline'
                        elif linnum % 3 == 2:
                            item['Border'] = '\cline{2-12}'
                        else:
                            item['Border'] = '\cline{3-12}'
                        item['LinNum'] = linnum
                        first_num = False
                        first_hd = False
                        linnum += 1
                        ## print(item)
                        items.append(item)
            ctx['Items'] = items
            for line in ps.Renderer().render(tpl, **ctx).splitlines():
                line = line.rstrip()
                if line:
                    fh.write(line)
                    fh.write('\n')

def do_plot_exp02(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"

    plt, sns = _init_()
    directory = Path(directory)
    df = _load_data(directory)
    print(sorted(set([(str(row[0]), str(row[1])) for _, row in df[['dist_code', 'wave_name']].iterrows()])))
    nums = [250, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000]
    plans = [
        ('ex01', ['sym3', 'sym4', 'db4']),
        ('ex02', ['sym3', 'sym4', 'db4']),
        ('ex03', ['sym3', 'sym4', 'db4']),
        ('ex04', ['sym3', 'sym4', 'db4']),
    ]
    os.makedirs(str(directory / 'plots2'), exist_ok=True)
    lines = []
    lines.append(_HTML_HEAD)
    for planx, a_plan in enumerate(plans):
        dist_code, wavelets = a_plan
        lines.append('<h1>%s</h1>' % (_EX[dist_code],))
        fig = _plot_true(plt, dist_code)
        figname = 'plot-%s-true.png' % (dist_code,)
        full_figname = str(directory / 'plots2' / figname)
        fig.savefig(full_figname, dpi=300)
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
            fig = _plot_fig(plt, sns, dist_code, num_obvs, wavelets, df, excess=0)
            figname = 'plot-%s-%d-D0.png' % (dist_code, num_obvs)
            full_figname = str(directory / 'plots2' / figname)
            fig.savefig(full_figname, dpi=300)
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

        lines.append('\n<hr/>\n')
        col = 0
        for num_obvs in nums:
            ##lines.append('<h2>N = %d</h2>' % (num_obvs,))
            fig = _plot_fig(plt, sns, dist_code, num_obvs, wavelets, df, excess=1)
            figname = 'plot-%s-%d-D1.png' % (dist_code, num_obvs)
            full_figname = str(directory / 'plots2' / figname)
            fig.savefig(full_figname, dpi=300)
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
    with open(str( directory / 'plots2' / 'plots.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots.html')

import matplotlib.lines as mlines

def _plot_fig(plt, sns, dist_code, num_obvs, waves, df, excess=0):
    fig, axs = plt.subplots(3, 1, sharey=True, sharex=True, figsize=(10, 9))
    # wave_name -> mode -> th_mode -> delta_j
    df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs)]
    max_hd = np.percentile(df1.hd, 99)
    kde_hd = {}
    for ix, row in df1[df1.algorithm == 'KDE'].iterrows():
        kde_hd[row.sample_no] = row.hd
    avg_hd = sum(kde_hd.values()) / len(kde_hd.values())
    print(dist_code, num_obvs, avg_hd)
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
                 (df.best_j + excess == df.start_j + df.delta_j) & (df.algorithm == 'WDE')].copy()
        print(df1.shape, end='; ')
        if df1.shape[0] == 0:
            print(dist_code, num_obvs, wave_name, excess)
            continue
        df1.reset_index()
        func = lambda row: '%s\n%s' % (_HD[row.opt_target], _TH[row.treshold_mode])
        #print(list(df1))
        # print(df1.shape)
        df1['mode'] = df1.apply(func, axis=1)
        # copy one
        df2 = df1[df1.delta_j == df1.delta_j.min()].copy()
        df2['hd'] = df2.apply(lambda row: kde_hd.get(row.sample_no, -1), axis=1)
        df2['delta_j'] = 'kde'
        df2 = df2[df2.hd != -1]
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
    #ax.set_title(_EX[dist.code])
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
