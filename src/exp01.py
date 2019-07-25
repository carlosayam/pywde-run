import os
import numpy as np
import pandas as pd
from pathlib import Path
from dist_codes import dist_from_code


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
    os.makedirs(str(directory / 'plots'), exist_ok=True)
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
                full_figname = str(directory / 'plots' / figname)
                fig.savefig(full_figname)
                plt.close(fig)
                print('>', figname)
                alt_tit = '%s, %d, %s' % (dist_code, num_obvs, wave_name)
                line = '<img src="%s" alt="%s" style="width:30%%"/>\n' % (figname, alt_tit)
                if col == 0:
                    lines.append('<div>')
                lines.append(line)
                if col == 2:
                    lines.append('</div>\n')
                col = (col + 1) % 3
            fig = _plot_true(plt, dist_code)
            figname = 'plot-%s-true.png' % (dist_code,)
            full_figname = str(directory / 'plots' / figname)
            fig.savefig(full_figname)
            plt.close(fig)
            print('>', figname)
            alt_tit = '%s, true' % (dist_code,)
            line = '<img src="%s" alt="%s" style="width:45%%"/>\n' % (figname, alt_tit)
            lines.append(line)
            lines.append('</div>\n')
            lines.append('<footer />\n')
    lines.append(_HTML_END)
    with open(str( directory / 'plots' / 'plots.html'), 'w') as fh:
        fh.writelines(lines)
    print('> plots.html')


def _plot_fig(plt, sns, dist_code, num_obvs, wave_name, df):
    fig, (ax1, ax2) = plt.subplots(2, 1, sharey=False, sharex=True, figsize=(4,5.5))
    df1 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs) & (df.wave_name == wave_name)]
    sns.lineplot(x='j', y='b_hat_j', hue='bestj_algo', data=df1, ax=ax1, ci='sd', err_style='band', estimator='mean')
    ax1.set_title('%s, N=%d, %s\n$\hat{\mathcal{B}}_j$ ' % (dist_code, num_obvs, wave_name))
    #ax1.legend_.set_title('Algorithm')
    sns.lineplot(x='j', y='hd', hue='bestj_algo', data=df1, ax=ax2, ci='sd', err_style='band', estimator='mean')
    ax2.set_title('$HD^2$')
    #ax1.legend_.set_title('Algorithm')
    df2 = df[(df.dist_code == dist_code) & (df.num_obvs == num_obvs) & (df.method == 'KDE')]
    ax2.plot(range(7), (df2.hd.mean(),) * 7, 'k:')
    ax2.fill_between(range(7), (df2.hd.mean()-2*df2.hd.std(),) * 7, (df2.hd.mean()+2*df2.hd.std(),) * 7, alpha=0.3, color='k')
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
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap='BuGn')
    ax.set_title(dist.code)
    ax.set_zlim(0, 1.1 * max_v)
    return fig


def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)



def _init_():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
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
