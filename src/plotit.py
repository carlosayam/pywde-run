import atexit
import csv
import itertools as itt
import sys
from pathlib import Path
from datetime import datetime


import click
import pandas as pd

from dist_codes import dist_from_code
from common import *
from mpl_toolkits.mplot3d import Axes3D
from pywde.square_root_estimator import WaveletDensityEstimator
from pywde.spwde import SPWDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate
from plotlib import (plot_dist, do_plot_kde, do_plot_wde, do_kde_contour, do_wde_contour,
                     do_dist_contour, plot_energy, plot_trace, do_plot_pdf, do_pdf_contour)


def read_data(fname):
    with open(fname, 'rt') as fh:
        reader = csv.reader(fh, delimiter='\t', quoting=csv.QUOTE_NONNUMERIC)
        data = []
        for row in reader:
            data.append(row)
        return np.array(data)



@click.group()
def main():
    pass


@main.command()
@click.argument('dist_code', metavar="DIST_CODE")
@click.argument('num_obvs', type=int, required=False, default=0)
@click.argument('sample_no', type=int, required=False, default=0)
@click.option('--contour', is_flag=True)
def plot_true(dist_code, **kwargs):
    dist = dist_from_code(dist_code)
    name = fname('true', dist_code)
    if kwargs['num_obvs']:
        source = sample_name(dist_code, kwargs['num_obvs'], kwargs['sample_no'])
        data = read_data(source)
    else:
        data = None
    if kwargs['contour']:
        do_dist_contour(name, dist, data)
    else:
        plot_dist(name, dist)


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.option('--contour', is_flag=True)
def plot_kde(dist_code, num_obvs, sample_no, **kwargs):
    """
    Plots KDE for sample
    :param dist_code:
    :param num_obvs:
    :param sample_no:
    :return:
    """
    dist = dist_from_code(dist_code)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml')  ## cv_ml
    png_file = png_name(dist_code, num_obvs, sample_no, 'kde')
    if kwargs['contour']:
        do_kde_contour(kde, png_file, dist, None)
    else:
        do_plot_kde(kde, png_file, dist, None)


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.option('--k', type=int)
@click.option('--j0', type=int, default=0)
@click.option('--delta-j', type=int)
@click.option('--kind', default='full', type=click.Choice(['full', 'cv', 'iter']))
@click.option('--loss', type=click.Choice(WaveletDensityEstimator.LOSSES))
@click.option('--ordering', type=click.Choice(WaveletDensityEstimator.ORDERINGS))
@click.option('--multi', is_flag=True)
@click.option('--contour', is_flag=True)
def plot_wde(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE for given k, j0 and delta-j for all possible options
    """
    dist = dist_from_code(dist_code)
    kind = kwargs['kind']
    loss = ordering = is_single = None
    if kind == 'full':
        what = 'wde_all'
        if 'loss' in kwargs and kwargs['loss']:
            raise ValueError('loss only for cv kind')
        if 'ordering' in kwargs and kwargs['ordering']:
            raise ValueError('ordering only for cv kind')
        delta_j = kwargs['delta_j']
    elif kind == 'cv':
        what = 'wde_cv'
        loss = kwargs['loss']
        ordering = kwargs['ordering']
        is_single = not kwargs['multi']
        if not loss:
            raise click.BadOptionUsage('loss', 'For cv, must specify loss')
        if not ordering:
            raise click.BadOptionUsage('ordering', 'For cv, must specify ordering')
        delta_j = kwargs['delta_j']
    else: # kind == 'iter'
        what = 'wde_iter'
        if 'loss' in kwargs and kwargs['loss']:
            raise click.BadOptionUsage('loss', 'For iter, must not specify ordering')
        if 'ordering' in kwargs and kwargs['ordering']:
            raise click.BadOptionUsage('ordering', 'For iter, must not specify ordering')
        if 'delta_j' not in kwargs or kwargs['delta_j'] is None or kwargs['delta_j'] <= 0:
            raise click.BadOptionUsage('delta_j', 'For iter, delta_j is specified')
        delta_j = kwargs['delta_j']
    k = kwargs['k']
    what = what + ('.k_%d' % k)
    j0 = kwargs['j0']
    what = what + ('.j0_%d' % j0)
    what = what + ('.delta_j_%d' % delta_j)
    if kind == 'cv':
        what += '.%s_%s_%s' % (kind, loss, ordering)
    what = wave_name + '-' + what
    png_file = png_name(dist_code, num_obvs, sample_no, what)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    wde = WaveletDensityEstimator(((wave_name, j0), (wave_name, j0)), k=k, delta_j=delta_j)
    if kind == 'full':
        wde.fit(data)
    elif kind == 'cv':
        wde.cvfit(data, loss, ordering, is_single)
    else: # kind == iter
        wde.iterfit(data)
        # import code
        # _env_ = locals().copy()
        # _env_['plt'] = plt
        # code.interact('** wde, plt', local=_env_)
    if hasattr(wde, 'threshold'):
        plot_energy(wde, str(png_file).replace('.png', '-energy.png'))
    if hasattr(wde, 'trace_v'):
        plot_trace(wde, str(png_file).replace('.png', '-trace.png'))
    if kwargs['contour']:
        do_wde_contour(wde, png_file, dist)
    else:
        do_plot_wde(wde, png_file, dist, 'view')



@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.argument('mode', type=click.Choice([SPWDE.TARGET_DIFF, SPWDE.TARGET_NORMED]))
@click.option('--k', type=int, default=1)
@click.option('--j0', type=int, default=0)
@click.option('--contour', is_flag=True)
def plot_best_j(dist_code, num_obvs, sample_no, wave_name, mode, **kwargs):
    """
    Calculates WDE for given k, j0 and delta-j for all possible options
    """
    dist = dist_from_code(dist_code)
    k = kwargs['k']
    what = 'best_j' + ('.k_%d' % k)
    j0 = kwargs['j0']
    what = what + ('.j0_%d' % j0)
    what = wave_name + '-' + what
    png_file = png_name(dist_code, num_obvs, sample_no, what)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    # wde = WaveletDensityEstimator(((wave_name, j0), (wave_name, j0)), k=k)
    # wde.best_j(data)
    spwde = SPWDE(((wave_name, j0), (wave_name, j0)), k=k)
    spwde.best_j(data, mode=mode)
    # return # << !!!
    for data_for_j in spwde.best_j_data:
        j, is_best, b_hat_j, pdf, elapsed = data_for_j
        print(j, is_best, '\tB hat =', b_hat_j, 't =', elapsed)
        if is_best:
            pdf.name = pdf.name + (' (best %s)' % mode)
        if kwargs['contour']:
            do_pdf_contour(pdf, 'test2-%d.png' % j, dist)
        else:
            do_plot_pdf(pdf, 'test2-%d.png' % j, dist, 'view')


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.argument('delta_j', type=int)
@click.argument('target', type=click.Choice([SPWDE.TARGET_NORMED, SPWDE.TARGET_DIFF]))
@click.argument('mode', type=click.Choice([SPWDE.TH_CLASSIC, SPWDE.TH_ADJUSTED, SPWDE.TH_EMP_STD, SPWDE.TH_EMP_STD_ADJ]))
@click.option('--k', type=int, default=1)
@click.option('--j0', default=None)
@click.option('--contour', is_flag=True)
def plot_best_c(dist_code, num_obvs, sample_no, wave_name, delta_j, target, mode, **kwargs):
    """
    Calculates WDE for best threshold options
    """
    dist = dist_from_code(dist_code)
    k = kwargs['k']
    what = 'best_j' + ('.k_%d' % k)
    j0 = kwargs['j0']
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    if j0 is None:
        spwde = SPWDE(((wave_name, 0),(wave_name, 0)) , k=1)
        best_j = spwde.best_j(data, mode=target, stop_on_max=True)
        j0 = best_j - delta_j
        print('Using J_0 =', j0, 'with +%d' % delta_j)
    else:
        j0 = int(j0)
        print('Params J_0 =', j0, 'with +%d' % delta_j)
    what = what + ('.j0_%d' % j0)
    what = wave_name + '-' + what
    png_file = png_name(dist_code, num_obvs, sample_no, what)
    spwde = SPWDE(((wave_name, j0), (wave_name, j0)), k=k)
    spwde.best_c(data, delta_j, target, th_mode=mode)
    # return # << !!!
    pdf = spwde.best_c_found[0]
    cc = spwde.best_c_found[1]
    if spwde.best_c_data:
        xy = np.array(spwde.best_c_data)
        import seaborn as sns
        import matplotlib.pyplot as plt
        if pdf.subtitle:
            subtit = pdf.subtitle
        else:
            subtit = ''
        title = ("%s - %s\n" r"$J_0 = %d$, $\Delta J = %d$, %s" % (dist_code, wave_name, j0, delta_j, subtit))
        ax = sns.lineplot(xy[:,0], xy[:,1])
        ax.set_title(title)
        mode_ix = 1 if target == SPWDE.TARGET_NORMED else 2
        ax.set(xlabel="$C$", ylabel=r"$HD_%d(C)$" % mode_ix, )
        ax.fill_between(xy[:,0], xy[:,1]-3*xy[:,2], xy[:,1]+3*xy[:,2], alpha=0.3)
        plt.show()
    if kwargs['contour']:
        do_pdf_contour(pdf, 'test3-%s.png' % cc[0], dist)
    else:
        do_plot_pdf(pdf, 'test3-%s.png' % cc[0], dist, 'view')


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.argument('delta_j', type=int)
@click.argument('target', type=click.Choice([SPWDE.TARGET_DIFF, SPWDE.TARGET_NORMED]))
@click.option('--k', type=int, default=1)
@click.option('--j0', default=None)
@click.option('--contour', is_flag=True)
def plot_greedy(dist_code, num_obvs, sample_no, wave_name, delta_j, target, **kwargs):
    """
    Calculate best WDE using greedy optimisation
    """
    dist = dist_from_code(dist_code)
    k = kwargs['k']
    what = 'best_j' + ('.k_%d' % k)
    j0 = kwargs['j0']
    png_file = png_name(dist_code, num_obvs, sample_no, what)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    if j0 is None:
        spwde = SPWDE(((wave_name, 0),(wave_name, 0)) , k=1)
        best_j = spwde.best_j(data, mode=target, stop_on_max=True)
        j0 = best_j - delta_j
        print('Using J_0 =', j0, 'with +%d' % delta_j)
    else:
        j0 = int(j0)
        print('Params J_0 =', j0, 'with +%d' % delta_j)
    what = what + ('.j0_%d' % j0)
    what = wave_name + '-' + what
    # wde = WaveletDensityEstimator(((wave_name, j0), (wave_name, j0)), k=k)
    # wde.best_j(data)
    spwde = SPWDE(((wave_name, j0), (wave_name, j0)), k=k)
    spwde.best_greedy(data, delta_j, j0, target)
    # return # << !!!
    if spwde.best_c_data:
        xy = np.array(spwde.best_c_data)
        import seaborn as sns
        import matplotlib.pyplot as plt
        title = ("%s - %s\n" r"$J_0 = %d$, $\Delta J = %d$,"
                 r"$\left| \beta_{j,q,z} \right| \geq C \sqrt{j + 1}$") % (dist_code, wave_name, j0, delta_j)
        ax = sns.lineplot(xy[:,0], xy[:,1])
        ax.set_title(title)
        mode_ix = 1 if target == SPWDE.TARGET_NORMED else 2
        ax.set(xlabel="$C$", ylabel=r"$HD_%d(C)$" % mode_ix, )
        plt.show()
    pdf = spwde.best_c_found[0]
    cc = spwde.best_c_found[1]
    if kwargs['contour']:
        do_pdf_contour(pdf, 'test3-%s.png' % cc, dist)
    else:
        do_plot_pdf(pdf, 'test3-%s.png' % cc, dist, 'view')

@main.command()
@click.argument('dist_name', metavar="DIST_CODE")
@click.argument('wave_name', metavar="WAVE_CODE")
@click.argument('num', type=int)
@click.argument('ix', type=int, nargs=2)
@click.argument('delta_js', nargs=-1, type=int)
# @click.option('--loss', help='Loss function', default=WaveletDensityEstimator.NEW_LOSS)
# @click.option('--ordering', help='Ordering method', default=WaveletDensityEstimator.T_ORD)
# @click.option('--k', type=int, default=1)
def run_with(dist_name, wave_name, num, ix, delta_js):
    dest = fname('results', dist_name, ext='-%02d.tab' % ix[0])
    with open(dest, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        i0, numi = ix
        for row in calc_with(dist_name, wave_name, num, i0, numi, delta_js):
            writer.writerow(row)
            fh.flush()

def calc_with(dist_name, wave_name, num, i0, numi, delta_js):
    dist = dist_from_code(dist_name)
    ## max_v = calc_maxv(dist)
    yield ['dist', 'wave', 'num', 'sample_num', 'method', 'k', 'delta_j', 'loss', 'ordering', 'HD']
    for ix in range(numi):
        data = dist.rvs(num)
        i = i0 + ix
        # save_data(data, fname('data', dist_name, num=num, wave_name=wave_name, ext='(%02d).csv' % i))
        for k, delta_j in itt.product([1,2], delta_js):
            wde = WaveletDensityEstimator(((wave_name, 0),(wave_name, 0)) , k=k, delta_j=delta_j)
            print('WDE', 'k=%d' % k, 'delta_j=%d' % delta_j)
            wde.fit(data)
            hd, corr_factor = hellinger_distance(dist, wde)
            yield [dist_name, wave_name, num, i, 'WDE', k, delta_j, '', '', hd]
            ## plot_wde(wde, fname('orig', dist_name, num, wave_name, delta_j), dist, 1.1 * max_v)
            for loss, ordering in WaveletDensityEstimator.valid_options():
                print('WDE', 'k=%d' % k, 'delta_j=%d' % delta_j, 'Loss', loss, 'Ord', ordering)
                wde.cvfit(data, loss, ordering)
                hd, corr_factor = hellinger_distance(dist, wde)
                yield [dist_name, wave_name, num, i, 'WDE_CV', k, delta_j, loss, ordering, hd]
                # what = 'new_%s.%s' % (loss, ordering)
                # plot_wde(wde, fname(what, dist_name, num, wave_name, delta_j), dist, 1.1 * max_v)
                # what = 'energy_%s.%s' % (loss, ordering)
                # plot_energy(wde, fname(what, dist_name, num, wave_name, delta_j))
        #print('Estimating KDE all data')
        # kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml') ## cv_ml
        # hd, corr_factor = hellinger_distance(dist, kde)
        # yield [dist_name, wave_name, num, i, 'KDE', '', '', '', '', hd]
        # plot_kde(kde, fname('kde_cv', dist_name, num), dist, 1.1 * max_v)


#
# TODO - check consitency bby increasing sample size !!!
#
# - chicken, cai - look
# - why 3 - double check formula and algorithm
# - then send again to Spiro & Gery
# - find better distributions to showcase
#

#dist = dist_from_code('tri1')
#print(dist.rvs(10))
#plot_dist('tri1.png', dist)
# dist = dist_from_code('pir1')
# data = dist.rvs(1024)
# plt.figure()
# plt.scatter(data[:,0], data[:,1])
# plt.show()


if __name__ == "__main__":
    def wtime(t0):
        secs = (datetime.now() - t0)
        click.echo("[walltime %s] python " % str(secs), err=True, nl=False)
        click.echo(" ".join(sys.argv), err=True)
    atexit.register(wtime, datetime.now())

    main()
