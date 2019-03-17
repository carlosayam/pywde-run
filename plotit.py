import atexit
import csv
import itertools as itt
import os
import pathlib
import sys
from collections import namedtuple
from datetime import datetime

import click
import numpy as np
import scipy.integrate as integrate
#from scipy.spatial import Delaunay
from scipy.interpolate import LinearNDInterpolator

import matplotlib.pyplot as plt
from dist_codes import dist_from_code
from common import *
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from pywde.square_root_estimator import WaveletDensityEstimator
from statsmodels.nonparametric.kernel_density import KDEMultivariate


# a 2d grid of [0,1], n x n, rendered as array of (n^2,2)
def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)


def calc_maxv(dist):
    grid_n = 70
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    print('sum=', zz.mean())
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    print('int =', zz_sum)
    return (zz / zz_sum).max()

def plot_dist(fname, dist):
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    print('sum=', zz.mean())
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    print('int =', zz_sum)
    max_v = (zz / zz_sum).max()
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz / zz_sum, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(dist.code)
    ax.set_zlim(0, 1.1 * max_v)
    plt.show()
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def do_plot_wde(wde, fname, dist):
    print('Plotting %s' % fname)
    zlim = calc_maxv(dist)
    hd, corr_factor = hellinger_distance(dist, wde)
    print(wde.name, 'HD=', hd)
    ##return
    grid_n = 40 ## 70
    xx, yy = grid_as_vector(grid_n)
    zz = wde.pdf((xx, yy)) / corr_factor
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(wde.name + ('\nHD = %g' % hd), wrap=True)
    ax.set_zlim(0, zlim)
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def plot_energy(wde, fname):
    fig = plt.figure()
    xx = wde.vals[:,0]
    yy = wde.vals[:,1]
    plt.plot(xx, yy)
    plt.axvline(x=wde.threshold, c='r')
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('C')
    plt.ylabel('$B_C$')
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)
    fname = fname.replace('energy', 'energy2')
    fig = plt.figure()
    xx = range(wde.vals.shape[0])
    yy = wde.vals[:,1]
    plt.plot(xx, yy)
    plt.axvline(x=wde.pos_k, c='r')
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('$i$')
    plt.ylabel('$B_i$')
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def do_plot_kde(kde, fname, dist, zlim):
    print('Plotting %s' % fname)
    hd, corr_factor = hellinger_distance(dist, kde)
    print('kde HD=', hd)
    grid_n = 40 ## 70
    xx, yy = grid_as_vector(grid_n)
    grid2 = np.array((xx.flatten(), yy.flatten())).T
    vals = kde.pdf(grid2)
    zz = vals.reshape(xx.shape[0], yy.shape[0])

    ##zz_sum = zz.sum() / grid_n / grid_n  # not always near 1

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(xx, yy, zz, edgecolors='k', linewidth=0.5, cmap=cm.get_cmap('BuGn'))
    ax.set_title(('KDE bw=%s' % str(kde.bw)) + ('\nHD = %g' % hd))
    ax.set_zlim(0, zlim)
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def hellinger_distance_wip(dist, dist_est):
    # import code
    # code.interact(local=locals())
    def ferr(x, y):
        args = np.array([(x, y)])
        pdf_vals = np.sqrt(dist.pdf(args))
        est_vals = np.sqrt(dist_est.pdf(args))/corr_factor
        return ((pdf_vals - est_vals) ** 2)[0]
    def pdf(x, y):
        est_vals = dist_est.pdf(np.array([(x, y)]))
        return est_vals[0]
    corr_factor = integrate.dblquad(pdf, 0.0, 1.0, lambda x:0.0, lambda x:1.0)
    err = integrate.dblquad(ferr, 0.0, 1.0, lambda x: 0.0, lambda x: 1.0)
    return err, corr_factor


def hellinger_distance(dist, dist_est):
    grid = grid_as_vector(256)
    pdf_vals = dist.pdf(grid)
    #print('DIST:', pdf_vals.mean())
    pdf_vals = pdf_vals / pdf_vals.mean()
    pdf_vals = np.sqrt(pdf_vals)
    if isinstance(dist_est, KDEMultivariate):
        X, Y = grid
        grid2 = np.array((X.flatten(), Y.flatten())).T
        vals = dist_est.pdf(grid2)
        pred_vals = vals.reshape(X.shape[0], Y.shape[0])
    else:
        pred_vals = dist_est.pdf(grid)
    #print('WDE:', pred_vals.mean())
    corr_factor = pred_vals.mean()
    print('corr factor = %g' % corr_factor)
    pred_vals = pred_vals / corr_factor
    pred_vals = np.sqrt(pred_vals)
    diff = pdf_vals - pred_vals
    err = (diff * diff).mean()  ## !!! /2
    return err, corr_factor



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
@click.argument('dist_name', metavar="DIST_CODE")
def plot_true(dist_name):
    dist = dist_from_code(dist_name)
    plot_dist(fname('true', dist_name), dist)



@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
def plot_kde(dist_code, num_obvs, sample_no):
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
    do_plot_kde(kde, png_file, dist, None)


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.option('--k', type=int)
@click.option('--j0', type=int, default=0)
@click.option('--delta-j', type=int)
@click.option('--kind', default='full', type=click.Choice(['all', 'cv1', 'cvx']))
@click.option('--loss', type=click.Choice(WaveletDensityEstimator.LOSSES))
@click.option('--ordering', type=click.Choice(WaveletDensityEstimator.ORDERINGS))
def plot_wde(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE for given k, j0 and delta-j for all possible options
    """
    dist = dist_from_code(dist_code)
    kind = kwargs['kind']
    loss = ordering = is_single = None
    if kind == 'all':
        what = 'wde_all'
        if 'loss' in kwargs and kwargs['loss']:
            raise ValueError('loss only for cv kind')
        if 'ordering' in kwargs and kwargs['ordering']:
            raise ValueError('loss only for cv kind')
    elif kind == 'cv1':
        what = 'wde_cv1'
        loss = kwargs['loss']
        ordering = kwargs['ordering']
        is_single = True
        if not loss:
            raise click.BadOptionUsage('loss', 'For cv1, must specify loss')
        if not ordering:
            raise click.BadOptionUsage('ordering', 'For cv1, must specify ordering')
    elif kind == 'cvx':
        what = 'wde_cvx'
        loss = kwargs['loss']
        ordering = kwargs['ordering']
        is_single = False
        if not loss:
            raise click.BadOptionUsage('loss', 'For cv1, must specify loss')
        if not ordering:
            raise click.BadOptionUsage('ordering', 'For cv1, must specify ordering')
    k = kwargs['k']
    what = what + ('.k_%d' % k)
    j0 = kwargs['j0']
    what = what + ('.j0_%d' % j0)
    delta_j = kwargs['delta_j']
    what = what + ('.delta_j_%d' % delta_j)
    if kind[:2] == 'cv':
        what += '.%s_%s_%s' % (kind, loss, ordering)
    what = wave_name + '-' + what
    png_file = png_name(dist_code, num_obvs, sample_no, what)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    wde = WaveletDensityEstimator(((wave_name, 0), (wave_name, 0)), k=k, delta_j=delta_j)
    wde.fit(data)
    if kind != 'all':
        wde.cvfit(data, loss, ordering, is_single)
    do_plot_wde(wde, png_file, dist)


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
