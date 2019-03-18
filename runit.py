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

from dist_codes import dist_from_code
from common import *
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
    # print('corr factor = %g' % corr_factor)
    pred_vals = pred_vals / corr_factor
    pred_vals = np.sqrt(pred_vals)
    diff = pdf_vals - pred_vals
    err = (diff * diff).mean()  ## !!! /2
    return err, corr_factor




Result = namedtuple('Result', [
    'dist_code',
    'num_obvs',
    'sample_no',
    'what', # KDE, WDE
    'wave_name',
    'k',
    'delta_j',
    'loss',
    'ordering',
    'is_single',
    'hd',
    'params',
    'elapsed_time'
])
Result.__new__.__defaults__ = ('', 0, 0, '', 0, 0, '', '', True, 0.0, 0, 0.0)


def result_kde(dist_code, num_obvs, sample_no, hd, elapsed):
    return Result(
        dist_code=dist_code,
        num_obvs=num_obvs,
        sample_no=sample_no,
        what='KDE',
        hd=hd,
        elapsed_time=elapsed
    )


def result_wde_classic(dist_code, num_obvs, sample_no, wave_name, k, delta_j, params, hd, elapsed):
    return Result(
        dist_code=dist_code,
        num_obvs=num_obvs,
        sample_no=sample_no,
        what='WDE_FULL',
        wave_name=wave_name,
        k=k,
        delta_j=delta_j,
        params=params,
        hd=hd,
        elapsed_time=elapsed
    )

def result_wde_cv(dist_code, num_obvs, sample_no, wave_name, k, delta_j, params, loss, ordering, is_single, hd, elapsed):
    return Result(
        dist_code=dist_code,
        num_obvs=num_obvs,
        sample_no=sample_no,
        what='WDE_CV',
        wave_name=wave_name,
        k=k,
        delta_j=delta_j,
        params=params,
        loss=loss,
        ordering=ordering,
        is_single=is_single,
        hd=hd,
        elapsed_time=elapsed
    )



def save_data(data, fname):
    with open(fname, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for row in data:
            writer.writerow(row)


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
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
def gen_samples(dist_code, num_obvs):
    """
    Generates 50 samples for given distribution and number of observations
    in each sample.

    1000-6000 <~ 5 mins to generate 50 samples
    """
    dist = dist_from_code(dist_code)
    for ix in range(1, NUM_SAMPLES + 1):
        dest = sample_name(dist_code, num_obvs, ix)
        data = dist.rvs(num_obvs)
        save_data(data, dest)
        print('.', end='')
        sys.stdout.flush()
    print()


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int, default='0')
def calc_kde(dist_code, num_obvs, sample_no):
    """
    Calculates KDE.

    1000, 2000 <~ 20 secs each sample
    3000, 4000 <~ 90 secs each sample
    5000, 6000 <~ 180 secs each sample
    """
    def gen():
        for ix in sample_range:
            source = sample_name(dist_code, num_obvs, ix)
            data = read_data(source)
            assert data.shape[0] == num_obvs
            t0 = datetime.now()
            kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml')  ## cv_ml
            elapsed = (datetime.now() - t0).total_seconds()
            hd, corr_factor = hellinger_distance(dist, kde)
            yield result_kde(dist_code, num_obvs, ix, hd, elapsed)

    dist = dist_from_code(dist_code)
    if sample_no == 0:
        sample_range = range(1, NUM_SAMPLES + 1)
        result_file = results_name(dist_code, num_obvs, 0, 'kde')
    else:
        sample_range = [sample_no]
        result_file = results_name(dist_code, num_obvs, sample_no, 'kde')
    with open(result_file, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for result in gen():
            writer.writerow(list(result))


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int, default='0')
@click.argument('wave_name')
@click.option('--k', type=int)
@click.option('--j0', type=int)
@click.option('--delta-j', type=int)
@click.option('--multi', is_flag=True)
def calc_wde(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE for given k, j0 and delta-j for all possible options
    """
    def gen():
        for ix, k, j0, delta_j in itt.product(sample_range, k_range, j0_range, delta_j_range):
            source = sample_name(dist_code, num_obvs, ix)
            data = read_data(source)
            assert data.shape[0] == num_obvs
            t0 = datetime.now()
            wde = WaveletDensityEstimator(((wave_name, 0),(wave_name, 0)) , k=k, delta_j=delta_j)
            wde.fit(data)
            elapsed = (datetime.now() - t0).total_seconds()
            hd, corr_factor = hellinger_distance(dist, wde)
            params = wde.pdf.nparams
            yield result_wde_classic(dist_code, num_obvs, ix, wave_name, k, delta_j, params, hd, elapsed)
            for loss, ordering, is_single in WaveletDensityEstimator.valid_options(single):
                t0 = datetime.now()
                wde.cvfit(data, loss, ordering, is_single=is_single)
                elapsed = (datetime.now() - t0).total_seconds()
                hd, corr_factor = hellinger_distance(dist, wde)
                params = wde.pdf.nparams
                yield result_wde_cv(dist_code, num_obvs, ix, wave_name, k, delta_j, params, loss, ordering, is_single, hd, elapsed)

    dist = dist_from_code(dist_code)
    what = wave_name
    single = not kwargs['multi']
    what = what + ('.single_%s' % single)
    if 'k' not in kwargs:
        k_range = [1, 2]
    else:
        k = kwargs['k']
        k_range = [k]
        what = what + ('.k_%d' % k)
    if 'j0' not in kwargs:
        j0_range = [0]
    else:
        j0 = kwargs['j0']
        j0_range = [j0]
        what = what + ('.j0_%d' % j0)
    if 'delta_j' not in kwargs:
        delta_j_range = range(0, 6)
    else:
        delta_j = kwargs['delta_j']
        delta_j_range = [delta_j]
        what = what + ('.delta_j_%d' % delta_j)
    what = 'wde-' + what
    if sample_no == 0:
        sample_range = range(1, NUM_SAMPLES + 1)
        result_file = results_name(dist_code, num_obvs, 0, what)
    else:
        sample_range = [sample_no]
        result_file = results_name(dist_code, num_obvs, sample_no, what)
    with open(result_file, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for result in gen():
            writer.writerow(list(result))


#
# TODO - check consistency bby increasing sample size !!!
#
# - chicken, cai - look
# - why 3 - double check formula and algorithm
# - then send again to Spiro & Gery
# - find better distributions to showcase
#

if __name__ == "__main__":
    click.echo("RUNNING python " + " ".join(sys.argv), err=True)
    def wtime(t0):
        secs = (datetime.now() - t0)
        click.echo("[walltime %s]" % str(secs), err=True)
    atexit.register(wtime, datetime.now())
    main()
