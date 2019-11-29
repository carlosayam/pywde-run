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
from common import NUM_SAMPLES, sample_name, results_name, grid_as_vector, calc_maxv, hellinger_distance, hellinger_distance_pdf
from pywde.square_root_estimator import WaveletDensityEstimator
from pywde.spwde import SPWDE
from statsmodels.nonparametric.kernel_density import KDEMultivariate


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
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.argument('results', type=click.Path(file_okay=True, dir_okay=False, writable=True))
def bestj_task(dist_code, num_obvs, sample_no, wave_name, results):
    "Run all stuff for 1 task"
    dist = dist_from_code(dist_code)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs

    def _kde():
        t0 = datetime.now()
        kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml')  ## cv_ml
        elapsed = (datetime.now() - t0).total_seconds()
        hd, corr_factor = hellinger_distance(dist, kde)
        return (dist_code, num_obvs, sample_no, 'KDE', '', '', 0, False, 0.0, hd, elapsed)

    def _bestj(kind, mode):
        spwde = SPWDE(((wave_name, 0),(wave_name, 0)) , k=1)
        spwde.best_j(data, mode=mode)
        for data_for_j in spwde.best_j_data:
            j, is_best, b_hat_j, pdf, elapsed  = data_for_j
            hd, corr_factor = hellinger_distance_pdf(dist, pdf)
            yield (dist_code, num_obvs, sample_no, 'WDE', wave_name, kind, j, is_best, b_hat_j, hd, elapsed)

    with open(results, 'a') as fh:
        writer = csv.writer(fh, delimiter='\t')
        row = list(_kde())
        print(row)
        writer.writerow(row)
        for row in _bestj('normed', SPWDE.TARGET_NORMED):
            print(row)
            writer.writerow(row)
        for row in _bestj('diff', SPWDE.TARGET_DIFF):
            print(row)
            writer.writerow(row)

@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('first_sample', type=int)
@click.argument('num_samples', type=int)
@click.argument('results', type=click.Path(file_okay=True, dir_okay=False, writable=True))
def kde(dist_code, num_obvs, first_sample, num_samples, results):
    "Run KDE for samples starting first_sample, up to num_samples"

    def _kde(sample_no, data):
        t0 = datetime.now()
        kde = KDEMultivariate(data, 'c' * data.shape[1], bw='cv_ml')  ## cv_ml
        elapsed = (datetime.now() - t0).total_seconds()
        hd, corr_factor = hellinger_distance(dist, kde)
        return (dist_code, num_obvs, sample_no, 'KDE', '', '', '', 0,
                0, 0, num_obvs, 0.0, hd, elapsed)

    dist = dist_from_code(dist_code)
    with open(results, 'a') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for sample_no in range(num_samples):
            sample_no = first_sample + sample_no
            if sample_no > 100:
                break
            source = sample_name(dist_code, num_obvs, sample_no)
            data = read_data(source)
            assert data.shape[0] == num_obvs
            row = _kde(sample_no, data)
            print(row)
            writer.writerow(row)


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int)
@click.argument('wave_name')
@click.argument('mode', type=click.Choice([SPWDE.TARGET_NORMED, SPWDE.TARGET_DIFF]))
@click.argument('results', type=click.Path(file_okay=True, dir_okay=False, writable=True))
def best_c(dist_code, num_obvs, sample_no, wave_name, mode, results):
    "Run all stuff for 1 task"
    dist = dist_from_code(dist_code)
    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs

    def calc_bestj(a_mode):
        t0 = datetime.now()
        spwde = SPWDE(((wave_name, 0),(wave_name, 0)) , k=1)
        best_j = spwde.best_j(data, mode=a_mode, stop_on_max=True)
        elapsed = (datetime.now() - t0).total_seconds()
        return best_j, elapsed


    def _bestc(best_j_dict):
        best_j, elapsed0 = best_j_dict[mode]
        yield (dist_code, num_obvs, sample_no, 'best_j_part', wave_name, mode, '', best_j,
               0, 0, 0, 0, 0, elapsed0)
        for th_mode, delta_j, excess_j in itt.product(
                (SPWDE.TH_CLASSIC, SPWDE.TH_ADJUSTED, SPWDE.TH_EMP_STD),
                (1, 2, 3),
                (0, 1)
        ):
            the_j = best_j - delta_j
            delta_j = delta_j + excess_j
            t0 = datetime.now()
            spwde = SPWDE(((wave_name, the_j), (wave_name, the_j)), k=1)
            spwde.best_c(data, delta_j, mode, th_mode)
            elapsed = (datetime.now() - t0).total_seconds()
            pdf, best_c_data = spwde.best_c_found
            hd, corr_factor = hellinger_distance_pdf(dist, pdf)
            num_coeffs, b_hat_j = best_c_data[3], best_c_data[1]
            yield (dist_code, num_obvs, sample_no, 'WDE', wave_name, mode, th_mode, best_j,
                   the_j, delta_j, num_coeffs, b_hat_j, hd, elapsed0 + elapsed)

    with open(results, 'a') as fh:
        writer = csv.writer(fh, delimiter='\t')
        best_j_dict = {mode: calc_bestj(mode)}
        for row in _bestc(best_j_dict):
            print(row)
            writer.writerow(row)


@main.command()
@click.argument('directory', type=click.Path(file_okay=False, dir_okay=True))
def exp01_plots(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    from exp01 import do_plot_exp01
    do_plot_exp01(directory)


@main.command()
@click.argument('directory', type=click.Path(file_okay=False, dir_okay=True))
def exp01_compare(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    from exp01 import do_compare_algos
    do_compare_algos(directory)


@main.command()
@click.argument('directory', type=click.Path(file_okay=False, dir_okay=True))
def exp02_plots(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    from exp02 import do_plot_exp02
    do_plot_exp02(directory)


@main.command()
@click.argument('directory', type=click.Path(file_okay=False, dir_okay=True))
def exp02_repl(directory):
    "Reads all *.tab files in [DIRECTORY] and produces corresponding plots in there"
    from exp02 import exp02_repl
    exp02_repl(directory)




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
        result_file = results_name(dist_code, num_obvs, (1, NUM_SAMPLES), 'kde')
    else:
        sample_range = [sample_no]
        result_file = results_name(dist_code, num_obvs, sample_no, 'kde')
    with open(result_file, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for result in gen():
            if len(sample_range) == 1:
                print(result)
            writer.writerow(list(result))


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', default='')
@click.argument('wave_name')
@click.option('--k', type=int, default=1)
@click.option('--j0', type=int, default=0)
@click.option('--delta-j', type=int, default=0)
@click.option('--multi', is_flag=True)
def calc_wde(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE for given k, j0 and delta-j for all possible options
    """
    def gen():
        for ix, k, delta_j in itt.product(sample_range, k_range, delta_j_range):
            source = sample_name(dist_code, num_obvs, ix)
            data = read_data(source)
            assert data.shape[0] == num_obvs
            t0 = datetime.now()
            wde = WaveletDensityEstimator(((wave_name, j0),(wave_name, j0)) , k=k, delta_j=delta_j)
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
        j0 = 0
    else:
        j0 = kwargs['j0']
    what = what + ('.j0_%d' % j0)
    if 'delta_j' not in kwargs:
        delta_j_range = range(0, 6)
    else:
        delta_j = kwargs['delta_j']
        delta_j_range = [delta_j]
        what = what + ('.delta_j_%d' % delta_j)
    what = 'wde-' + what
    if sample_no == '':
        sample_range = range(1, NUM_SAMPLES + 1)
        result_file = results_name(dist_code, num_obvs, (1,NUM_SAMPLES), what)
    elif ':' in sample_no:
        min_n, max_n = map(int, sample_no.split(':',2))
        sample_range = range(min_n, max_n + 1)
        result_file = results_name(dist_code, num_obvs, (min_n, max_n), what)
    else:
        sample_range = [sample_no]
        result_file = results_name(dist_code, num_obvs, sample_no, what)
    with open(result_file, 'wt') as fh:
        writer = csv.writer(fh, delimiter='\t')
        for result in gen():
            writer.writerow(list(result))


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int, default='0')
@click.argument('wave_name')
@click.option('--k', type=int, default=1)
@click.option('--j0', type=int, default=0)
@click.option('--delta-j', type=int, default=0)
@click.option('--loss', type=click.Choice(WaveletDensityEstimator.LOSSES), default=None)
@click.option('--ordering', type=click.Choice(WaveletDensityEstimator.ORDERINGS.keys()), default=None)
@click.option('--multi', is_flag=True)
@click.option('--plot', type=click.Choice(['save', 'view']), default=None)
def calc_wde_cv(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE CV for given params
    """
    dist = dist_from_code(dist_code)
    #what = wave_name
    single = not kwargs['multi']
    #what = what + ('.single_%s' % single)
    k = kwargs['k']
    #what = what + ('.k_%d' % k)
    j0 = kwargs['j0']
    #what = what + ('.j0_%d' % j0)
    delta_j = kwargs['delta_j']
    #what = what + ('.delta_j_%d' % delta_j)
    #what = 'wde-' + what
    loss = kwargs['loss']
    ordering = kwargs['ordering']
    plot = kwargs['plot']

    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    t0 = datetime.now()
    wde = WaveletDensityEstimator(((wave_name, j0), (wave_name, j0)), k=k, delta_j=delta_j)
    wde.fit(data)
    wde.cvfit(data, loss, ordering, is_single=single)
    if plot:
        import plotlib
        fname = 'test.png'
        plotlib.do_plot_wde(wde, fname, dist, plot)
    else:
        elapsed = (datetime.now() - t0).total_seconds()
        hd, corr_factor = hellinger_distance(dist, wde)
        params = wde.pdf.nparams
        print('RESULT', dist_code, num_obvs, sample_no, wave_name, k, delta_j, params, loss, ordering, single, hd, elapsed)


@main.command()
@click.argument('dist_code')
@click.argument('num_obvs', type=int)
@click.argument('sample_no', type=int, default='0')
@click.argument('wave_name')
@click.option('--k', type=int, default=1)
@click.option('--j0', type=int, default=0)
@click.option('--delta-j', type=int, default=0)
@click.option('--plot', type=click.Choice(['save', 'view']), default=None)
def calc_wde1(dist_code, num_obvs, sample_no, wave_name, **kwargs):
    """
    Calculates WDE CV for given params
    """
    dist = dist_from_code(dist_code)
    #what = what + ('.single_%s' % single)
    k = kwargs['k']
    #what = what + ('.k_%d' % k)
    j0 = kwargs['j0']
    #what = what + ('.j0_%d' % j0)
    delta_j = kwargs['delta_j']
    # plot
    plot = kwargs['plot']

    source = sample_name(dist_code, num_obvs, sample_no)
    data = read_data(source)
    assert data.shape[0] == num_obvs
    t0 = datetime.now()
    wde = WaveletDensityEstimator(((wave_name, j0), (wave_name, j0)), k=k, delta_j=delta_j)
    wde.fit(data)
    if plot:
        import plotlib
        fname = 'test.png'
        plotlib.do_plot_wde(wde, fname, dist, plot)
    else:
        elapsed = (datetime.now() - t0).total_seconds()
        hd, corr_factor = hellinger_distance(dist, wde)
        params = wde.pdf.nparams
        print('RESULT', dist_code, num_obvs, sample_no, wave_name, k, delta_j, params, None, None, None, hd, elapsed)



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
