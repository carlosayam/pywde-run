import os
import pathlib
from itertools import zip_longest

import numpy as np

from statsmodels.nonparametric.kernel_density import KDEMultivariate


ROOT_DIR = pathlib.Path('RESP')
NUM_SAMPLES = 100


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def fname(what, dist_name, num=None, wave_name=None, delta_j=None, ext='.png'):
    strn = '%04d' % num if num is not None else None
    strd = '%d' % delta_j if delta_j is not None else None
    strs = [what, dist_name, strn, wave_name, strd]
    strs = [v for v in strs if v]
    return '%s/%s%s' % (ROOT_DIR, '-'.join(strs), ext)


def ensure_dir(a_path):
    if not a_path.exists():
        os.makedirs(a_path.absolute())
    return a_path


def base_dir(dist_name, num_obvs):
    return ensure_dir(ROOT_DIR / dist_name / ('%04d' % num_obvs))


def sample_name(dist_name, num_obvs, sample_no):
    filename = 'sample-%03d.tab' % sample_no
    path = ensure_dir(base_dir(dist_name, num_obvs) / 'samples') / filename
    return path.absolute()


def results_name(dist_name, num_obvs, sample_no, what):
    if type(sample_no) == int:
        filename = '%s-%03d.tab' % (what, sample_no)
    else: # type(sample_no) == tuple
        min_n, max_n = sample_no
        filename = '%s-%03d_%03d.tab' % (what, min_n, max_n)
    path = ensure_dir(base_dir(dist_name, num_obvs) / 'results') / filename
    return path.absolute()


def png_name(dist_name, num_obvs, sample_no, what):
    filename = '%s-%03d.png' % (what, sample_no)
    path = ensure_dir(base_dir(dist_name, num_obvs) / 'plots') / filename
    return path.absolute()


def grid_as_vector(n):
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    return np.meshgrid(x, y)


def hellinger_distance(dist, dist_est):
    grid = grid_as_vector(128)
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
    err = (diff * diff).mean() / 2.0
    return err, corr_factor


def hellinger_distance_pdf(dist, pdf):
    grid = grid_as_vector(128)
    pdf_vals = dist.pdf(grid)
    #print('DIST:', pdf_vals.mean())
    pdf_vals = pdf_vals / pdf_vals.mean()
    pdf_vals = np.sqrt(pdf_vals)

    grid = np.array(grid)
    pred_vals = pdf(grid.T.reshape(-1, 2)).reshape((128, 128)).T
    #print('WDE:', pred_vals.mean())
    corr_factor = pred_vals.mean()
    # print('corr factor = %g' % corr_factor)
    pred_vals = pred_vals / corr_factor
    pred_vals = np.sqrt(pred_vals)
    diff = pdf_vals - pred_vals
    err = (diff * diff).mean() / 2.0
    return err, corr_factor


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




def calc_maxv(dist):
    grid_n = 70
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    print('sum=', zz.mean())
    zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    print('int =', zz_sum)
    return (zz / zz_sum).max()
