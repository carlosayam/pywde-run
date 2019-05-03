import atexit
import csv
import itertools as itt
import os
import pathlib
import sys
from collections import namedtuple
from datetime import datetime
import random

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
    if fname is not None:
        plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)


def do_dist_contour(fname, dist, data=None):
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    zz = dist.pdf((xx, yy))
    fig = plt.figure()
    max_z = zz.max()
    print('max_z', max_z)
    alpha = 0.7 if data is None else 0.5
    cs = plt.contour(xx, yy, zz, alpha=alpha, levels=np.linspace(0, max_z, 12))
    if data is not None:
        plt.scatter(data[:,0], data[:,1], alpha=0.7, s=3, c='k')
    plt.clabel(cs, inline=1, fontsize=10)
    #plt.scatter(data[:,0], data[:,1], s=2, alpha=0.0001)
    plt.show()
    if fname is not None:
        plt.savefig(fname)
    plt.clf()
    plt.close()


def do_plot_wde(wde, fname, dist, interact=None):
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
    if interact == 'view':
        plt.show()
    elif interact == 'save':
        print('Saving %s' % fname)
        plt.savefig(fname)
    else:
        raise ValueError('Unknown plot option %s' % interact)
    print('%s saved' % fname)


def do_wde_contour(wde, fname, dist):
    print('Plotting %s' % fname)
    zlim = calc_maxv(dist)
    hd, corr_factor = hellinger_distance(dist, wde)
    print(wde.name, 'HD=', hd)
    ##return
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    zz = wde.pdf((xx, yy)) / corr_factor
    max_z = zz.max()
    fig = plt.figure()
    plt.title(wde.name + ('\nHD = %g' % hd), wrap=True)
    print('max_z', max_z)
    cs = plt.contour(xx, yy, zz, alpha=0.7, levels=np.linspace(0, max_z, 12))
    plt.clabel(cs, inline=1, fontsize=10)
    plt.show()
    plt.savefig(fname)
    print('%s saved' % fname)


def plot_energy(wde, fname):
    plt.figure()
    vals = wde.threshold.values
    jj = vals[:,2]
    for j in range(wde.delta_j):
        xx0 = vals[jj == j][:, 0]
        yy0 = vals[jj == j][:, 1]
        plt.plot(xx0, yy0, label='j=%d' % j)
    plt.axvline(x=wde.threshold.threshold, c='r')
    yy = vals[:, 1]
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('C')
    plt.ylabel('$B_C$')
    plt.legend()
    lbl = random.randint(1,10000)
    plt.title(str(lbl))
    print('Title >', lbl)
    plt.show()
    plt.savefig(fname)
    plt.close()
    print('%s saved' % fname)
    fname = fname.replace('energy', 'energy2')
    plt.figure()
    xx = np.array(range(vals.shape[0]))
    for j in range(wde.delta_j):
        xx0 = xx[jj == j]
        yy0 = vals[jj == j][:, 1]
        plt.plot(xx0, yy0, label='j=%d' % j)
    #plt.axvline(x=wde.threshold.pos_k, c='r')
    plt.ylim(min(yy)*0.95, max(yy)*1.05)
    plt.xlabel('$i$')
    plt.ylabel('$B_i$')
    plt.legend()
    plt.savefig(fname)
    plt.show()
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
    #plt.savefig(fname)
    plt.show()
    plt.close()
    #print('%s saved' % fname)


def do_kde_contour(kde, fname, dist, zlim):
    print('Plotting %s' % fname)
    hd, corr_factor = hellinger_distance(dist, kde)
    print('kde HD=', hd)
    grid_n = 100
    xx, yy = grid_as_vector(grid_n)
    grid2 = np.array((xx.flatten(), yy.flatten())).T
    vals = kde.pdf(grid2)
    zz = vals.reshape(xx.shape[0], yy.shape[0])

    ##zz_sum = zz.sum() / grid_n / grid_n  # not always near 1
    fig = plt.figure()
    max_z = zz.max()
    print('max_z', max_z)
    cs = plt.contour(xx, yy, zz, alpha=0.7, levels=np.linspace(0, max_z, 12))
    plt.title(('KDE bw=%s' % str(kde.bw)) + ('\nHD = %g' % hd))
    plt.clabel(cs, inline=1, fontsize=10)
    #plt.scatter(data[:,0], data[:,1], s=2, alpha=0.0001)
    plt.show()
    if fname is not None:
        plt.savefig(fname)
    plt.clf()
    plt.close()

    ##ax.set_title(('KDE bw=%s' % str(kde.bw)) + ('\nHD = %g' % hd))
    ##ax.set_zlim(0, zlim)



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