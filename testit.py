import atexit
import math
import sys
from datetime import datetime

from sklearn.neighbors import BallTree


import click
import numpy as np
from scipy.special import gamma
from scipy.ndimage import gaussian_filter1d


from dist_codes import dist_from_code


def calculate_nearest_balls(xs, k):
    "Calculate and store (k+1)-th nearest balls"
    dim = xs.shape[1]
    ball_tree = BallTree(xs)
    dist, inx = ball_tree.query(xs, k + 1)
    k_near_radious = dist[:, -1]
    xs_balls = np.power(k_near_radious, dim / 2.0)
    sqrt_vunit = (np.pi ** (dim / 4.0)) / (gamma(dim / 2.0 + 1) ** 0.5)
    return xs_balls * sqrt_vunit

def calculate_nearest_balls_1d(data, xs, k):
    "Calculate and store (k+1)-th nearest balls"
    dim = 1
    data = np.atleast_2d(data).T
    ball_tree = BallTree(data)
    dist, inx = ball_tree.query(np.atleast_2d(xs).T, k + 1)
    k_near_radious = dist[:, -1]
    xs_balls = np.power(k_near_radious, dim / 2.0)
    sqrt_vunit = (np.pi ** (dim / 4.0)) / (gamma(dim / 2.0 + 1) ** 0.5)
    return xs_balls * sqrt_vunit


def calculate_nearest_plus_one(data, xs, k):
    "Calculate and store (k+1)-th nearest balls"
    dim = 1
    data = np.atleast_2d(data).T
    ball_tree = BallTree(data)
    dist, inx = ball_tree.query(np.atleast_2d(xs).T, k + 2)
    k_near_radious = dist[:, -2]
    k_plus_one = dist[:,-1]
    k_balls = np.power(k_near_radious, dim / 2.0)
    k_p1_balls = np.power(k_plus_one, dim/2.0)
    sqrt_vunit = (np.pi ** (dim / 4.0)) / (gamma(dim / 2.0 + 1) ** 0.5)
    return inx, k_balls * sqrt_vunit, k_p1_balls * sqrt_vunit


@click.group()
def main():
    pass


@main.command()
@click.argument('dist_code')
def esr(dist_code):
    """
    Normalise the Empirical Square Root construction
    """
    dist = dist_from_code(dist_code)
    pp = []
    k = 1
    omega = gamma(k) / gamma(k + 0.5)
    omega *= omega ## 4/Pi
    print('omega^2 =', omega)
    for num_obvs in range(100, 3000, 25):
        data = dist.rvs(num_obvs)
        b1 = calculate_nearest_balls(data, k)
        pp.append((num_obvs, (b1 * b1).sum()))
        print('.', end='')
        sys.stdout.flush()
    print()
    pp = np.array(pp)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pp[:,0], pp[:,1])
    plt.show()
    plt.close()


@main.command()
@click.argument('num_obs', type=int)
@click.option('--k', type=int)
def hist_cv(num_obs, **kwargs):
    """
    Demo picking histograms bin width (1D) using CV and ESR
    -> CORRECT.
    """
    ## dist = dist_from_code(dist_code)
    pp = []
    k = kwargs['k'] if 'k' in kwargs and kwargs['k'] is not None else 1
    omega = gamma(k) / gamma(k + 0.5)
    data = np.hstack((np.random.normal(0.5, 0.5, size=int(num_obs/2)),
                      np.random.normal(3, 1, size=int(num_obs / 2))))
    print('shape=', data.shape)
    x0, x1 = data.min(), data.max() + 0.0001
    pp = []
    print('trials each', int(math.log(num_obs)))
    for nbins in range(2, int(num_obs/4)):
        ss = []
        while len(ss) < math.log(num_obs):
            idx0 = np.random.randint(0, num_obs, size=int(num_obs/10))
            idx1 = [ix for ix in range(num_obs) if ix not in idx0]
            data0 = np.delete(data, idx1)
            data1 = np.delete(data, idx0)
            b1 = calculate_nearest_balls_1d(data1, data0, k)
            hist, bin_edges = np.histogram(data1, bins=nbins, range=(x0, x1), density=True)
            sqrt_h = lambda x: math.sqrt(hist[int(nbins * (x - x0)/(x1 - x0))])
            ys = np.array([sqrt_h(xi) for xi in data0])
            val = omega * (ys * b1).sum() / math.sqrt(len(data0))
            ss.append(val)
        val = np.array(ss)
        px0, pxd = val.mean(), val.std()
        pp.append((nbins, px0, px0 - pxd, px0 + pxd))
        ## print(nbins, ',', val)
    pp = np.array(pp)
    sigma = 1 + int(math.log(num_obs))
    smo = gaussian_filter1d(pp[:,1], sigma=sigma, mode='nearest')
    nbins = int(pp[np.argmax(smo),0]/0.9)
    print('best width >>', nbins, '(sigma = %d)' % sigma)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pp[:,0], pp[:,1], 'k.', markersize=1)
    plt.plot(pp[:,0], smo, 'r-')
    plt.show()
    plt.close()
    plt.figure()
    plt.hist(data, bins=nbins)
    plt.show()


@main.command()
@click.argument('num_obs', type=int)
@click.option('--k', type=int)
def hist_cv2(num_obs, **kwargs):
    """
    Demo picking histograms bin width (1D) using CV and ESRD
    -> WRONG
    """
    ## dist = dist_from_code(dist_code)
    pp = []
    k = kwargs['k'] if 'k' in kwargs and kwargs['k'] is not None else 1
    omega = gamma(k) / gamma(k + 0.5)
    data = np.hstack((np.random.normal(0.5, 0.5, size=int(num_obs/2)),
                      np.random.normal(3, 1, size=int(num_obs / 2))))
    print('shape=', data.shape)
    x0, x1 = data.min(), data.max() + 0.0001
    pp = []
    print('trials each', int(math.log(num_obs)))
    # b1 = calculate_nearest_balls_1d(data, data, k)
    for nbins in range(2, int(num_obs/4)):
        ss = []
        idx0 = list(set(np.random.randint(0, num_obs, size=20)))
        idx1 = [ix for ix in range(num_obs) if ix not in idx0]
        data0 = np.delete(data, idx1)
        b1 = calculate_nearest_balls_1d(data, data0, k)
        ys = []
        for ix in idx0:
            data1 = np.delete(data, [ix])
            hist, bin_edges = np.histogram(data1, bins=nbins, range=(x0, x1), density=True)
            sqrt_h = lambda x: math.sqrt(hist[int(nbins * (x - x0)/(x1 - x0))])
            ys.append(sqrt_h(data[ix]))
        ys = np.array(ys)
        val = omega * (ys * b1).sum() / math.sqrt(len(idx0))
        ss.append(val)
        val = np.array(ss)
        px0, pxd = val.mean(), val.std()
        pp.append((nbins, px0, px0 - pxd, px0 + pxd))
        ## print(nbins, ',', val)
    pp = np.array(pp)
    sigma = 1 + int(math.log(num_obs))
    smo = gaussian_filter1d(pp[:,1], sigma=sigma, mode='nearest')
    nbins = int(pp[np.argmax(smo),0]/0.9)
    print('best bins # >>', nbins, '(sigma = %d)' % sigma)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pp[:,0], pp[:,1], 'k.', markersize=1)
    plt.plot(pp[:,0], smo, 'r-')
    plt.show()
    plt.close()
    plt.figure()
    plt.hist(data, bins=nbins)
    plt.show()


@main.command()
@click.argument('num_obs', type=int)
@click.option('--k', type=int)
def hist_cv3(num_obs, **kwargs):
    """
    Demo picking histograms bin width (1D) using CV and ESRD
    if i' = nn(i) => remove i', nn(i) = i'' =>  Sum H^{-i'}(X_i) Sqrt{NN_i} for i such nn(i)=i' ??
    """
    ## dist = dist_from_code(dist_code)
    pp = []
    k = kwargs['k'] if 'k' in kwargs and kwargs['k'] is not None else 1
    omega = gamma(k) / gamma(k + 0.5)
    data = np.hstack((np.random.normal(0.5, 0.5, size=int(num_obs/2)),
                      np.random.normal(3, 1, size=int(num_obs / 2))))
    print('shape=', data.shape)
    pp = []
    print('trials each', int(math.log(num_obs)))
    inx, k_balls, k_p1_balls = calculate_nearest_plus_one(data, data, k)
    # import code
    # code.interact('**', local=locals())
    for nbins in range(2, int(num_obs/4)):
        tot = 0.0
        print(nbins)
        for nni in set(inx[:,-2]):
            data1 = np.delete(data, [nni])
            x0, x1 = data1.min(), data1.max() + 0.0001
            hist, bin_edges = np.histogram(data1, bins=nbins, range=(x0, x1), density=True)
            sqrt_h = lambda x: math.sqrt(hist[int(nbins * (x - x0)/(x1 - x0))])
            xs = data[inx[inx[:,-2] == nni,0]]
            bs = k_p1_balls[inx[:,-2] == nni]
            ys = np.array([sqrt_h(x) for x in xs])
            tot += omega * (ys * bs).sum()
        pp.append((nbins, tot))
        ## print(nbins, ',', val)
    pp = np.array(pp)
    sigma = 1 + int(math.log(num_obs))
    smo = gaussian_filter1d(pp[:,1], sigma=sigma, mode='nearest')
    nbins = int(pp[np.argmax(smo),0])
    print('best bins # >>', nbins, '(sigma = %d)' % sigma)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(pp[:,0], pp[:,1], 'k.', markersize=1)
    plt.plot(pp[:,0], smo, 'r-')
    plt.show()
    plt.close()
    plt.figure()
    plt.hist(data, bins=nbins)
    plt.show()


if __name__ == "__main__":
    click.echo("RUNNING python " + " ".join(sys.argv), err=True)
    def wtime(t0):
        secs = (datetime.now() - t0)
        click.echo("[walltime %s]" % str(secs), err=True)
    atexit.register(wtime, datetime.now())
    main()

