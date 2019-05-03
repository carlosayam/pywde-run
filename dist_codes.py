import math
import random

import numpy as np
import scipy.stats as stats

from functools import reduce


from scipy.interpolate import LinearNDInterpolator


def mise_mesh(d=2):
    grid_n = 256 if d == 2 else 40
    VVs = [np.linspace(0.0,1.0, num=grid_n) for i in range(d)]
    return np.meshgrid(*VVs)


def sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def in_triangle(pt, points):
    v1, v2, v3 = points
    b1 = sign(pt, v1, v2) < 0.0
    b2 = sign(pt, v2, v3) < 0.0
    b3 = sign(pt, v3, v1) < 0.0
    return (b1 == b2) and (b2 == b3)

def triangle_area(points):
    a, b, c = points
    return math.fabs(a[0]*b[1] + b[0]*c[1] + c[0]*a[1] - a[0]*c[1] - b[0]*a[1] - c[0]*b[1])/2.0


def _pdf(probs, dists, grid):
    assert len(probs) == len(dists)
    if type(grid) == tuple or type(grid) == list:
        pos = np.stack(grid, axis=0)
        pos = np.moveaxis(pos, 0, -1)
    else:
        pos = grid
    pdf_vals = None
    for prob, dist in zip(probs, dists):
        vals = dist.pdf(pos)
        if pdf_vals is None:
            pdf_vals = vals * prob
        else:
            pdf_vals = np.add(pdf_vals, vals * prob)
    return pdf_vals


class TempleDist(object):
    def __init__(self, points, centre, code='tem1'):
        self.code = code
        self.dim = 2
        x0, y0 = points[0]
        x1, y1 = points[1]
        assert x0 <= centre[0] <= x1
        assert y0 <= centre[1] <= y1
        self.height = 3 / (x1 - x0) / (y1 - y0)
        pp = [(x, y, 0) for x,y in [(x0,y0), (x0,y1), (x1,y1), (x1,y0)]]
        pp += [(centre[0], centre[1], self.height)]
        pp = np.array(pp)
        self.fun = LinearNDInterpolator(pp[:,0:2], pp[:,2], fill_value=0.0)
        minp = np.array([x0, y0])
        maxp = np.array([x1, y1])
        self.gen = lambda : np.random.rand(2) * (maxp - minp) + minp

    def rvs(self, num):
        data = []
        while num > 0:
            pp = self.gen()
            u = random.random()
            if u <= max(0,self.fun(*pp)) / self.height:
                data.append(pp)
                num -= 1
                if num == 0:
                    break
        return np.array(data)

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            X, Y = grid
            grid = np.array((X.flatten(), Y.flatten())).T
            vals = np.clip(self.fun(grid),0,None)
            vals = vals.reshape(X.shape[0], Y.shape[0])
            return vals
        else:
            return np.clip(self.fun(grid),0,None)


class PyramidDist(object):
    def __init__(self, points, centre, code='pyr1'):
        if not in_triangle(centre, points):
            raise ValueError("centre must be inside")
        self.code = code
        self.dim = 2
        vol = triangle_area(points) / 3
        self.height = 1.0 / vol
        pp = [(x, y, 0) for x,y in points]
        #pp += [(0,0,0),(0,1,0),(1,0,0),(1,1,0)]
        pp += [(centre[0], centre[1], self.height)]
        pp = np.array(pp)
        self.fun = LinearNDInterpolator(pp[:,0:2], pp[:,2], fill_value=0.0)
        minp = pp[:,[0,1]].min(axis=0)
        maxp = pp[:,[0,1]].max(axis=0)
        self.gen = lambda : np.random.rand(2) * (maxp - minp) + minp

    def rvs(self, num):
        data = []
        while num > 0:
            pp = self.gen()
            u = random.random()
            if u <= max(0,self.fun(*pp)) / self.height:
                data.append(pp)
                num -= 1
                if num == 0:
                    break
        return np.array(data)

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            X, Y = grid
            grid = np.array((X.flatten(), Y.flatten())).T
            vals = np.clip(self.fun(grid),0,None)
            vals = vals.reshape(X.shape[0], Y.shape[0])
            return vals
        else:
            return np.clip(self.fun(grid),0,None)

class TriangleDist(object):
    def __init__(self, points, code='tri1'):
        self.points = points
        self.code = code
        self.dim = 2
        self._h = 1/triangle_area(points)

    def rvs(self, num):
        data = []
        while num > 0:
            pp = (random.random(), random.random())
            if in_triangle(pp, self.points):
                data.append(pp)
                num -= 1
                if num == 0:
                    break
        return np.array(data)

    def pdf(self, grid):
        @np.vectorize
        def inside(x, y):
            return in_triangle((x,y), self.points)
        return np.where(inside(*grid), self._h, 0.0)

class Beta2D(object):
    def __init__(self, a, b, code='beta'):
        self.dist = stats.beta(a, b)
        self.code = code
        self.dim = 2

    def rvs(self, num):
        data = []
        while num > 0:
            for d in self._rvs():
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)

    def pdf(self, grid):
        return self.dist.pdf(grid[0]) * self.dist.pdf(grid[1])


class UniformDistribution(object):
    def __init__(self):
        self.code = 'unif'
        self.dim = 2

    def rvs(self, num):
        return 0.25 + np.random.uniform(size=(num, self.dim))/2

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            x, y = grid
        else:
            x = grid[:,0]
            y = grid[:, 1]
        vals = np.less(0.25, x) & np.less(x, 0.75) & np.less(0.25, y) & np.less(y, 0.75)
        return 4 * vals

class TruncatedMultivariateNormal(stats._multivariate.multi_rv_generic):
    """Truncate multivariate to [0,1]^2"""
    def __init__(self, seed=None):
        super(TruncatedMultivariateNormal, self).__init__(seed)



class TruncatedMultiNormalD(object):
    """Truncated mixture or multivariate normal distributions. Dimension is inferred from first $\mu$"""

    def __init__(self, probs, mus, covs, code='mult'):
        self.code = code
        self.probs = probs
        self.dists = [stats.multivariate_normal(mean=mu, cov=cov) for mu, cov in zip(mus, covs)]
        self.dim = len(mus[0])
        z = _pdf(self.probs, self.dists, mise_mesh(self.dim))
        nns = reduce(lambda x, y: (x-1) * (y-1), z.shape)
        self.sum = z.sum()/nns

    def mathematica(self):
        # render Mathematica code to plot
        def fn(norm_dist):
            mu = np.array2string(norm_dist.mean, separator=',')
            mu = mu.replace('[','{').replace(']','}').replace('e','*^')
            cov = np.array2string(norm_dist.cov, separator=',')
            cov = cov.replace('[','{').replace(']','}').replace('e','*^')
            return 'MultinormalDistribution[%s,%s]' % (mu, cov)
        probs = '{%s}' % ','.join([str(f / min(self.probs)) for f in self.probs])
        dists = '{%s}' % ','.join([fn(d) for d in self.dists])
        resp = 'MixtureDistribution[%s,%s]' % (probs, dists)
        return resp

    def _rvs(self):
        while True:
            for xvs in zip(*[dist.rvs(100) for dist in self.dists]):
                yield xvs

    def rvs(self, num):
        data = []
        while num > 0:
            for dd in self._rvs():
                i = np.random.choice(np.arange(0,len(self.probs)), p=self.probs)
                d = dd[i]
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)

    def _pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            pos = np.stack(grid, axis=0)
            pos = np.moveaxis(pos, 0, -1)
        else:
            pos = grid
        vals = [dist.pdf(pos) for dist in self.dists]
        pdf_vals = vals[0] * self.probs[0]
        for i in range(len(self.probs) - 1):
            pdf_vals = np.add(pdf_vals, vals[i+1] * self.probs[i+1])
        #pdf_vals = pdf_vals / total
        return pdf_vals

    def pdf(self, grid):
        return self._pdf(grid)/self.sum


class TruncatedLaplace2D(object):
    def __init__(self, mu, scale, code='lap1', angle=30.):
        self.mu = mu
        self.scale = scale
        self.dim = 2
        self.code = code
        theta = (angle / 180.) * np.pi
        self.rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        self.sum = self._pdfsum()

    def rvs(self, num):
        resp = None
        while num > 0:
            vs = np.random.laplace(0.0, scale=self.scale, size=(num + 10, 2))
            vs = np.matmul(vs, self.rot) + self.mu
            cond = (0 <= vs[:, 0]) & (vs[:, 0] <= 1.0) & (0 <= vs[:, 1]) & (vs[:, 1] <= 1)
            vs = vs[cond][:num]
            if len(vs) == 0:
                continue
            num -= vs.shape[0]
            if resp is not None:
                resp = np.concatenate((resp, vs))
            else:
                resp = vs
        return resp

    def pdf(self, grid):
        if type(grid) == tuple or type(grid) == list:
            pos = np.stack(grid, axis=0)
            vals = np.moveaxis(pos, 0, -1)
            print('new shape', vals.shape)
        else:
            vals = grid
        # print('vals', vals)
        # print('vals - mu', vals - self.mu)
        # print('(vals - mu) rot^T', np.matmul(vals - self.mu, self.rot.T))
        vals = np.matmul(vals - self.mu, self.rot.T)
        vals = np.exp(-abs(vals)/self.scale) / (2 * self.scale)
        #print('pdf 2', vals)
        if len(vals.shape) == 2:
            vals = vals[:,0] * vals[:,1]
        else:
            vals = vals[:,:,0] * vals[:,:,1]
        return vals / self.sum

    def _pdfsum(self):
        grid = mise_mesh(self.dim)
        pos = np.stack(grid, axis=0)
        vals = np.moveaxis(pos, 0, -1)
        vals = np.matmul(vals - self.mu, self.rot.T)
        resp = np.exp(-abs(vals)/self.scale) / (2 * self.scale)
        xx, yy = resp[:,:,0], resp[:,:,1]
        nns = reduce(lambda x, y: x * y, resp.shape)/resp.shape[-1]
        resp = (xx * yy).sum()/nns
        return resp


class MixtureDistribution(object):
    def __init__(self, probs, dists, code=None):
        self.probs = np.array(probs) / sum(probs)
        self.dists = dists
        self.dim = dists[0].dim
        assert all([self.dim == dist.dim for dist in dists])
        if code:
            self.code = code
        else:
            self.code = 'x'.join([dist.code for dist in dists])

    def pdf(self, grid):
        return _pdf(self.probs, self.dists, grid)

    def _rvs(self):
        while True:
            for xvs in zip(*[dist.rvs(100) for dist in self.dists]):
                yield xvs

    def rvs(self, num):
        data = []
        while num > 0:
            for dd in self._rvs():
                i = np.random.choice(np.arange(0,len(self.probs)), p=self.probs)
                d = dd[i]
                if 0 <= d[0] and d[0] <= 1 and 0 <= d[1] and d[1] <= 1:
                    data.append(d)
                    num -= 1
                    if num == 0:
                        break
        return np.array(data)

def dist_from_code(code):
    if code == 'beta':
        return Beta2D(2, 4, code=code)
    elif code == 'mult' or code == 'mul2':
        sigma = 0.01
        return TruncatedMultiNormalD(
            [1.5/9, 7.5/9],
            [np.array([0.2, 0.3]), np.array([0.7, 0.7])],
            [np.array([[sigma/6, 0], [0, sigma/6]]), np.array([[0.015, sigma/64], [sigma/64, 0.015]])],
            code=code
        )
    elif code == 'mul3':
        sigma = 0.01
        return TruncatedMultiNormalD(
            [0.4, 0.3, 0.3],
            [np.array([0.3, 0.4, 0.35]),
             np.array([0.7, 0.7, 0.6]),
             np.array([0.7, 0.6, 0.35])],
            [np.array([[0.02, 0.01, 0.], [0.01, 0.02, 0.], [0., 0., 0.02]]),
             np.array([[0.0133333, 0., 0.], [0., 0.0133333, 0.], [0., 0., 0.0133333]]),
             np.array([[0.025, 0., 0.], [0., 0.025, 0.01], [0., 0.01, 0.025]])
             ],
            code=code
            )
    elif code == 'mix1':
        sigma = 0.05
        m1 = np.array([[sigma/6, 0], [0, sigma/6.5]])
        return TruncatedMultiNormalD(
            [1/2, 1/2],
            [np.array([0.2, 0.3]), np.array([0.7, 0.7])],
            [m1, m1],
            code=code
            )
    elif code == 'mix2':
        sigma = 0.05
        angle = 10.
        theta = (angle/180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        m1 = np.array([[sigma/6, 0], [0, sigma/8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        return TruncatedMultiNormalD(
            [1/2, 1/2],
            [np.array([0.4, 0.3]), np.array([0.7, 0.7])],
            [m1, m2],
            code=code
            )
    elif code == 'mix3':
        sigma = 0.03
        angle = 10.
        theta = (angle/180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])
        m1 = np.array([[sigma/6, 0], [0, sigma/7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8,4,2,1])
        prop = prop/prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85])],
            [m1, m2/2, m1/4, m2/8],
            code=code
            )
    elif code == 'mix4':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 4, 2, 1, 384])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]), np.array([0.5, 0.5])],
            [m1, m2 / 2, m1 / 4, m2 / 8, 0.18 * np.eye(2, 2)],
            code=code
        )
    elif code == 'mix5':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 7]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 4, 2, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]),
             np.array([0.5, 0.5])],
            [m1, m2 / 2, m1 / 6, m2 / 8],
            code=code
        )
    elif code == 'mix6':
        theta = np.pi / 4
        rot = lambda angle : np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
        m0 = np.array([[0.1, 0], [0, 0.0025]])
        m1 = np.dot(rot(theta), np.dot(m0, rot(theta).T)) / 2
        m2 = np.dot(rot(-theta), np.dot(m0, rot(-theta).T)) / 2
        prop = np.array([1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.3, 0.3]), np.array([0.7, 0.3])], [m1, m2],
            code=code
        )
    elif code == 'mix7': ## not good
        m0 = np.array([[0.1, 0], [0, 0.005]])
        m1 = np.array([[0.005, 0], [0, 0.1]])
        prop = np.array([1, 1, 1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.5, 0.3]),
             np.array([0.5, 0.7]),
             np.array([0.3, 0.5]),
             np.array([0.7, 0.5])],
            [m0, m0, m1, m1],
            code=code
        )
    elif code == 'mix8':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([8, 6, 5, 3])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85]),
             np.array([0.5, 0.5])],
            [m1, m2 / 1.5, m1 / 2, m2 / 3],
            code=code
        )
    elif code == 'mix9':
        sigma = 0.03
        angle = 10.
        theta = (angle / 180.) * np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)]])
        m1 = np.array([[sigma / 6, 0], [0, sigma / 8]])
        m2 = np.dot(rot, np.dot(m1, rot.T))
        prop = np.array([5, 50, 1, 1])
        prop = prop / prop.sum()
        return TruncatedMultiNormalD(
            prop.tolist(),
            [np.array([0.2, 0.3]), np.array([0.5, 0.5]), np.array([0.65, 0.7]), np.array([0.82, 0.85])],
            [m1, np.array([[0.05,0],[0,0.05]]), m2 / 4.5, m1 / 5],
            code=code
        )
    elif code == 'lap1':
        return TruncatedLaplace2D(np.array([0.5, 0.5]), 0.1, code)
    elif code == 'lap2':
        return TruncatedLaplace2D(np.array([0.5, 0.5]), 0.1, code, angle=45.)
    elif code == 'lap3':
        return TruncatedLaplace2D(np.array([0.4, 0.4]), 0.1, code, angle=45.)
    elif code == 'tri1':
        return TriangleDist(((0.1,0.2),(0.3,0.7),(0.8,0.2)))
    elif code == 'pyr1':
        return PyramidDist(((0.1,0.2),(0.4,0.9),(0.8,0.2)), (0.4,0.3))
    elif code == 'pyr2':
        return PyramidDist(((0.1, 0.2), (0.4, 0.9), (0.8, 0.2)), (0.4, 0.3))
    elif code == 'pmx1':
        ws = np.array([7, 7, 1, 1])
        ws = ws / sum(ws)
        dist0a = PyramidDist(((0.1,0.1),(0.4,0.9),(0.9,0.2)), (0.4,0.3))
        dist0b = PyramidDist(((0.1,0.9),(0.4,0.1),(0.9,0.9)), (0.4,0.3))
        dist1 = PyramidDist(((0.2,0.2),(0.3,0.3),(0.5,0.25)), (0.3,0.25))
        dist2 = PyramidDist(((0.3,0.4),(0.4,0.5),(0.5,0.45)), (0.4,0.45))
        return MixtureDistribution(
            ws,
            [dist0a, dist0b, dist1, dist2],
            code)
    elif code == 'pmx2':
        ws = np.array([20, 2, 2])
        ws = ws / sum(ws)
        dist0a = PyramidDist(((0.05, 0.05), (0.4, 0.95), (0.95, 0.05)), (0.4, 0.3))
        dist1 = PyramidDist(((0.2, 0.2), (0.3, 0.3), (0.5, 0.25)), (0.3, 0.25))
        dist2 = PyramidDist(((0.5, 0.4), (0.6, 0.5), (0.7, 0.45)), (0.6, 0.45))
        return MixtureDistribution(
            ws,
            [dist0a, dist1, dist2],
            code)
    elif code == 'unif':
        return UniformDistribution()
    elif code == 'tem1':
        return TempleDist([(0.1,0.1), (0.7,0.6)], (0.5, 0.4), 'tem1')
    elif code == 'tem2':
        return TempleDist([(0.25, 0.25), (0.75, 0.75)], (0.5, 0.5), 'tem2')
    elif code == 'tmx1':
        ws = np.array([7, 7, 1, 2])
        ws = ws / sum(ws)
        dist0a = TempleDist([(0.1,0.1), (0.7,0.8)], (0.5, 0.4), '')
        dist0b = TempleDist([(0.2,0.3), (0.9,0.9)], (0.4, 0.5), '')
        dist1 = TempleDist([(0.3,0.175), (0.7,0.225)], (0.35, 0.20), '')
        dist2 = TempleDist([(0.7,0.4), (0.8,0.8)], (0.75, 0.5), '')
        return MixtureDistribution(
            ws,
            [dist0a, dist0b, dist1, dist2],
            code)
    elif code == 'tmx2':
        ws = np.array([2, 2, 1, 1])
        ws = ws / sum(ws)
        dist0a = TempleDist([(0.1, 0.1), (0.7, 0.8)], (0.5, 0.4), '')
        dist0b = TempleDist([(0.2, 0.3), (0.9, 0.9)], (0.4, 0.5), '')
        dist1 = TempleDist([(0.3, 0.2), (0.7, 0.4)], (0.35, 0.25), '')
        dist2 = TempleDist([(0.6, 0.4), (0.8, 0.8)], (0.75, 0.5), '')
        return MixtureDistribution(
            ws,
            [dist0a, dist0b, dist1, dist2],
            code)
    elif code == 'tmx3':
        ws = np.array([1, 1, 3])
        ws = ws / sum(ws)
        dist1 = TempleDist([(0.125, 0.25), (0.25, 0.75)], (0.1875, 0.5), '')
        dist2 = TempleDist([(0.25, 0.125), (0.75, 0.25)], (0.5, 0.1875), '')
        dist3 = TempleDist([(0.125, 0.125), (0.75, 0.75)], (0.5, 0.5), '')
        return MixtureDistribution(
            ws,
            [dist1, dist2, dist3],
            code)
    elif code == 'tmx4':
        ws = np.array([1, 1])
        ws = ws / sum(ws)
        dist1 = TempleDist([(0.125, 0.625), (0.75, 0.75)], (0.25, 0.6875), '')
        dist2 = TempleDist([(0.5, 0.125), (0.625, 0.875)], (0.5625, 0.25), '')
        return MixtureDistribution(
            ws,
            [dist1, dist2],
            code)

    else:
        raise NotImplemented('Unknown distribution code [%s]' % code)
