import os
import csv
import struct
import pywt
import math
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime

from typing import Tuple, List, Optional, Dict

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from sklearn.manifold import SpectralEmbedding
from sklearn.cluster import KMeans

from common import ensure_dir

# mnist
# DB_PATH = './mnist'
# EXP_PATH = './RESP/mnist'
# fsion
DB_PATH = './mnist'
EXP_PATH = './RESP/mnist'


#
# Based on
# -- https://gist.github.com/akesling/5358964
# .. in turn inspired by
# -- http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
# which is GPL licensed
#

# Number of observations in each label
# label 0 ; total = 5923
# label 1 ; total = 6742
# label 2 ; total = 5958
# label 3 ; total = 6131
# label 4 ; total = 5842
# label 5 ; total = 5421
# label 6 ; total = 5918
# label 7 ; total = 6265
# label 8 ; total = 5851
# label 9 ; total = 5949

# runtime for diffs
# 0: [walltime 2:24:38.320335]
# 1: [walltime 4:10:13.823205]
# 2: [walltime 3:26:02.012994]
# 3: [walltime 3:40:54.393864]
# 4: [walltime 3:17:32.920296]
# 5: [walltime 2:51:20.263187]
# 6: [walltime 3:13:34.918116]
# 7: [walltime 3:38:54.247671]
# 8: [walltime 3:10:23.654660]
# 9: [walltime 3:17:26.720949]


class DatasetKind(enumerate):
    TRAINING = 'training'
    TEST = 'test'



class LabelReader(object):
    def __init__(self, path, label, loader: 'MnistLoad'):
        self._loader = loader
        self._nums = []
        with open(str(path / ('label-%d.csv' % label)), 'r') as fh:
            reader = csv.reader(fh)
            for row in reader:
                self._nums.append(int(row[0]))

    def __len__(self):
        return len(self._nums)

    def __iter__(self):
        return self._nums.__iter__()

    def __next__(self):
        pos = next(self._nums)
        img = self._loader.get_image(pos)
        return pos, img

    def __getitem__(self, idx):
        pos = self._nums[idx]
        img = self._loader.get_image(pos)
        return pos, img


class LabelFilesWriter(object):
    def __init__(self, path):
        self._fh = {}
        self._csv = {}
        self._path = path

    def __enter__(self):
        for label in range(10):
            fh = open(str(self._path / ('label-%d.csv' % label)), 'w')
            self._fh[label] = fh
            self._csv[label] = csv.writer(fh)
        return self

    def writerow(self, label, row):
        self._csv[label].writerow(row)

    def __exit__(self, exc_type, exc_val, exc_tb):
        exc1 = None
        for label, fh in self._fh.items():
            try:
                fh.close()
            except Exception as exc:
                if exc1 is None:
                    exc1 = exc
        if exc1:
            raise exc1


class MnistLoad(object):
    """
    Python class for importing the MNIST data set.  It implements
    -- get_image : callable, index: int -> tuple(label, numpy image)
    -- num : int, numer of images
    """

    def __init__(self, corpus, dataset):

        assert corpus in ['mnist', 'fsion']

        path = str(Path('RESP') / corpus / 'corpus')

        if dataset is DatasetKind.TRAINING:
            fname_img = os.path.join(path, 'train-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 'train-labels-idx1-ubyte')
        elif dataset is DatasetKind.TEST:
            fname_img = os.path.join(path, 't10k-images-idx3-ubyte')
            fname_lbl = os.path.join(path, 't10k-labels-idx1-ubyte')
        else:
            raise ValueError("dataset must be valid")

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            self.lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            self.img = np.fromfile(fimg, dtype=np.uint8).reshape(len(self.lbl), rows, cols)

    def get_image(self, idx):
        return self.img[idx]

    def get_label(self, idx):
        return self.lbl[idx]

    def __getitem__(self, idx):
        return self.get_label(idx), self.get_image(idx)

    def __len__(self):
        return len(self.lbl)

    def filter_label(self, label):
        return np.where(self.lbl == label)[0]

    def filter_no_label(self, label):
        return np.where(self.lbl != label)[0]


class DwtImg(object):
    """Embed image into the hyper-sphere"""
    def __init__(self, img, name=None, wave_name='sym3'):
        # wdec :: List of
        #   level 0: LL :: numpy.ndarray
        #   level N: (LH, HL, HH), N > 0, LH, HL, HH :: numpy.ndarray
        self._wave_name = wave_name
        if img is not None:
            sqrt1 = np.sqrt(img / img.sum())
            self.wdec = pywt.wavedec2(data=sqrt1, wavelet=wave_name, mode='zero')
        else:
            # wdec initialised directly
            self.wdec = None
        self.name = name

    def __matmul__(self, other) -> float:
        "piggyback dot product into a @ b"
        tot = 0.0
        for c1, c2 in zip(self.wdec, other.wdec):
            if type(c1) == tuple:
                # TODO check types/arguments
                for det_coeff1, det_coeff2 in zip(c1, c2):
                    tot += (det_coeff1 * det_coeff2).sum()
            else:
                tot += (c1 * c2).sum()
        return tot

    def __mul__(self, fnum) -> 'DwtImg':
        "Scalar multiplication"
        if type(fnum) not in [int, float, complex]:
            raise ValueError('Only multiplication by int, float or complex')
        return DwtImg.scalar_mult(self, fnum)

    def __rmul__(self, fnum) -> 'DwtImg':
        if type(fnum) not in [int, float, complex]:
            raise ValueError('Only multiplication by int, float or complex')
        return DwtImg.scalar_mult(self, fnum)

    def __neg__(self) -> 'DwtImg':
        return DwtImg.scalar_mult(self, -1)

    def __sub__(self, other):
        return self + DwtImg.scalar_mult(other, -1)

    def __add__(self, other: 'DwtImg') -> 'DwtImg':
        "This add to be understood in the tangent space"
        resp = []
        for c1, c2 in zip(self.wdec, other.wdec):
            if type(c1) == tuple:
                # TODO check types/arguments
                item = []
                for det_coeff1, det_coeff2 in zip(c1, c2):
                    item.append(det_coeff1 + det_coeff2)
                item = tuple(item)
            else:
                item = c1 + c2
            resp.append(item)
        obj = DwtImg(None)
        obj.wdec = resp
        return obj

    def __repr__(self):
        resp = []
        for cc in self.wdec:
            if type(cc) == tuple:
                items = []
                for pos, coefs in zip(['LH', 'HL', 'HH'], cc):
                    items.append(f'{pos} {coefs.shape} max={coefs.max()}')
                resp.append(f'({"; ".join(items)})')
            else:
                resp.append(f'LL {cc.shape} max={cc.max()}')
        resp = ', '.join(resp)
        resp = f'[DwtImg name={self.name} {resp}]'
        return resp

    @staticmethod
    def scalar_mult(obj, fnum):
        """Strictly speaking, this returns an object outside the hyper-sphere $|x| = 1$"""
        resp = []
        for c1 in obj.wdec:
            if type(c1) == tuple:
                item = []
                # TODO check types/arguments
                for det_coeff1 in c1:
                    item.append(fnum * det_coeff1)
                item = tuple(item)
            else:
                item = fnum * c1
            resp.append(item)
        obj = DwtImg(None)
        obj.wdec = resp
        return obj

    def calc_angle(self, other: 'DwtImg') -> float:
        tot, norm2_1, norm2_2 = 0.0, 0.0, 0.0
        for c1, c2 in zip(self.wdec, other.wdec):
            if type(c1) == tuple:
                # TODO check types/arguments
                for det_coeff1, det_coeff2 in zip(c1, c2):
                    tot += (det_coeff1 * det_coeff2).sum()
                    norm2_1 += (det_coeff1 * det_coeff1).sum()
                    norm2_2 += (det_coeff2 * det_coeff2).sum()
            else:
                tot += (c1 * c2).sum()
                norm2_1 += (c1 * c1).sum()
                norm2_2 += (c2 * c2).sum()
        # print(f'{tot}, {norm2_1}, {norm2_2}, {math.sqrt(norm2_1 * norm2_2)}')
        resp = math.acos(min(tot / math.sqrt(norm2_1 * norm2_2), 1.0))
        return resp


    def log(self, other: 'DwtImg'):
        dotprod = self @ other
        tilde = other - dotprod * self
        norm_tilde = math.sqrt(abs(tilde @ tilde))
        if dotprod > 1:
            dotprod = 1.0
        elif dotprod < -1:
            dotprod = -1.0
        resp = math.acos(dotprod)/norm_tilde * tilde
        # print(f'Log {self.name} ({other.name})')
        return resp

    def exp(self, vect: 'DwtImg') -> 'DwtImg':
        norm_vect = math.sqrt(vect @ vect)
        if norm_vect < 0.00001:
            return self
        resp = math.cos(norm_vect) * self + (math.sin(norm_vect) / norm_vect) * self
        norm_vect = math.sqrt(resp @ resp)
        return resp * (1.0 / norm_vect)

    def to_img(self):
        imgt = pywt.waverec2(self.wdec, wavelet=self._wave_name, mode='zero')
        return imgt * imgt


class KarcherEstimator(object):
    def __init__(self, means_all, k):
        for lbl in sorted(means_all.keys()):
            print('Means ', lbl, len(means_all[lbl]))
        self._means: Dict[int, List[DwtImg]] = means_all
        assert type(k) == int and k >= 1 and k <= min([len(v) for v in self._means.values()])
        self._k = k

    def predict(self, img: DwtImg) -> int:
        resp = []
        for lbl, means in self._means.items():
            for kmean in means:
                dist = img.calc_angle(kmean)
                resp.append((dist, lbl))
        resp = sorted(resp, key=lambda tup: tup[0])[:self._k]
        counter = defaultdict(lambda: 0)
        for _, lbl in resp:
            counter[lbl] += 1
        counter = list(counter.items())
        counter = sorted(counter, key=lambda tup: tup[1], reverse=True)
        return counter[0][0]




def calc_diff(img1, img2, wave_name='sym3'):
    rep1 = DwtImg(img1, wave_name)
    rep2 = DwtImg(img2, wave_name)
    angle = rep1.calc_angle(rep2)
    return angle


class CalcRandomCenters(object):
    def __init__(self, corpus):
        self._corpus = corpus

    def run(self):
        loader = MnistLoad(self._corpus, DatasetKind.TRAINING)
        data = load_sample(label, is_main=True)
        idxs = np.random.choice(data[:,0], N_CLUSTERS, replace=False)
        means = []
        for ix in idxs:
            lbl, img = loader[ix]
            means.append(img)
        fmeans = str(Path('RESP') / DB_PATH / 'knn' / f'means-{label}.pickle')
        with open(fmeans, 'wb') as fh:
            pickle.dump(means, fh)
        print(f'Done Random means saved to {fmeans}')

FACTORS = {
    'mnist': {
        0: 0.981,
        1: 0.987,
        2: 0.866,
        3: 0.879,
        4: 0.716,
        5: 0.68,
        6: 0.924,
        7: 0.842,
        8: 0.728,
        9: 0.831
    },
    'fsion' : {
        0: 0.6,
        1: 0.917,
        2: 0.558,
        3: 0.725,
        4: 0.594,
        5: 0.44,
        6: 0.413,
        7: 0.851,
        8: 0.893,
        9: 0.915
    }
}

class CalcKarcherMeans(object):
    def __init__(self, corpus, affinity, embed, means: str, knn):
        self._corpus = corpus
        self._embed = embed
        self._affinity = affinity
        if means.endswith('+'):
            self._factor = FACTORS[corpus]
            self._means = int(means[:-1])
            maxf = max(self._factor.values())
            self._factor = {k: v/maxf for k, v in self._factor.items()}
        else:
            self._factor = {lbl: 1.0 for lbl in range(10)}
            self._means = int(means)
        self._knn = knn
        self._all_new_obvs = {}

    @property
    def _root(self) -> Path:
        return Path('RESP') / self._corpus

    def _source_diff(self, label):
        return str(self._root / 'diffs' / f'diff-{label}.npz')

    def _load_diff(self, label):
        with np.load(self._source_diff(label)) as npz:
            arr = npz['arr_0']

        if self._affinity == 'dist':
            max_d = arr[:, 2].max()
            data = 1 - arr[:, 2] / max_d
        else:
            sigma = float(self._affinity)
            data = np.exp(-(arr[:, 2] ** 2) / (sigma ** 2))

        num_obvs = arr[:, 1].astype(int).max() + 1
        ix = arr[:, 0].astype(int)
        jx = arr[:, 1].astype(int)

        # turn triangular sup matrix into full matrix
        data = np.concatenate((data, data, np.ones(num_obvs)), axis=0)
        ii = np.concatenate((ix, jx, np.array(range(num_obvs))), axis=0)
        jj = np.concatenate((jx, ix, np.array(range(num_obvs))), axis=0)
        arr_sparse = coo_matrix((data, (ii, jj)))
        return arr_sparse.toarray()

    def pre_run(self):
        all_new_obvs = {}
        for label in range(10):
            aff_mat = self._load_diff(label)
            if self._embed != 'no':
                # calculating a spectral embedding
                embedding = SpectralEmbedding(n_components=int(self._embed), affinity='precomputed')
                new_obs = embedding.fit_transform(aff_mat, aff_mat)
            else:
                new_obs = aff_mat
            all_new_obvs[label] = new_obs
        self._all_new_obvs = all_new_obvs

    def run(self):
        self.pre_run()
        for _ in range(10):
            for resp in self.run1():
                print(resp)

    def run1(self):
        means_all = {}

        for label in range(10):
            new_obs = self._all_new_obvs[label]

            # computing clusters as a way to cover this embedding
            n_clusters = int(self._means / (self._factor[label] ** 2))
            kmeans = KMeans(n_clusters=n_clusters, init='random')
            new_clusters = kmeans.fit_predict(new_obs, new_obs)
            # calculate Karcher mean of each cluster
            means = []
            for num_cluster in range(n_clusters):
                idxs = np.argwhere(new_clusters == num_cluster).flatten()
                karcher_mean: DwtImg = calc_karcher_mean(self._corpus, label, idxs)
                means.append(karcher_mean)
            means_all[label] = means

        yield self._error_on_test(means_all)

    def _error_on_test(self, means_all):
        karcher_estimator = KarcherEstimator(means_all, self._knn)
        loader = MnistLoad(self._corpus, dataset=DatasetKind.TEST)
        resp = []
        t0 = datetime.now()
        for ix in range(len(loader)):
            true_lbl, img = loader[ix]
            pred_lbl = karcher_estimator.predict(DwtImg(img, name=f'test-{ix}'))
            resp.append((ix, true_lbl, pred_lbl))
            if ix % 100 == 99:
                print('.', end='')
                sys.stdout.flush()
        agv_t = (datetime.now() - t0).total_seconds() / len(loader)
        print('')
        resp = np.array(resp)
        result = []
        for lbl in range(10):
            resp_lbl = resp[resp[:,1] == lbl,:]
            num_lbl = resp_lbl.shape[0]
            accurate = (resp_lbl[:,1] == resp_lbl[:,2]).sum()
            print(f'Acur {lbl}: {accurate/num_lbl}')
            result.append(accurate/num_lbl)
        accurate = (resp[:, 1] == resp[:, 2]).sum()
        result.append(accurate/resp.shape[0])
        result.append(agv_t)
        return result


def calc_karcher_mean(corpus, label, idxs) -> DwtImg:
    """Calculates the Riemannian centre of mass"""

    def get_it(ix: int) -> DwtImg:
        """Memoized image loading into DwtImg"""
        if ix in cache:
            return cache[ix]
        lbl, img = loader[data[idxs[ix]]]
        img = DwtImg(img, name='Img%04d' % data[idxs[ix]])
        cache[ix] = img
        # print(repr(img), '->', f'{img @ img}')
        return img

    loader = MnistLoad(corpus, DatasetKind.TRAINING)
    cache = {}

    conv_rate = 0.01
    data = loader.filter_label(label)
    mu: DwtImg = get_it(0)
    epsilon, num_iter, num = 10, 0, len(idxs)
    gamma_t_1 = None
    # print(f'Karcher mean: Label {label}, Cluster#: {num_cluster_1}, Cluster size: {num}')
    while epsilon > 0.0001:
        the_sum = None
        for i in range(num):
            item = mu.log(get_it(i))
            if the_sum is None:
                the_sum = item
            else:
                the_sum = the_sum + item
            # print(i, repr(item),'\n', repr(the_sum))
        gamma_t: DwtImg
        gamma_t = (conv_rate/num) * the_sum
        mu = mu.exp(gamma_t)
        if gamma_t_1 is None:
            epsilon = 1
        else:
            epsilon = gamma_t.calc_angle(gamma_t_1)
        num_iter += 1
        mu.name = 'Mu_%d' % num_iter

        # print(repr(gamma_t))
        # print(f'gamma_t^2 = {gamma_t @ gamma_t}')
        # print(repr(mu),'\n')
        # print(f'mu^2 = {mu @ mu}')

        gamma_t_1 = gamma_t
        # print(f'{num_iter}: {epsilon}')
    return mu


def load_sample(label, is_main):
    prefix = 'label' if is_main else 'other'
    #print('Load labels', str(Path(EXP_PATH) / 'labels'))
    fname = str(Path(EXP_PATH) / 'labels' / f'{prefix}-{label}.csv')
    df = pd.read_csv(fname, header=None, delimiter=',', names=['x'])
    #print(f'{prefix}-{label} = {df.values.shape} - min {df.values.min()}, max {df.values.max()}')
    return df.values


class CalcLabels(object):

    def __init__(self, corpus):
        self._corpus = corpus

    def run(self):
        loader = MnistLoad(self._corpus, DatasetKind.TRAINING)
        self._ensure_dir()

        for lbl in range(10):

            # save in-set positions
            arr = loader.filter_label(lbl)
            fdest = self._dest_label(lbl)
            np.savez(fdest, arr)

            # save not in-set positions (a sample)
            others = loader.filter_no_label(lbl)
            size = int(0.1 * len(others))
            others_arr = np.random.choice(others, size=size, replace=False)
            fdest = self._dest_others(lbl)
            np.savez(fdest, others_arr)

    def _ensure_dir(self):
        ensure_dir(self._root)

    @property
    def _root(self) -> Path:
        return Path('RESP') / self._corpus / 'labels'

    def _dest_label(self, lbl):
        return str(self._root / f'label-{lbl}.npz')

    def _dest_others(self, lbl):
        return str(self._root / f'other-{lbl}.npz')


class ConvertDiffs(object):

    def __init__(self, corpus):
        self._corpus = corpus

    def run(self):
        for lbl in range(10):
            print(self._source(lbl), '->', self._dest(lbl))
            df = pd.read_csv(self._source(lbl))
            arr = df.values
            arr = arr[:, 2:]
            np.savez_compressed(self._dest(lbl), arr)

    @property
    def _root(self) -> Path:
        return Path('RESP') / self._corpus / 'diffs'

    def _source(self, lbl):
        return str(self._root / f'diff-{lbl}.csv')

    def _dest(self, lbl):
        return str(self._root / f'diff-{lbl}.npzz')


def calc_diffs_all(label, wave_name):
    """
    calculate distance in image manifold
    :param label:
    :param path:
    :return:
    """
    loader = MnistLoad(DatasetKind.TRAINING)
    fdiffname = Path('RESP') / DB_PATH / 'diffs' / ('diff-%s.csv' % label)
    ensure_dir(fdiffname.parent)
    total, label = 0, int(label)
    with open(str(fdiffname), 'wt') as fh:
        writer = csv.writer(fh)
        # num1 and num2 are the original indexes in the MNIST database
        writer.writerow(['num1', 'num2', 'pos1', 'pos2', 'angle'])
        reader = LabelReader(Path('RESP') / 'mnist' / 'labels', label, loader)
        total = len(reader)
        for pos1 in range(total - 1):
            num1, img1 = reader[pos1]
            for pos2 in range(pos1 + 1, total):
                num2, img2 = reader[pos2]
                angle = calc_diff(img1, img2, wave_name=wave_name)
                writer.writerow([num1, num2, pos1, pos2, angle])
    print('label', label, '; total =', total)


def show_image(plts, image, lbl):
    """
    Render a given numpy.uint8 2D array of pixel data.
    """
    mpl, plt = plts
    plt = mpl.pyplot
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.title(str(lbl))
    plt.show()
