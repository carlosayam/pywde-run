import os
import csv
import struct
import numpy as np
import pywt
import math

from pathlib import Path
from common import ensure_dir


#
# Based on
# -- https://gist.github.com/akesling/5358964
# .. in turn inspired by
# -- http://abel.ee.ucla.edu/cvxopt/_downloads/mnist.py
# which is GPL licensed
#


class DatasetKind(enumerate):
    TRAINING = 'training'
    TEST = 'test'


class LabelReader(object):
    def __init__(self, path, label, get_image):
        self._get = get_image
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
        _, img = self._get(pos)
        return pos, img

    def __getitem__(self, idx):
        pos = self._nums[idx]
        _, img = self._get(pos)
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


def load(dataset = DatasetKind.TRAINING, path = "."):
    """
    Python function for importing the MNIST data set.  It returns a
    tuple: get_image, num
    -- get_image : callable, index: int -> tuple(label, numpy image)
    -- num : int, numer of images
    """

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
        print(magic, num)
        lbl = np.fromfile(flbl, dtype=np.int8)

    with open(fname_img, 'rb') as fimg:
        magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
        img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

    get_img = lambda idx: (lbl[idx], img[idx])

    return get_img, len(lbl)


def calc_diff(img1, img2, wave_name='sym3'):
    sqrt1 = np.sqrt(img1 / img1.sum())
    sqrt2 = np.sqrt(img2 / img2.sum())
    wdec1 = pywt.wavedec2(data=sqrt1, wavelet=wave_name, mode='zero')
    wdec2 = pywt.wavedec2(data=sqrt2, wavelet=wave_name, mode='zero')
    angle = calc_angle(wdec1, wdec2)
    return angle


def calc_angle(wdec1, wdec2):
    tot, norm2_1, norm2_2 = 0.0, 0.0, 0.0
    for c1, c2 in zip(wdec1, wdec2):
        if type(c1) == tuple:
            # TODO check types/arguments
            for det_coeff1, det_coeff2 in zip(c1,c2):
                tot += (det_coeff1 * det_coeff2).sum()
                norm2_1 += (det_coeff1 * det_coeff1).sum()
                norm2_2 += (det_coeff2 * det_coeff2).sum()
        else:
            tot += (c1 * c2).sum()
            norm2_1 += (c1 * c1).sum()
            norm2_2 += (c2 * c2).sum()
    resp = math.acos(min(tot / math.sqrt(norm2_1 * norm2_2), 1.0))
    return resp


def calc_labels(path):
    get_image, num = load(DatasetKind.TRAINING, path)
    fdiffdir = Path('RESP') / 'mnist' / 'labels'
    ensure_dir(fdiffdir)
    with LabelFilesWriter(fdiffdir) as csvs:
        for num in range(num):
            lbl1, _ = get_image(num)
            lbl1 = int(lbl1)
            csvs.writerow(lbl1, [num])


def calc_diffs_all(label, path, wave_name):
    """
    calculate distance in image manifold
    :param label:
    :param path:
    :return:
    """
    get_image, num = load(DatasetKind.TRAINING, path)
    fdiffname = Path('RESP') / 'mnist' / 'diffs' / ('diff-%s.csv' % label)
    ensure_dir(fdiffname.parent)
    total, label = 0, int(label)
    with open(str(fdiffname), 'wt') as fh:
        writer = csv.writer(fh)
        # num1 and num2 are the original indexes in the MNIST database
        writer.writerow(['num1', 'num2', 'pos1', 'pos2', 'angle'])
        reader = LabelReader(Path('RESP') / 'mnist' / 'labels', label, get_image)
        total = len(reader)
        exp = total * (total - 1) // 2
        num = 0
        for pos1 in range(total - 1):
            num1, img1 = reader[pos1]
            for pos2 in range(pos1 + 1, total):
                num2, img2 = reader[pos2]
                angle = calc_diff(img1, img2, wave_name=wave_name)
                writer.writerow([num1, num2, pos1, pos2, angle])
                num += 1
                if num % 1000 == 0:
                    print('\n%d/%d ' % (num, exp))
                else:
                    print('.', end='')
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
