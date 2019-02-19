import os
import pathlib

ROOT_DIR = pathlib.Path('RESP')
NUM_SAMPLES = 100


def fname(what, dist_name, num=None, wave_name=None, delta_j=None, ext='.png'):
    strn = '%04d' % num if num is not None else None
    strd = '%d' % delta_j if delta_j is not None else None
    strs = [dist_name, what, strn, wave_name, strd]
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
    filename = '%s-%03d.tab' % (what, sample_no)
    path = ensure_dir(base_dir(dist_name, num_obvs) / 'results') / filename
    return path.absolute()


def png_name(dist_name, num_obvs, sample_no, what):
    filename = '%s-%03d.png' % (what, sample_no)
    path = ensure_dir(base_dir(dist_name, num_obvs) / 'plots') / filename
    return path.absolute()

