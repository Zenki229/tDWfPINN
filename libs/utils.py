## This code is from https://github.com/scaomath/eit-transformer
import gc
import os
import shutil
import sys
import yaml
import argparse
import math
import pickle
import copy
import random as rd
from collections import defaultdict
from contextlib import contextmanager
from datetime import date
import time 
import matplotlib

import numpy as np
# import pandas as pd # line 105
import psutil
import platform
import subprocess
import re
import torch
import logging
import tabulate
import torch.nn as nn
from tqdm.auto import tqdm
# import h5py
from scipy.io import loadmat
# matplotlib.use('agg')
from matplotlib import rc, rcParams, tri, cm
from matplotlib.colors import TwoSlopeNorm
import matplotlib.pyplot as plt
# try:
#     import plotly.express as px
#     import plotly.figure_factory as ff
#     import plotly.graph_objects as go
#     import plotly.io as pio
#     from plotly.subplots import make_subplots
# except ImportError as e:
#     print('Plotly not installed. Plotly is used to plot meshes. Run 3D code like Wave2D, NS equation should install plotly')

def loss_plot(path):
    count = 0
    loss = []
    while os.path.exists(path+'/train'+f'/loss_{count}.pkl'):
        with open(path+'/train'+f'/loss_{count}.pkl', 'rb') as f:
            data = pickle.load(f)
            loss.extend(data)
        count = count + 1
    # plot loss
    fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
    ax.semilogy(np.array(loss))
    plt.savefig(path+'/train'+'/loss_plot.jpg', dpi=150)
    plt.close()
def loss_err_plot(path):
    count = 0
    loss = []
    err = []
    while os.path.exists(path+'/train'+f'/loss_{count}.pkl'):
        with open(path+'/train'+f'/loss_{count}.pkl', 'rb') as f:
            data = pickle.load(f)
            loss.extend(data['loss'])
            err.extend(data['err'])
        count = count + 1
    # plot loss
    fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
    ax.semilogy(np.array(loss))
    plt.savefig(path+'/train'+'/loss_plot.jpg', dpi=150)
    plt.close()
    fig, ax = plt.subplots(layout='constrained', figsize=(19.2, 4.8))
    ax.semilogy(np.array(err))
    plt.savefig(path+'/train'+'/err_plot.jpg', dpi=150)
    plt.close()
def generate_path_save(config):
    path_save = config.path_save
    if os.path.exists(path_save):
        if config.overwrite:
            print(f"Overwriting existing directory: {path_save}")
            time.sleep(2)
            # Use shutil.rmtree to delete the entire directory and its contents
            shutil.rmtree(path_save)
            # Recreate the directory
            os.makedirs(path_save, exist_ok=True)
            os.mkdir(os.path.join(path_save, 'train'))
            os.mkdir(os.path.join(path_save, 'net'))
            os.mkdir(os.path.join(path_save, 'img'))
        else:
            # If directory exists and overwrite is not allowed, raise an error
            raise FileExistsError(
                f"Directory {path_save} already exists. Use --overwrite=true to overwrite it."
            )
    else:
        # If directory does not exist, create it (including any necessary parent directories)
        os.makedirs(path_save, exist_ok=True)
        os.mkdir(os.path.join(path_save, 'train'))
        os.mkdir(os.path.join(path_save, 'net'))
        os.mkdir(os.path.join(path_save, 'img'))


def get_size(bytes, suffix='B'):
    '''
    by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MiB'
        1253656678 => '1.17GiB'
    '''
    for unit in ['', 'Ki', 'Mi', 'Gi', 'Ti', 'Pi', 'Ei', 'Zi']:
        if abs(bytes) < 1024.0:
            return f"{bytes:3.2f} {unit}{suffix}"
        bytes /= 1024.0
    return f"{bytes:3.2f} 'Yi'{suffix}"


def get_file_size(filename):
    file_size = os.stat(filename)
    return get_size(file_size.st_size)


def get_processor_name():
    if platform.system() == "Windows":
        return platform.processor()
    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.check_output(command).strip()
    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.check_output(command, shell=True).strip()
        for line in all_info.decode("utf-8").split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)


def get_system():
    print("="*40, "CPU Info", "="*40)
    # number of cores
    print("Device name       :", get_processor_name())
    print("Physical cores    :", psutil.cpu_count(logical=False))
    print("Total cores       :", psutil.cpu_count(logical=True))
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print(f"Max Frequency    : {cpufreq.max:.2f} Mhz")
    print(f"Min Frequency    : {cpufreq.min:.2f} Mhz")
    print(f"Current Frequency: {cpufreq.current:.2f} Mhz")

    print("="*40, "Memory Info", "="*40)
    # get the memory details
    svmem = psutil.virtual_memory()
    print(f"Total     : {get_size(svmem.total)}")
    print(f"Available : {get_size(svmem.available)}")
    print(f"Used      : {get_size(svmem.used)}")

    print("="*40, "Software Info", "="*40)
    print('Python     : ' + sys.version.split('\n')[0])
    print('Numpy      : ' + np.__version__)
    # print('Pandas     : ' + pd.__version__)
    print('PyTorch    : ' + torch.__version__)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if device.type == 'cuda':
        print("="*40, "GPU Info", "="*40)
        print(f'Device     : {torch.cuda.get_device_name(0)}')
        print(f"{'Mem total': <15}: {round(torch.cuda.get_device_properties(0).total_memory/1024**3,1)} GB")
        print(
            f"{'Mem allocated': <15}: {round(torch.cuda.memory_allocated(0)/1024**3,1)} GB")
        print(
            f"{'Mem cached': <15}: {round(torch.cuda.memory_reserved(0)/1024**3,1)} GB")

    print("="*30, "system info print done", "="*30)


def get_seed(s, printout=True, cudnn=True):
    # rd.seed(s)
    os.environ['PYTHONHASHSEED'] = str(s)
    np.random.seed(s)
    # pd.core.common.random_state(s)
    # Torch
    torch.manual_seed(s)
    torch.cuda.manual_seed(s)
    if cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

    message = f'''
    os.environ['PYTHONHASHSEED'] = str({s})
    numpy.random.seed({s})
    torch.manual_seed({s})
    torch.cuda.manual_seed({s})
    '''
    if cudnn:
        message += f'''
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        '''

    if torch.cuda.is_available():
        message += f'''
        torch.cuda.manual_seed_all({s})
        '''
    if printout:
        print("\n")
        print(f"The following code snippets have been run.")
        print("="*50)
        print(message)
        print("="*50)


@contextmanager
def simple_timer(title):
    t0 = time()
    yield
    print("{} - done in {:.1f} seconds.\n".format(title, time() - t0))


class Colors:
    """Defining Color Codes to color the text displayed on terminal.
    """

    red = "\033[91m"
    green = "\033[92m"
    yellow = "\033[93m"
    blue = "\033[94m"
    magenta = "\033[95m"
    end = "\033[0m"


def color(string: str, color: Colors = Colors.yellow) -> str:
    return f"{color}{string}{Colors.end}"


@contextmanager
def timer(label: str, compact=False) -> None:
    '''
    https://www.kaggle.com/c/riiid-test-answer-prediction/discussion/203020#1111022
    print
    1. the time the code block takes to run
    2. the memory usage.
    '''
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    start = time()  # Setup - __enter__
    if not compact:
        print(color(f"{label}: start at {start:.2f};", color=Colors.blue))
        print(
            color(f"LOCAL RAM USAGE AT START: {m0:.2f} GB", color=Colors.green))
        try:
            yield  # yield to body of `with` statement
        finally:  # Teardown - __exit__
            m1 = p.memory_info()[0] / 2. ** 30
            delta = m1 - m0
            sign = '+' if delta >= 0 else '-'
            delta = math.fabs(delta)
            end = time()
            print(color(
                f"{label}: done at {end:.2f} ({end - start:.6f} secs elapsed);", color=Colors.blue))
            print(color(
                f"LOCAL RAM USAGE AT END: {m1:.2f}GB ({sign}{delta:.2f}GB)", color=Colors.green))
            print('\n')
    else:
        yield
        print(
            color(f"{label} - done in {time() - start:.6f} seconds. \n", color=Colors.blue))


def get_memory(num_var=10):
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in globals().items()), key=lambda x: -x[1])[:num_var]:
        print(color(f"{name:>30}:", color=Colors.green),
              color(f"{get_size(size):>8}", color=Colors.magenta))


def find_files(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        for _file in files:
            if name in _file:
                result.append(os.path.join(root, _file))
    return sorted(result)


def print_file_size(files):
    for file in files:
        size = get_file_size(file)
        filename = file.split('/')[-1]
        filesize = get_file_size(file)
        print(color(f"{filename:>30}:", color=Colors.green),
              color(f"{filesize:>8}", color=Colors.magenta))


@contextmanager
def trace(title: str):
    t0 = time()
    p = psutil.Process(os.getpid())
    m0 = p.memory_info()[0] / 2. ** 30
    yield
    m1 = p.memory_info()[0] / 2. ** 30
    delta = m1 - m0
    sign = '+' if delta >= 0 else '-'
    delta = math.fabs(delta)
    print(f"[{m1:.1f}GB ({sign}{delta:.3f}GB): {time() - t0:.2f}sec] {title} ", file=sys.stderr)


def get_cmap(n, cmap='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(cmap, n)


def get_date():
    today = date.today()
    return today.strftime("%b-%d-%Y")


def get_num_params(model):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    num_params = 0
    for p in model_parameters:
        num_params += np.prod(p.size()+(2,) if p.is_complex() else p.size())
        # num_params += p.numel() * (1 + p.is_complex())
    return num_params

def save_pickle(var, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump(var, f)


def load_pickle(load_path):
    with open(load_path, 'rb') as f:
        u = pickle.load(f)
    return u


def default(value, d):
    '''
    helper taken from https://github.com/lucidrains/linear-attention-transformer
    '''
    return d if value is None else value


class DotDict(dict):
    """
    https://stackoverflow.com/a/23689767/622119
    https://stackoverflow.com/a/36968114/622119
    dot.notation access to dictionary attributes
    """

    def __getattr__(self, attr):
        return self.get(attr)
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __getstate__(self):
        return self

    def __setstate__(self, state):
        self.update(state)
        self.__dict__ = self


def load_yaml(filename, key=None):
    with open(filename) as f:
        dictionary = yaml.full_load(f)
    if key is None:
        return DotDict(dictionary)
    else:
        return DotDict(dictionary[key])

def log_gen(path):
    logger = logging.getLogger('logger')
    logger.propagate = False
    logger.setLevel(logging.INFO)
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.StreamHandler):
            logger.removeHandler(handler)
    hdlr_f = logging.FileHandler(filename=path + '/logger.log', mode = 'w')
    hdlr_f.setLevel(logging.INFO)
    fmt_f = logging.Formatter(fmt="%(asctime)s-8s : %(lineno)s line - %(message)s", datefmt="%Y/%m/%d %H:%M:%S")
    hdlr_f.setFormatter(fmt_f)
    logger.addHandler(hdlr_f)
    return logger


if __name__ == "__main__":
    get_system()
    get_memory()
    # fig, ax = plt.subplots(figsize=(5,5),layout='constrained')
    # ax.set_xlim(left=0., right=1.)
    # ax.set_ylim(ymin=0., ymax=1.)
    # intervalx = np.arange(start=0.1, stop=1, step=0.1)
    # intervaly = np.arange(start=0.1, stop=1, step=0.2)
    # X, Y = np.meshgrid(intervalx, intervaly)
    # node = np.stack([X,Y], axis=-1).reshape((-1,2))
    # # print(node)
    # shownode(node, ax, marker='.', markersize=1, color='r', linestyle='none')
    # plt.show()
