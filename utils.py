import random
import cPickle
from collections import defaultdict
import numpy as np
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

import torch

def seedme(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def config_logging(logging, outf, fname='log.log'):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename='{}/{}'.format(outf, fname),
                        filemode='w')
    # define a Handler which writes INFO messages or higher to the sys.stderr
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    # set a format which is simpler for console use
    formatter = logging.Formatter('%(message)s')
    # tell the handler to use this format
    console.setFormatter(formatter)
    # add the handler to the root logger
    logging.getLogger('').addHandler(console)

def bin2tanh(x):  # [0,1] -> [-1,1]
    return 2.0*(x - 0.5)

def distance_matrix(sample):
    m = sample.size(0)
    sample_norm = sample.mul(sample).sum(dim=1)
    sample_norm = sample_norm.expand(m, m)
    mat = sample.mm(sample.t()).mul(-2) + sample_norm.add(sample_norm.t())
    return mat

def sample_entropy(sample):
    """ Estimator based on kth nearest neighbor, "A new class of random
    vector entropy estimators (Goria et al.)" """
    sample = sample.view(sample.size(0), -1)
    m, n = sample.shape

    mat_ = distance_matrix(sample)
    mat, _ = mat_.sort(dim=1)
    k = int(np.round(np.sqrt(sample.size(0))))  # heuristic
    rho = mat[:,k]  # kth nearest
    entropy = 0.5*(rho + 1e-16).log().sum()
    entropy *= float(n)/m

    return entropy

def load_condfile(fname, skiprows=5):
    data = np.loadtxt(fname, skiprows=skiprows)
    jj, ii, vals = data.T
    ij = np.asarray(zip(ii,jj), dtype=int)
    return ij, vals

def medfilt_plot(ax, y, x=None, start=500, filter_width=101, symlog=True, **kws):
    if x is None:
        x = range(len(y))[start:][(filter_width/2):-(filter_width/2)]
    ax.plot(x, signal.medfilt(y[start:], filter_width)[(filter_width/2):-(filter_width/2)], **kws)
    if symlog:
        ax.set_yscale('symlog')

class History(object):
    def __init__(self, outdir):
        self.outdir = outdir
        self.history = defaultdict(list)

    def dump(self, **kws):
        for k, v in kws.iteritems():
            self.history[k].append(v)

    def flush(self):
        fig, axs = plt.subplots(len(self.history), 1, sharex=True, figsize=(6,6))
        for ax, k in zip(axs.flat, self.history):
            ax.set_ylabel(k)
            medfilt_plot(ax, self.history[k])
        axs[-1].set_xlabel('iteration')
        fig.savefig('{0}/history.png'.format(self.outdir))
        plt.close(fig)

        with open('{}/history.pkl'.format(self.outdir), 'wb') as f:
            cPickle.dump(self.history, f)

class NetGI(object):
    def __init__(self, netG, netI):
        self.netG = netG
        self.netI = netI

    def __call__(self, w):
        z = self.netI(w)
        x = self.netG(z.view(z.shape[0],z.shape[1],1,1))
        return x

def fill_imgs(axs, imgs):
    for img, ax in zip(imgs, axs.flat):
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.xaxis.set_major_locator(NullLocator())
        ax.yaxis.set_major_locator(NullLocator())
        ax.set_xlim(0, img.shape[1]-1)
        ax.set_ylim(0, img.shape[0]-1)
        ax.imshow(img, origin='lower')

def scatter_ij(ax, ij, vals):
    ij0, ij1 = ij[np.where(vals==0)], ij[np.where(vals==1)]
    ax.scatter(ij0[:,1], ij0[:,0], marker='x', s=20, c='C1')
    ax.scatter(ij1[:,1], ij1[:,0], marker='o', s=20, c='C0')

class Plotter(object):
    def __init__(self, outdir, netG, netI, condfile, w_plots):
        self.outdir = outdir
        self.netGI = NetGI(netG, netI)
        self.w_plots = w_plots

        self.ij, self.vals = load_condfile(condfile)

    def flush(self, iteration):
        self.netGI.netI.eval()
        with torch.no_grad():
            x = self.netGI(self.w_plots).detach().cpu().numpy().squeeze()
        self.netGI.netI.train()

        ncols = 8
        nrows = len(x) / ncols
        fig, axs = plt.subplots(nrows, ncols, figsize=(10, 10*nrows/ncols))
        fig.subplots_adjust(top=1,right=1,bottom=0,left=0, hspace=.05, wspace=.05)
        fill_imgs(axs, x)
        for ax in axs.flat:
            scatter_ij(ax, self.ij, self.vals)

        fig.savefig('{}/samples_{}.png'.format(self.outdir, iteration), bbox_inches='tight', pad_inches=0)
        plt.close(fig)
