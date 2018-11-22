import random
import torch
import numpy as np
import scipy.signal as signal

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator

def distance_matrix(sample):
    m = sample.size(0)
    sample_norm = sample.mul(sample).sum(dim=1)
    sample_norm = sample_norm.expand(m, m)
    mat = sample.mm(sample.t()).mul(-2) + sample_norm.add(sample_norm.t())
    return mat

def sample_entropy(sample):
    """
    Estimator based on kth nearest neighbor, "A new class of random vector
    entropy estimators (Goria et al.)"
    """
    sample = sample.view(sample.size(0), -1)
    m, n = sample.shape

    mat_ = distance_matrix(sample)
    mat, _ = mat_.sort(dim=1)
    k = int(np.round(np.sqrt(sample.size(0))))  # heuristic
    rho = mat[:,k]  # kth nearest
    entropy = 0.5*(rho + 1e-16).log().sum()
    entropy *= float(n)/m

    return entropy

def load_harddata(hddfile, skiprows=5):
    data = np.loadtxt(hddfile, skiprows=skiprows)
    ii, jj, vv = data.T
    ii = ii.astype(int)
    jj = jj.astype(int)
    return ii, jj, vv

def seedme(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class Logger(object):
    def __init__(self, outf, netG, netI, hddfile, device):
        self.outf = outf
        self.netG = netG
        self.netI = netI
        self.device = device
        self.hddfile = hddfile

        self.losses = []
        self.ents = []
        self.logps = []

        # fot plots
        self.nrows, self.ncols = 8, 8
        self.w = torch.randn(self.nrows*self.ncols,30).to(self.device)

        # load and sort harddata, for plotting
        ii, jj, vv = load_harddata(hddfile)
        kk0 = np.where(vv == 0.)
        kk1 = np.where(vv == 1.)
        ij0 = (ii[kk0], jj[kk0])
        ij1 = (ii[kk1], jj[kk1])
        self.ij0, self.ij1 = ij0, ij1

    def dump(self, loss, logp, ent):
        self.losses.append(loss)
        self.ents.append(ent)
        self.logps.append(logp)

    def plot_loss(self):
        fig, axs = plt.subplots(3, 1, sharex=True, figsize=(6,6))
        axs[0].plot(signal.medfilt(self.losses, 5)[2:-2])
        axs[0].set_ylabel('KL')
        axs[1].semilogy(signal.medfilt(self.ents, 5)[2:-2])
        axs[1].set_ylabel('entropy')
        axs[2].semilogy(signal.medfilt(self.logps, 5)[2:-2])
        axs[2].set_ylabel('-logp')
        axs[-1].set_xlabel('iteration')
        fig.savefig('{0}/loss.png'.format(self.outf))
        plt.close(fig)

    def plot_sample(self, i):
        netG = self.netG
        netI = self.netI
        ij0 = self.ij0
        ij1 = self.ij1
        nrows, ncols = self.nrows, self.ncols
        w = self.w

        netI.eval()
        with torch.no_grad():
            z = netI(w)
            x = netG(z.view(-1,30,1,1)).detach().cpu().numpy().squeeze()
        netI.train()

        fig = plt.figure(figsize=(10, 10*nrows/ncols))
        fig.subplots_adjust(top=1,right=1,bottom=0,left=0, hspace=.1, wspace=.01)
        axs = fig.subplots(nrows, ncols).ravel()
        for x_, ax in zip(x, axs):
            ax.xaxis.set_visible(False)
            ax.yaxis.set_visible(False)
            ax.xaxis.set_major_locator(NullLocator())
            ax.yaxis.set_major_locator(NullLocator())
            ax.imshow(x_, origin='lower')
            ax.scatter(ij1[0], ij1[1], marker='o', s=20)
            ax.scatter(ij0[0], ij0[1], marker='x', s=20)
            ax.set_xlim(0,63)
            ax.set_ylim(0,63)
        fig.savefig('{}/sample_{}.png'.format(self.outf, i), bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    def flush(self, i):
        self.plot_loss()
        self.plot_sample(i)
        torch.save(self.netI.state_dict(), '{}/netI_iter_{}.pth'.format(self.outf, i+1))
