import os
import argparse
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import models
import utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--outf', default='experiment00')
    parser.add_argument('--hddfile', default='harddata00.dat')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--niter', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    os.system('mkdir -p {0}'.format(args.outf))

    utils.seedme(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    GMODEL = 'netG.pth'         # path to pre-trained generator
    netG = models.DCGAN_G(isize=64, nz=30, nc=1, ngf=64).to(device)
    netG.load_state_dict(torch.load(GMODEL))
    for p in netG.parameters():
        p.requires_grad_(False)
    netG.eval()
    print netG

    netI = models.FC_leaky(idim=30, odim=30).to(device)
    print netI

    # load harddata
    ii, jj, vv = utils.load_harddata(args.hddfile)
    vv = 2.0*(vv - 0.5)         # move to tanh range
    vv = torch.from_numpy(vv.astype(np.float32)).to(device)

    # define mean logp
    def _mlogp(z):
        alpha = 0.1
        x = netG(z)
        llh = -0.5*((x[...,jj,ii] - vv)**2).sum()
        llh = llh/alpha
        lprior = (-0.5*(z**2).sum())
        return (llh + lprior)/args.batch_size

    w = torch.FloatTensor(args.batch_size, 30).to(device)

    optimizerI = optim.Adam(netI.parameters(), lr=1e-4, amsgrad=True)

    logger = utils.Logger(args.outf, netG, netI, args.hddfile, device)

    for i in xrange(args.niter):
        optimizerI.zero_grad()
        w.normal_(0,1)
        z = netI(w)
        mlogp = _mlogp(z.view(-1,30,1,1))
        ent = utils.sample_entropy(z)
        loss = - mlogp - ent
        loss.backward()
        optimizerI.step()

        logger.dump(loss.item(), -mlogp.item(), ent.item())

        # --- logging
        if (i+1) % int(args.niter/20) == 0:
            print '[{}/{}] loss: {} -logp: {} -ent: {}'.format(i+1, args.niter, loss.item(), -mlogp.item(), -ent.item())
            logger.flush(i+1)
