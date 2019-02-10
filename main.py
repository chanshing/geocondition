import os
import argparse
import logging
import numpy as np

import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
cudnn.benchmark = True

import models
import utils

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--outdir', default='gen/cond01', help='directory to store results')
parser.add_argument('--condfile', default='dat/cond01.dat', help='file with conditioning config')
parser.add_argument('--alpha', type=float, default=0.1, help='likelihood variance')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--niter', type=int, default=1000)
parser.add_argument('--lr', type=float, default=1e-4)
# --- netG params
parser.add_argument('--archG', default='DCGAN_G')
parser.add_argument('--netG', default='netG.pth')
parser.add_argument('--image_size', type=int, default=64)
parser.add_argument('--image_depth', type=int, default=1, help='e.g. 1 for B&W, 3 for RGB')
parser.add_argument('--num_filters', type=int, default=64, help='(\propto) number of conv filters per layer')
parser.add_argument('--nz', type=int, default=30, help='size of latent vector z')
# --- netI params
parser.add_argument('--archI', default='FC_selu')
parser.add_argument('--netI', default=None)
parser.add_argument('--hidden_layer_size', type=int, default=512)
parser.add_argument('--num_extra_layers', type=int, default=2)
parser.add_argument('--nw', type=int, default=30, help='size of latent vector w')
args = parser.parse_args()

os.system('mkdir -p {0}'.format(args.outdir))
utils.config_logging(logging, args.outdir)
utils.seedme(args.seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

netG = getattr(models, args.archG)(image_size=args.image_size, nz=args.nz, image_depth=args.image_depth, num_filters=args.num_filters).to(device)
netG.load_state_dict(torch.load(args.netG))
for p in netG.parameters():
    p.requires_grad_(False)
netG.eval()
print netG

netI = getattr(models, args.archI)(input_size=args.nw, output_size=args.nz, hidden_layer_size=args.hidden_layer_size, num_extra_layers=args.num_extra_layers).to(device)
if args.netI:
    print "Found netI, loading..."
    netI.load_state_dict(torch.load(args.netI))
print netI

# load conditioning configurations
ij, vals = utils.load_condfile(args.condfile)
vals = utils.bin2tanh(vals)  # [0,1] -> [-1,1]
vals = torch.from_numpy(vals.astype(np.float32)).to(device)

def logp(z):  # log posterior distribution
    x = netG(z)
    lpr = -0.5*(z**2).view(z.shape[0], -1).sum(-1)  # log prior
    llh = -0.5*((x[...,ij[:,0],ij[:,1]] - vals)**2).view(x.shape[0], -1).sum(-1) / args.alpha  # log likelihood
    return llh + lpr

optimizer = optim.Adam(netI.parameters(), lr=args.lr, amsgrad=True, betas=(0.5, 0.9))
w = torch.FloatTensor(args.batch_size, args.nw).to(device)

history = utils.History(args.outdir)
plotter = utils.Plotter(args.outdir, netG, netI, args.condfile, torch.randn(64, args.nw).to(device))

for i in xrange(args.niter):

    optimizer.zero_grad()
    w.normal_(0,1)
    z = netI(w)
    z = z.view(z.shape[0], z.shape[1], 1, 1)
    err = -logp(z).mean()
    ent = utils.sample_entropy(z)
    kl = err - ent
    kl.backward()
    optimizer.step()

    history.dump(KL=kl.item(), nlogp=err.item(), entropy=ent.item())

    # --- logging
    if (i+1) % int(max(min(args.niter/20,500),100)) == 0:
        history.flush()
        plotter.flush(i+1)

        logging.info('[{}/{}] kl: {:.4f} | -logp: {:.4f} | entropy: {:.4f}'.format(
            i+1, args.niter, kl.item(), err.item(), ent.item()))
