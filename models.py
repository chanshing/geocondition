import torch.nn as nn

class DCGAN_G(nn.Module):
    """
    DCGAN architecture, " Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks" (Radford et al.)
    """
    def __init__(self, isize, nz, nc, ngf, n_extra_layers=0):
        super(DCGAN_G, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf//2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        main.add_module('initial_{0}-{1}_convt'.format(nz, cngf),
                        nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        main.add_module('initial_{0}_batchnorm'.format(cngf),
                        nn.BatchNorm2d(cngf))
        main.add_module('initial_{0}_relu'.format(cngf),
                        nn.ReLU(True))

        csize = 4
        while csize < isize//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, nc),
                        nn.ConvTranspose2d(cngf, nc, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(nc),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        output = self.main(input)
        return output


class FC_leaky(nn.Module):
    """
    Fully connected net with LeakyReLU activations
    """
    def __init__(self, idim=1, odim=1, hdim=512):
        super(FC_leaky, self).__init__()

        main = nn.Sequential(
            nn.Linear(idim, hdim),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(hdim, hdim),
            nn.LeakyReLU(0.5, inplace=True),
            nn.Linear(hdim, odim),
        )
        self.main = main

    def forward(self, x):
            output = self.main(x)
            return output
