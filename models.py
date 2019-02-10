import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class DCGAN_G(nn.Module):
    """
    DCGAN_G architecture, " Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks" (Radford et al.)
    """
    def __init__(self, image_size, nz, image_depth, num_filters, num_extra_layers=0):
        super(DCGAN_G, self).__init__()
        assert image_size % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = num_filters//2, 4
        while tisize != image_size:
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
        while csize < image_size//2:
            main.add_module('pyramid_{0}-{1}_convt'.format(cngf, cngf//2),
                            nn.ConvTranspose2d(cngf, cngf//2, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(cngf//2),
                            nn.BatchNorm2d(cngf//2))
            main.add_module('pyramid_{0}_relu'.format(cngf//2),
                            nn.ReLU(True))
            cngf = cngf // 2
            csize = csize * 2

        # Extra layers
        for t in range(num_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cngf),
                            nn.Conv2d(cngf, cngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cngf),
                            nn.BatchNorm2d(cngf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cngf),
                            nn.ReLU(True))

        main.add_module('final_{0}-{1}_convt'.format(cngf, image_depth),
                        nn.ConvTranspose2d(cngf, image_depth, 4, 2, 1, bias=False))
        main.add_module('final_{0}_tanh'.format(image_depth),
                        nn.Tanh())
        self.main = main

    def forward(self, x):
        return self.main(x)

class DCGAN_D(nn.Module):
    """
    DCGAN_D architecture, " Unsupervised Representation Learning with
    Deep Convolutional Generative Adversarial Networks" (Radford et al.)
    """
    def __init__(self, image_size, image_depth, num_filters, num_extra_layers=0):
        super(DCGAN_D, self).__init__()
        assert image_size % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is nc x isize x isize
        main.add_module('initial_conv_{0}-{1}'.format(image_depth, num_filters),
                        nn.Conv2d(image_depth, num_filters, 4, 2, 1, bias=False))
        main.add_module('initial_relu_{0}'.format(num_filters),
                        nn.LeakyReLU(0.2, inplace=True))
        csize, cndf = image_size / 2, num_filters

        # Extra layers
        for t in range(num_extra_layers):
            main.add_module('extra-layers-{0}_{1}_conv'.format(t, cndf),
                            nn.Conv2d(cndf, cndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}_{1}_batchnorm'.format(t, cndf),
                            nn.BatchNorm2d(cndf))
            main.add_module('extra-layers-{0}_{1}_relu'.format(t, cndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while csize > 4:
            in_feat = cndf
            out_feat = cndf * 2
            main.add_module('pyramid_{0}-{1}_conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid_{0}_batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid_{0}_relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cndf = cndf * 2
            csize = csize / 2

        # state size. K x 4 x 4
        main.add_module('final_{0}-{1}_conv'.format(cndf, 1),
                        nn.Conv2d(cndf, 1, 4, 1, 0, bias=False))
        self.main = main

    def forward(self, x):

        return self.main(x).view(-1,1)

class FC_leaky(nn.Module):
    """
    Fully connected net with LeakyReLU activations
    """
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=512, a=0.5):
        super(FC_leaky, self).__init__()

        main = nn.Sequential(
            nn.Linear(input_size, hidden_layer_size),
            nn.LeakyReLU(a, inplace=True),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(a, inplace=True),
            nn.Linear(hidden_layer_size, hidden_layer_size),
            nn.LeakyReLU(a, inplace=True),
            nn.Linear(hidden_layer_size, output_size),
        )
        self.main = main

        for m in self.main.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=a, mode='fan_in', nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.main(x)

class SELU(nn.Module):
    def __init__(self):
        super(SELU, self).__init__()
        self.alpha = 1.6732632423543772848170429916717
        self.scale = 1.0507009873554804934193349852946

    def forward(self, x):
        return self.scale * F.elu(x, self.alpha)

class FC_selu(nn.Module):
    """
    Fully connected with selu activations
    """
    def __init__(self, input_size=1, output_size=1, hidden_layer_size=512, num_extra_layers=2):
        super(FC_selu, self).__init__()

        main = nn.Sequential()
        main.add_module('layer-0', nn.Sequential(nn.Linear(input_size, hidden_layer_size, bias=False), SELU()))
        for i in range(num_extra_layers):
            main.add_module('layer-{}'.format(i+1), nn.Sequential(nn.Linear(hidden_layer_size, hidden_layer_size, bias=False), SELU()))
        main.add_module('layer-{}'.format(num_extra_layers+1), nn.Sequential(nn.Linear(hidden_layer_size, output_size, bias=True)))

        for m in main.modules():
            if isinstance(m, nn.Linear):
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
                nn.init.normal_(m.weight, 0, np.sqrt(1./fan_in))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.main = main

    def forward(self, x):
        return self.main(x)
