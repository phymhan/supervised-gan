import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
from torch.nn import Parameter
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss-1.)/2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]
    h = np.exp(-(x*x + y*y) / (2.*sigma*sigma))
    h[h < np.finfo(h.dtype).eps*h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def init_gauss_filters(nf, kw, sigma):
    filters = np.zeros((nf, nf, kw, kw))
    for i in range(nf):
        filters[i, i, :, :] = matlab_style_gauss2D((kw, kw), sigma)
    return filters


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, n_layers_G=5,
             use_residual=False, use_fcn=False, noise_nc=0, add_gaussian_noise=False, gaussian_sigma=0.1,
             n_layers_G_skip=-1, upsample_mode='convt', share_label_weights=True, n_layers_CRN_block=1, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9,
                               use_residual=use_residual, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6,
                               use_residual=use_residual, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_residual=use_residual, add_gaussian_noise=add_gaussian_noise,
                             gaussian_sigma=gaussian_sigma, num_skips=n_layers_G_skip, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                             use_residual=use_residual, add_gaussian_noise=add_gaussian_noise,
                             gaussian_sigma=gaussian_sigma, num_skips=n_layers_G_skip, gpu_ids=gpu_ids)
    elif which_model_netG == 'autoencoder':
        netG = AutoEncoder(input_nc, output_nc, n_layers_G, ngf, norm_layer=norm_layer, use_dropout=use_dropout,
                           gpu_ids=gpu_ids)
    elif which_model_netG == 'crn':
        netG = CascadedRefinementNetwork(input_nc, output_nc, noise_nc, ngf=ngf, n_layers=n_layers_G,
                                         norm_layer=norm_layer, concat_label=False, upsample_mode=upsample_mode,
                                         add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma,
                                         share_label_weights=share_label_weights, n_layers_block=n_layers_CRN_block,
                                         gpu_ids=gpu_ids)
    elif which_model_netG == 'fcgan':
        netG = FCGANGenerator(noise_nc, input_nc, ngf, n_layers=n_layers_G, norm_layer=nn.BatchNorm2d,
                              use_dropout=use_dropout, use_fcn=use_fcn, gpu_ids=gpu_ids)
    elif which_model_netG == 'fcgan_star':
        netG = FCGANGeneratorStar(noise_nc, input_nc, ngf, n_layers=n_layers_G, norm_layer=nn.BatchNorm2d,
                                  use_dropout=use_dropout, use_fcn=use_fcn, gpu_ids=gpu_ids)
    elif which_model_netG == 'dcgan':
        netG = DCGANGenerator(gpu_ids=gpu_ids, nz=noise_nc, nc=input_nc, ngf=ngf)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD, n_layers_D=3, norm='batch', use_sigmoid=False, scale_factor=1,
             num_classes=2, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   scale_factor=scale_factor, num_classes=num_classes, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid,
                                   scale_factor=scale_factor, num_classes=num_classes, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers_sep':
        netD = NLayerDiscriminatorSep(input_nc, ndf, n_layers=n_layers_D, norm_layer=norm_layer,
                                      use_sigmoid=use_sigmoid, scale_factor=scale_factor, num_classes=num_classes,
                                      gpu_ids=gpu_ids)
    elif which_model_netD == 'dcgan':
        netD = DCGANDiscriminator(gpu_ids=gpu_ids, nc=input_nc, ndf=ndf)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % which_model_netD)
    netD.apply(weights_init)
    if scale_factor > 1:
        for param in netD.gauss_filter.parameters():
            sigma = scale_factor / 2
            kw = 4 * sigma + 1
            param.data = torch.FloatTensor(init_gauss_filters(input_nc, kw, sigma))
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


class GANLossMultiClass(nn.Module):
    def __init__(self, use_lsgan=False, num_classes=3, use_gpu=False):
        super(GANLossMultiClass, self).__init__()
        assert (use_lsgan is False)
        self.Tensor = torch.cuda.LongTensor if use_gpu else torch.LongTensor
        self.num_classes = num_classes
        self.loss = nn.CrossEntropyLoss()

    def get_target_tensor(self, input, target_label):
        size_ = input.size()
        return Variable(self.Tensor(size_[0], 1, size_[2], size_[3]).fill_(target_label), requires_grad=False)

    def __call__(self, input, target_label):
        target_tensor = self.get_target_tensor(input, target_label)
        return self.loss(input.permute(0, 2, 3, 1).contiguous().view(-1, self.num_classes), target_tensor.view(-1))


class WeightedL1Loss(nn.Module):
    def __init__(self):
        super(WeightedL1Loss, self).__init__()

    def __call__(self, x, y, w=None):
        if w is not None:
            z = torch.mul(torch.abs(x - y), w)
        else:
            z = torch.abs(x - y)
        return z.mean()


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect', use_residual=False, gpu_ids=[]):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        self.use_residual = use_residual

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        if not use_residual:
            model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            y = nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            y = self.model(x)
        return nn.Tanh()(x + y) if self.use_residual else nn.Tanh()(y)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        y = x + self.conv_block(x)
        return y


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_residual=False, add_gaussian_noise=False, gaussian_sigma=0.1, num_skips=-1, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        self.use_residual = use_residual
        self.add_gauss = add_gaussian_noise
        if num_skips < 0:
            num_skips = num_downs

        # construct unet structure
        add_skip_this = True if num_skips >= 1 else False
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True,
                                             add_gaussian_noise=self.add_gauss, gaussian_sigma=gaussian_sigma,
                                             add_skip_this=add_skip_this)
        for i in range(num_downs - 5):
            add_skip_sub = add_skip_this
            add_skip_this = True if num_skips >= i + 2 else False
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, unet_block, norm_layer=norm_layer,
                                                 use_dropout=use_dropout, add_gaussian_noise=self.add_gauss,
                                                 gaussian_sigma=gaussian_sigma,
                                                 add_skip_this=add_skip_this, add_skip_sub=add_skip_sub)
        add_skip_sub = add_skip_this
        add_skip_this = True if num_skips >= num_downs - 3 else False
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, unet_block, norm_layer=norm_layer,
                                             add_gaussian_noise=self.add_gauss, gaussian_sigma=gaussian_sigma,
                                             add_skip_this=add_skip_this, add_skip_sub=add_skip_sub)
        add_skip_sub = add_skip_this
        add_skip_this = True if num_skips >= num_downs - 2 else False
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, unet_block, norm_layer=norm_layer,
                                             add_gaussian_noise=self.add_gauss, gaussian_sigma=gaussian_sigma,
                                             add_skip_this=add_skip_this, add_skip_sub=add_skip_sub)
        add_skip_sub = add_skip_this
        add_skip_this = True if num_skips >= num_downs - 1 else False
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, unet_block, norm_layer=norm_layer,
                                             add_gaussian_noise=self.add_gauss, gaussian_sigma=gaussian_sigma,
                                             add_skip_this=add_skip_this, add_skip_sub=add_skip_sub)
        nc_mult = 2 if add_skip_this else 1
        downconv = nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1)
        upconv = nn.ConvTranspose2d(ngf * nc_mult, output_nc, kernel_size=4, stride=2, padding=1)
        model = [downconv, unet_block, nn.ReLU(False), upconv]

        self.model = nn.Sequential(*model)

    def forward(self, x, noise=None, activation=nn.Tanh()):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            y = nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            y = self.model(x)
        return activation(x + y) if self.use_residual else activation(y)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d,
                 use_dropout=False, add_gaussian_noise=False, gaussian_sigma=.1, add_skip_this=True, add_skip_sub=True):
        super(UnetSkipConnectionBlock, self).__init__()
        assert (outermost is False)
        self.outermost = outermost
        self.innermost = innermost
        self.add_gauss = add_gaussian_noise
        self.gauss_sigma = gaussian_sigma
        self.add_skip_this = add_skip_this
        self.add_skip_sub = add_skip_sub

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4, stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, False)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(False)
        upnorm = norm_layer(outer_nc)

        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            nc_mult = 2 if self.add_skip_sub else 1
            upconv = nn.ConvTranspose2d(inner_nc * nc_mult, outer_nc, kernel_size=4, stride=2, padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            y = self.model(x)
            if self.add_gauss:
                Tensor = torch.cuda.FloatTensor if isinstance(x.data, torch.cuda.FloatTensor) else torch.FloatTensor
                noise = self.gauss_sigma * Variable(Tensor(y.data.size()).normal_(0, 1), requires_grad=False)
                return torch.cat([y + noise, x], 1) if self.add_skip_this else y + noise
            else:
                return torch.cat([y, x], 1) if self.add_skip_this else y


class AutoEncoder(nn.Module):
    def __init__(self, input_nc, output_nc, n_layers=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 gpu_ids=[]):
        super(AutoEncoder, self).__init__()
        self.gpu_ids = gpu_ids

        ## Encoder
        nf_mult = 1
        sequence = [
            nn.Conv2d(input_nc, ngf, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(ngf),
            nn.ReLU(True),
        ]
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if use_dropout:
                sequence += [
                    nn.Conv2d(nf_mult_prev * ngf, ngf * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                    norm_layer(ngf * nf_mult),
                    nn.Dropout(0.2),
                    nn.ReLU(True),
                ]
            else:
                sequence += [
                    nn.Conv2d(nf_mult_prev * ngf, ngf * nf_mult, kernel_size=4, stride=2, padding=1, bias=True),
                    norm_layer(ngf * nf_mult),
                    nn.ReLU(True),
                ]
        latent_nc = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(nf_mult * ngf, latent_nc, kernel_size=4, stride=2, padding=1, bias=False)
        ]

        ## Decoder
        nf_mult = min(2 ** (n_layers - 1), 8)
        sequence += [
            nn.ConvTranspose2d(latent_nc, ngf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * nf_mult),
            nn.ReLU(True),
        ]
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 8)
            if use_dropout:
                sequence += [
                    nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=4, stride=2, padding=1),
                    norm_layer(ngf * nf_mult),
                    nn.Dropout(0.5),
                    nn.ReLU(True),
                ]
            else:
                sequence += [
                    nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=4, stride=2, padding=1),
                    norm_layer(ngf * nf_mult),
                    nn.ReLU(True),
                ]
        sequence += [
            nn.ConvTranspose2d(ngf, output_nc, kernel_size=4, stride=2, padding=1, bias=False),
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x, noise=None, activation=nn.Tanh()):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            y = nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            y = self.model(x)
        return activation(y)


class FCGANGenerator(nn.Module):
    def __init__(self, noise_nc, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_fcn=False, gpu_ids=[]):
        super(FCGANGenerator, self).__init__()
        self.gpu_ids = gpu_ids
        kw = 4
        padw = 1
        nf_mult = min(2 ** (n_layers - 1), 8)
        if use_fcn:
            conv = nn.ConvTranspose2d(noise_nc, ngf * nf_mult, kernel_size=kw, stride=2, padding=1, bias=False)
        else:
            conv = nn.ConvTranspose2d(noise_nc, ngf * nf_mult, kernel_size=kw, stride=1, padding=0, bias=False)
        sequence = [
            conv,
            norm_layer(ngf * nf_mult),
            nn.ReLU(False),
        ]

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** (n_layers - n - 1), 8)
            if use_dropout:
                sequence += [
                    nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(ngf * nf_mult),
                    nn.Dropout(0.5),
                    nn.ReLU(False),
                ]
            else:
                sequence += [
                    nn.ConvTranspose2d(ngf * nf_mult_prev, ngf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                    norm_layer(ngf * nf_mult),
                    nn.ReLU(False),
                ]

        sequence += [
            nn.ConvTranspose2d(ngf, input_nc, kernel_size=kw, stride=2, padding=padw, bias=False),
            # nn.Tanh()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x, activation=nn.Tanh()):
        if self.gpu_ids and isinstance(x.data, torch.cuda.FloatTensor):
            y = nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            y = self.model(x)
        return activation(y)


class FCGANGeneratorStar(nn.Module):
    def __init__(self, noise_nc, input_nc, ngf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 use_fcn=False, gpu_ids=[]):
        super(FCGANGeneratorStar, self).__init__()
        self.gpu_ids = gpu_ids
        self.noise_nc = int(noise_nc / 2)
        assert (n_layers == 5)
        assert (use_fcn is True)
        assert (input_nc == 2)
        input_nc = 1
        #############################################################################################
        # net a:
        self.conv0a = nn.Sequential(
            nn.ConvTranspose2d(self.noise_nc, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 8),
            nn.ReLU(True),
        )
        self.conv1a = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv2a = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 4),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv3a = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 2),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv4a = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 1),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv5a = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1, input_nc, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
        )
        #############################################################################################
        # net b:
        self.conv0b = nn.Sequential(
            nn.ConvTranspose2d(self.noise_nc, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 8),
            nn.ReLU(True),
        )
        self.conv1b = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 8, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 8),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv2b = nn.Sequential(
            nn.ConvTranspose2d(ngf * 8 * 2, ngf * 4, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 4),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv3b = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4 * 2, ngf * 2, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 2),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv4b = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2 * 2, ngf * 1, kernel_size=4, stride=2, padding=1, bias=False),
            norm_layer(ngf * 1),
            # nn.Dropout(0.5),
            nn.ReLU(True),
        )
        self.conv5b = nn.Sequential(
            nn.ConvTranspose2d(ngf * 1 * 2, input_nc, kernel_size=4, stride=2, padding=1, bias=False),
            # nn.Tanh()
        )

    def forward(self, noise, activation=nn.Tanh()):
        noise1 = noise.narrow(1, 0, self.noise_nc)
        noise2 = noise.narrow(1, self.noise_nc, self.noise_nc)
        hb = self.conv0b(noise1)
        ha = self.conv0a(noise2)
        hb = self.conv1b(torch.cat([ha, hb], dim=1))
        ha = self.conv1a(ha)
        hb = self.conv2b(torch.cat([ha, hb], dim=1))
        ha = self.conv2a(ha)
        hb = self.conv3b(torch.cat([ha, hb], dim=1))
        ha = self.conv3a(ha)
        hb = self.conv4b(torch.cat([ha, hb], dim=1))
        ha = self.conv4a(ha)
        hb = self.conv5b(torch.cat([ha, hb], dim=1))
        ha = self.conv5a(ha)
        return activation(torch.cat([ha, hb], dim=1))


class CascadedRefinementNetwork(nn.Module):
    def __init__(self, input_nc, output_nc, noise_nc, ngf=64, n_layers=5, norm_layer=nn.BatchNorm2d,
                 concat_label=False, upsample_mode='convt', add_gaussian_noise=False, gaussian_sigma=0.1,
                 share_label_weights=True, n_layers_block=1, gpu_ids=[]):
        super(CascadedRefinementNetwork, self).__init__()
        self.gpu_ids = gpu_ids
        self.concat_label = concat_label
        self.tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.FloatTensor
        self.share_label_weights = share_label_weights
        assert (n_layers == 5)

        self.blockh5 = nn.Sequential(
            CrnUpsampleBlock(noise_nc + input_nc, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, ngf, n_layers=n_layers_block, norm_layer=norm_layer)
        )
        self.blockh4 = nn.Sequential(
            CrnUpsampleBlock(ngf + ngf, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, ngf, n_layers=n_layers_block, norm_layer=norm_layer)
        )
        self.blockh3 = nn.Sequential(
            CrnUpsampleBlock(ngf + ngf, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, ngf, n_layers=n_layers_block, norm_layer=norm_layer)
        )
        self.blockh2 = nn.Sequential(
            CrnUpsampleBlock(ngf + ngf, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, ngf, n_layers=n_layers_block, norm_layer=norm_layer)
        )
        self.blockh1 = nn.Sequential(
            CrnUpsampleBlock(ngf + ngf, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=add_gaussian_noise, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, ngf, n_layers=n_layers_block, norm_layer=norm_layer)
        )
        self.blockh0 = nn.Sequential(
            CrnUpsampleBlock(ngf + ngf, ngf, mode=upsample_mode, norm_layer=norm_layer,
                             add_gaussian_noise=False, gaussian_sigma=gaussian_sigma, tensor=self.tensor),
            CrnInterBlock(ngf, output_nc, n_layers=n_layers_block, norm_layer=norm_layer, outer_most=True)
        )

        if self.share_label_weights:
            self.blockl = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )
        else:
            self.blockl4 = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )
            self.blockl3 = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )
            self.blockl2 = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )
            self.blockl1 = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )
            self.blockl0 = nn.Sequential(
                nn.Conv2d(input_nc, ngf, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(ngf)
            )

    def forward(self, label, noise, activation=nn.Tanh()):
        l = torch.nn.AvgPool2d(64, 64, 0)(label)
        h = self.blockh5(torch.cat([l, noise], dim=1))

        l = torch.nn.AvgPool2d(32, 32, 0)(label)
        l = self.blockl(l) if self.share_label_weights else self.blockl4(l)
        h = self.blockh4(torch.cat([l, h], dim=1))

        l = torch.nn.AvgPool2d(16, 16, 0)(label)
        l = self.blockl(l) if self.share_label_weights else self.blockl3(l)
        h = self.blockh3(torch.cat([l, h], dim=1))

        l = torch.nn.AvgPool2d(8, 8, 0)(label)
        l = self.blockl(l) if self.share_label_weights else self.blockl2(l)
        h = self.blockh2(torch.cat([l, h], dim=1))

        l = torch.nn.AvgPool2d(4, 4, 0)(label)
        l = self.blockl(l) if self.share_label_weights else self.blockl1(l)
        h = self.blockh1(torch.cat([l, h], dim=1))

        l = torch.nn.AvgPool2d(2, 2, 0)(label)
        l = self.blockl(l) if self.share_label_weights else self.blockl0(l)
        h = self.blockh0(torch.cat([l, h], dim=1))

        return torch.cat([label, activation(h)], dim=1) if self.concat_label else activation(h)


class CrnUpsampleBlock(nn.Module):
    def __init__(self, input_nc, output_nc, mode='convt', norm_layer=nn.BatchNorm2d, add_gaussian_noise=False,
                 gaussian_sigma=0.1, tensor=torch.FloatTensor):
        super(CrnUpsampleBlock, self).__init__()
        self.tensor = tensor
        self.add_gauss = add_gaussian_noise
        self.gauss_sigma = gaussian_sigma
        if mode == 'convt':
            self.model = nn.Sequential(
                nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1, bias=False),
                norm_layer(output_nc),
            )
        elif mode == 'bilinear':
            self.model = nn.Sequential(
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
                nn.Upsample(scale_factor=2, mode='bilinear'),
                norm_layer(output_nc),
            )
        else:
            raise NotImplementedError('UpsampleBlock mode [%s] is not recognized' % mode)

    def forward(self, x):
        y = self.model(x)
        if self.add_gauss:
            return y + self.gauss_sigma * Variable(self.tensor(y.data.size()).normal_(0, 1), requires_grad=False)
        else:
            return y


class CrnInterBlock(nn.Module):
    def __init__(self, input_nc, output_nc, n_layers=1, norm_layer=nn.BatchNorm2d, outer_most=False):
        super(CrnInterBlock, self).__init__()
        sequence = None
        for i in range(1, n_layers):
            block = [
                nn.ReLU(False),
                nn.Conv2d(input_nc, input_nc, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(input_nc),
            ]
            sequence = block if sequence is None else sequence + block
        if not outer_most:
            block = [
                nn.ReLU(False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True),
                norm_layer(output_nc),
            ]
        else:
            block = [
                nn.ReLU(False),
                nn.Conv2d(input_nc, output_nc, kernel_size=3, stride=1, padding=1, bias=True)
            ]
        sequence = block if sequence is None else sequence + block

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, scale_factor=1,
                 num_classes=2, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
        self.gauss_filter = None
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        logit_nc = 1 if num_classes == 2 else num_classes
        if scale_factor > 1:
            sigma_ = scale_factor / 2
            kw_ = int(4 * sigma_ + 1)
            self.gauss_filter = nn.Sequential(
                nn.Conv2d(input_nc, input_nc, kernel_size=kw_, stride=1, padding=2 * sigma_, bias=False),
                nn.AvgPool2d(kernel_size=1, stride=scale_factor)
            )
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, False)
        ]
        nf_mult = 1
        # nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]
        sequence += [nn.Conv2d(ndf * nf_mult, logit_nc, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        if self.gauss_filter is not None:
            x = self.gauss_filter(x)
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, x, self.gpu_ids)
        else:
            return self.model(x)


# EncoderSep: encoder that separates input's channel and feeds them into separate sub-networks
class NLayerDiscriminatorSep(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, scale_factor=1,
                 num_classes=2, gpu_ids=[]):
        super(NLayerDiscriminatorSep, self).__init__()
        self.gpu_ids = gpu_ids
        self.gauss_filter = None
        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        logit_nc = 1 if num_classes == 2 else num_classes
        n_sep = 2

        assert (input_nc == 3)
        if scale_factor > 1:
            sigma_ = scale_factor / 2
            kw_ = int(4 * sigma_ + 1)
            self.gauss_filter = nn.Sequential(
                nn.Conv2d(input_nc, input_nc, kernel_size=kw_, stride=1, padding=2 * sigma_, bias=False),
                nn.AvgPool2d(kernel_size=1, stride=scale_factor)
            )

        # netA:
        sequence = [
            nn.Conv2d(2, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, False)
        ]
        nf_mult = 1
        for n in range(1, n_sep):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]
        self.netA = nn.Sequential(*sequence)

        # netB:
        sequence = [
            nn.Conv2d(1, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, False)
        ]
        nf_mult = 1
        for n in range(1, n_sep):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]
        self.netB = nn.Sequential(*sequence)

        # fc:
        nf_mult = 2 * nf_mult
        sequence = None
        for n in range(n_sep, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            block = [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]
            sequence = block if sequence is None else sequence + block
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        block = [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, False)
        ]
        sequence = block if sequence is None else sequence + block
        sequence += [nn.Conv2d(ndf * nf_mult, logit_nc, kernel_size=kw, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        if self.gauss_filter is not None:
            x = self.gauss_filter(x)
        x_A = x.narrow(1, 0, 2)
        x_B = x.narrow(1, 2, 1)

        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            y_A = nn.parallel.data_parallel(self.netA, x_A, self.gpu_ids)
            y_B = nn.parallel.data_parallel(self.netB, x_B, self.gpu_ids)
        else:
            y_A = self.netA(x_A)
            y_B = self.netA(x_B)
        y = torch.cat([y_A, y_B], dim=1)
        return self.model(y)


## Original DCGAN:
# class DCGANGenerator(nn.Module):
#     def __init__(self, gpu_ids=[], nz=100, nc=3, ngf=64):
#         super(DCGANGenerator, self).__init__()
#         self.ngpu = len(gpu_ids)
#         self.model = nn.Sequential(
#             # input is Z, going into a convolution
#             nn.ConvTranspose2d(     nz, ngf * 8, 4, 1, 0, bias=False),
#             nn.BatchNorm2d(ngf * 8),
#             nn.ReLU(True),
#             # state size. (ngf*8) x 4 x 4
#             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 4),
#             nn.ReLU(True),
#             # state size. (ngf*4) x 8 x 8
#             nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf * 2),
#             nn.ReLU(True),
#             # state size. (ngf*2) x 16 x 16
#             nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ngf),
#             nn.ReLU(True),
#             # state size. (ngf) x 32 x 32
#             nn.ConvTranspose2d(    ngf,      nc, 4, 2, 1, bias=False),
#             nn.Tanh()
#             # state size. (nc) x 64 x 64
#         )
#
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
#         else:
#             output = self.model(input)
#         return output
#
#
# class DCGANDiscriminator(nn.Module):
#     def __init__(self, gpu_ids=[], nc=3, ndf=64):
#         super(DCGANDiscriminator, self).__init__()
#         self.ngpu = len(gpu_ids)
#         self.model = nn.Sequential(
#             # input is (nc) x 64 x 64
#             nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf) x 32 x 32
#             nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 2),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*2) x 16 x 16
#             nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 4),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*4) x 8 x 8
#             nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
#             nn.BatchNorm2d(ndf * 8),
#             nn.LeakyReLU(0.2, inplace=True),
#             # state size. (ndf*8) x 4 x 4
#             nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, input):
#         if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
#             output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
#         else:
#             output = self.model(input)
#         return output.view(-1, 1).squeeze(1)


## Modified DCGAN:
class DCGANGenerator(nn.Module):
    def __init__(self, gpu_ids=[], nz=100, nc=3, ngf=64):
        super(DCGANGenerator, self).__init__()
        self.ngpu = len(gpu_ids)
        self.model = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, int(ngf / 2), 4, 2, 1, bias=False),
            nn.BatchNorm2d(int(ngf / 2)),
            nn.ReLU(True),
            # state size. (nc) x 64 x 64

            nn.ConvTranspose2d(int(ngf / 2), nc, 4, 2, 1, bias=False),
            # state size. (nc) x 128 x 128

            # nn.ConvTranspose2d(int(ngf / 2), int(ngf / 4), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ngf / 4)),
            # nn.ReLU(True),
            # # state size. (nc) x 128 x 128
            # nn.ConvTranspose2d(int(ngf / 4), nc, 4, 2, 1, bias=False),
            # # state size. (nc) x 256 x 256

            # nn.ConvTranspose2d(int(ngf / 2), int(ngf / 4), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ngf / 4)),
            # nn.ReLU(True),
            # # state size. (nc) x 128 x 128
            # nn.ConvTranspose2d(int(ngf / 4), int(ngf / 8), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ngf / 8)),
            # nn.ReLU(True),
            # # state size. (nc) x 256 x 256
            # nn.ConvTranspose2d(int(ngf / 8), nc, 4, 2, 1, bias=False),
            # # state size. (nc) x 512 x 512

            nn.Tanh()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output


class DCGANDiscriminator(nn.Module):
    def __init__(self, gpu_ids=[], nc=3, ndf=64):
        super(DCGANDiscriminator, self).__init__()
        self.ngpu = len(gpu_ids)
        self.model = nn.Sequential(
            # # input is (nc) x 512 x 512
            # nn.Conv2d(nc, int(ndf / 8), 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf/8) x 256 x 256
            # nn.Conv2d(int(ndf / 8), int(ndf / 4), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ndf / 4)),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf/4) x 128 x 128
            # nn.Conv2d(int(ndf / 4), int(ndf / 2), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ndf / 2)),
            # nn.LeakyReLU(0.2, inplace=True),

            # # input size. (nc) x 256 x 256
            # nn.Conv2d(nc, int(ndf / 4), 4, 2, 1, bias=False),
            # nn.LeakyReLU(0.2, inplace=True),
            # # state size. (ndf/4) x 128 x 128
            # nn.Conv2d(int(ndf / 4), int(ndf / 2), 4, 2, 1, bias=False),
            # nn.BatchNorm2d(int(ndf / 2)),
            # nn.LeakyReLU(0.2, inplace=True),

            # input is (nc) x 128 x 128
            nn.Conv2d(nc, int(ndf / 2), 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # state size. (ndf/2) x 64 x 64
            nn.Conv2d(int(ndf / 2), ndf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.model, input, range(self.ngpu))
        else:
            output = self.model(input)
        return output.view(-1, 1).squeeze(1)
