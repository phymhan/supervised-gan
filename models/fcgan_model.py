import numpy as np
import torch
import os, sys
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import itertools
import functools
import torch.nn.functional as F

import torch.optim as optim
from tqdm import tqdm
from scipy.stats import multivariate_normal
import random


def compute_rec_error(rec, data):
    rec_error = torch.nn.BCELoss()(Variable(rec).detach(), Variable(data).detach())
    return rec_error

def im_to_original(im):
    return (im-0.5) / 0.5


class FCGANModel(BaseModel):
    def name(self):
        return 'FCGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # # Random seed
        # self.use_gpu = len(opt.gpu_ids) and torch.cuda.is_available()
        # if opt.manualSeed is None:
        #     opt.manualSeed = random.randint(1, 10000)
        # print("Random Seed: ", opt.manualSeed)
        # random.seed(opt.manualSeed)
        # np.random.seed(opt.manualSeed)
        # torch.manual_seed(opt.manualSeed)
        # if self.use_gpu:
        #     torch.cuda.manual_seed_all(opt.manualSeed)

        # parse which_channel
        idx_dict = {'r': 0, 'g': 1, 'b':2}
        self.chnl_idx_input = []
        self.chnl_idx_visual = []
        for s in opt.which_channel.split('_'):
            self.chnl_idx_visual.append([])
            for c in s:
                self.chnl_idx_input.append(idx_dict[c])
                self.chnl_idx_visual[-1].append(idx_dict[c])
            self.chnl_idx_visual[-1] = self.LongTensor(self.chnl_idx_visual[-1])
        self.chnl_idx_input = self.LongTensor(self.chnl_idx_input)
        opt.input_nc = len(self.chnl_idx_input)

        # define tensors
        self.input = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.noise = None
        self.noise_ = self.Tensor(opt.batchSize, opt.noise_nc, self.opt.noiseSize, self.opt.noiseSize)
        self.fixed_noiseA = self.Tensor(opt.batchSize, opt.noise_nc, self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1)
        self.fixed_noiseA = Variable(self.fixed_noiseA)
        self.fixed_noiseB = self.Tensor(opt.batchSize, opt.noise_nc, self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1)
        self.fixed_noiseB = Variable(self.fixed_noiseB)

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 0, opt.ngf, opt.which_model_netG, opt.norm, not opt.no_dropout,
                                      n_layers_G=opt.n_layers_G, use_residual=opt.use_residual,
                                      use_fcn=opt.noiseSize != 1, noise_nc=opt.noise_nc,
                                      add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                      upsample_mode=opt.upsample_mode, n_layers_CRN_block=opt.n_layers_CRN_block,
                                      share_label_weights=not opt.no_share_label_block_weights, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            assert (len(opt.scale_factor) == len(opt.lambda_D) == len(opt.n_layers_D))
            self.n_netD = len(opt.scale_factor)
            self.netD = []
            for scale, n_layers in zip(opt.scale_factor, opt.n_layers_D):
                self.netD.append(networks.define_D(opt.input_nc, opt.ndf, opt.which_model_netD, n_layers_D=n_layers,
                                                   norm=opt.norm, use_sigmoid=use_sigmoid, scale_factor=scale,
                                                   gpu_ids=self.gpu_ids))
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                for netD, n in zip(self.netD, range(self.n_netD)):
                    self.load_network(netD, 'D_%d' % n, opt.which_epoch)

        if self.isTrain:
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            """
            Notice all learnable parameters should be in netD.model!!!
            """
            for netD in self.netD:
                if opt.which_model_netD != 'dcgan':
                    params = itertools.chain(params, netD.model.parameters())
                else:
                    params = itertools.chain(params, netD.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            for netD in self.netD:
                networks.print_network(netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AorB = self.opt.which_direction == 'A'
        data = input['A' if AorB else 'B'].index_select(1, self.chnl_idx_input.cpu())
        self.input.resize_(data.size()).copy_(data)
        self.image_paths = input['A_paths' if AorB else 'B_paths']

    def forward(self):
        self.real = Variable(self.input)
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.fake = self.netG.forward(self.noise)

    def sample_noise(self):
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.fake = self.netG.forward(self.noise)

    # no backprop gradients
    def test(self):
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.fake = self.netG.forward(self.noise)
        print('Random check: {}'.format(self.noise.data[0, 0, 0, 0]))

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        fake = self.fake_pool.query(self.fake)
        self.loss_D_fake = 0
        for netD in self.netD:
            pred_fake = netD.forward(fake.detach())
            self.loss_D_fake += self.criterionGAN(pred_fake, False)

        # Real
        real = self.real
        self.loss_D_real = 0
        for netD in self.netD:
            pred_real = netD.forward(real)
            self.loss_D_real += self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        fake = self.fake
        self.loss_G = 0
        for netD, lambda_D in zip(self.netD, self.opt.lambda_D):
            pred_fake = netD.forward(fake)
            if not self.opt.no_logD_trick:
                self.loss_G += self.criterionGAN(pred_fake, True) * lambda_D
            else:
                self.loss_G += -self.criterionGAN(pred_fake, False) * lambda_D

        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()

        for _ in range(self.opt.n_update_D):
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()
            if self.opt.n_update_D > 1:
                self.sample_noise()

        for _ in range(self.opt.n_update_G):
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            if self.opt.n_update_G > 1:
                self.sample_noise()

    def get_current_errors(self):
        err_list = [('G_GAN', self.loss_G.data[0]),
                    ('D_real', self.loss_D_real.data[0]),
                    ('D_fake', self.loss_D_fake.data[0])]
        return OrderedDict(err_list)

    def get_current_visuals(self, save_real=False, save_as_single_image=True):
        if self.isTrain or save_real:
            if len(self.chnl_idx_visual) == 2:
                real_label = util.tensor2im(self.real.data.index_select(1, self.chnl_idx_visual[0]))
                real_image = util.tensor2im(self.real.data.index_select(1, self.chnl_idx_visual[1]))
                fake_label = util.tensor2im(self.fake.data.index_select(1, self.chnl_idx_visual[0]))
                fake_image = util.tensor2im(self.fake.data.index_select(1, self.chnl_idx_visual[1]))
                return OrderedDict([('real_label', real_label), ('real_image', real_image),
                                    ('fake_label', fake_label), ('fake_image', fake_image)])
            else:
                real = util.tensor2im(self.real.data)
                fake = util.tensor2im(self.fake.data)
                return OrderedDict([('real', real), ('fake', fake)])
        else:
            if len(self.chnl_idx_visual) == 2:
                fake_label = util.tensor2im(self.fake.data.index_select(1, self.chnl_idx_visual[0]))
                fake_image = util.tensor2im(self.fake.data.index_select(1, self.chnl_idx_visual[1]))
                return OrderedDict([('fake_label', fake_label), ('fake_image', fake_image)])
            else:
                fake = util.tensor2im(self.fake.data)
                return OrderedDict([('fake', fake)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        for netD, n in zip(self.netD, range(self.n_netD)):
            self.save_network(netD, 'D_%d' % n, label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def reconstruction(self):
        self.real = Variable(self.input)

        mvn = multivariate_normal(np.zeros(self.opt.noise_nc * self.opt.noiseSize**2),
                                  np.identity(self.opt.noise_nc * self.opt.noiseSize**2))
        num_trials = 3
        reconstructions_best = self.real.clone()
        reconstructions_best.data.zero_()
        reconstructions_best_init = self.real.clone()
        reconstructions_best_init.data.zero_()
        reconstructions_error_best = float('inf')
        ll_noise_best = float('inf')
        ll_noise_init_best = float('inf')

        for i_trial in range(num_trials):
            print('trial {0} of {1}'.format(i_trial + 1, num_trials))
            noise_init, noise = self.reconstruct_cells(self.real, self.netG, self.opt, 50, 0.1)
            recon_init = self.netG.forward(noise_init)
            recon = self.netG.forward(noise)

            # get LLN
            noise_np = noise.view(-1).data.cpu().numpy()
            ll_noise = -mvn.logpdf(noise_np)
            noise_init_np = noise_init.view(-1).data.cpu().numpy()
            ll_noise_init = -mvn.logpdf(noise_init_np)
            l2_dist = compute_rec_error((recon.data + 1) / 2.0, (self.real.data + 1) / 2.0)

            if l2_dist.data.cpu().numpy() < reconstructions_error_best:
                reconstructions_error_best = l2_dist.data.cpu().numpy()
                reconstructions_best = self.netG.forward(noise)
                reconstructions_best_init = recon_init
                ll_noise_best = ll_noise
                ll_noise_init_best = ll_noise_init
                self.noise = noise.clone()

        self.fake = reconstructions_best.clone()
        self.fake_init = reconstructions_best_init.clone()

        return reconstructions_error_best, ll_noise_best, ll_noise_init_best

    def reconstruct_cells(self, img, netG, opt, n_bfgs_iter=100, lbfgs_lr=0.1):
        noise = self.Tensor(int(opt.batchSize), opt.noise_nc, opt.noiseSize, opt.noiseSize)
        noise.normal_(0, 1)

        if len(opt.gpu_ids) > 0:
            noise = noise.cuda()

        noise = Variable(noise, requires_grad=True)
        noise_init = noise.clone()

        zero = noise.clone()
        zero.data.zero_()

        optim_input = optim.LBFGS([noise], lr=lbfgs_lr)

        label = (img + 1) / 2.0

        def closure():
            optim_input.zero_grad()
            gen_img = netG.forward(noise)
            pred = (gen_img + 1) / 2.0
            loss = torch.nn.BCELoss()(pred, label)
            # loss += torch.nn.MSELoss()(noise, zero.detach()) * 0.01
            loss.backward()
            return loss

        for _ in tqdm(range(n_bfgs_iter)):
            optim_input.step(closure)

        return noise_init, noise

    def interpolate(self, alpha):
        alpha = Variable(self.Tensor(1).fill_(alpha))
        noise = alpha * self.fixed_noiseB + (1-alpha) * self.fixed_noiseA
        self.fake = self.netG.forward(noise)
        self.real = Variable(self.input)

    def set_fixed_noise(self, which_one):
        if which_one == 'A':
            self.fixed_noiseA = self.noise
        else:
            self.fixed_noiseB = self.noise
