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
import random


class CGANModel(BaseModel):
    def name(self):
        return 'cGANModel'

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

        # parse which_channel: eg: 'rg_b' means rg --> b
        idx_dict = {'r': 0, 'g': 1, 'b': 2}
        self.chnl_idx_input = []
        for s in opt.which_channel.split('_'):
            self.chnl_idx_input.append([])
            for c in s:
                self.chnl_idx_input[-1].append(idx_dict[c])
            self.chnl_idx_input[-1] = self.LongTensor(self.chnl_idx_input[-1])
        assert (len(self.chnl_idx_input) == 2)
        opt.input_nc = len(self.chnl_idx_input[0])
        opt.output_nc = len(self.chnl_idx_input[1])

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc, opt.fineSize, opt.fineSize)
        self.noise = None
        self.noise_ = self.Tensor(opt.batchSize, opt.noise_nc, self.opt.noiseSize, self.opt.noiseSize)

        if 'bilinear' in opt.transform_1to2:
            sc = int(opt.transform_1to2.split('_')[1])
            self.transform = torch.nn.Upsample(scale_factor=sc, mode='bilinear')
            self.transform_inverse = torch.nn.AvgPool2d(kernel_size=sc, stride=sc)
        else:
            self.transform = lambda x: x
            self.transform_inverse = lambda x: x

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.which_model_netG, opt.norm,
                                      not opt.no_dropout, n_layers_G=self.opt.n_layers_G, use_residual=opt.use_residual,
                                      use_fcn=opt.noiseSize != 1, noise_nc=opt.noise_nc,
                                      add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                      upsample_mode=opt.upsample_mode, n_layers_CRN_block=opt.n_layers_CRN_block,
                                      share_label_weights=not opt.no_share_label_block_weights,
                                      n_layers_G_skip=opt.n_layers_G_skip, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            assert (len(self.opt.scale_factor) == len(self.opt.lambda_D) == len(self.opt.n_layers_D))
            self.n_netD = len(self.opt.scale_factor)
            self.netD = []
            if opt.no_cgan:
                netD_input_nc = opt.output_nc
            else:
                netD_input_nc = opt.output_nc + opt.input_nc
            for scale, n_layers in zip(self.opt.scale_factor, self.opt.n_layers_D):
                self.netD.append(networks.define_D(netD_input_nc, opt.ndf, opt.which_model_netD, n_layers_D=n_layers,
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
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = networks.WeightedL1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            """
            Notice all learnable parameters should be in netD.model!!!
            """
            for netD in self.netD:
                if opt.which_model_netD == 'dcgan':
                    params = itertools.chain(params, netD.parameters())
                elif opt.which_model_netD == 'n_layers_sep':
                    params = itertools.chain(params, netD.model.parameters(),
                                             netD.netA.parameters(), netD.netB.parameters())
                else:
                    params = itertools.chain(params, netD.model.parameters())
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            for netD in self.netD:
                networks.print_network(netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        if self.opt.dataset_mode == 'aligned':
            input_A = input['A' if AtoB else 'B'].index_select(1, self.chnl_idx_input[0].cpu())
            input_B = input['B' if AtoB else 'A'].index_select(1, self.chnl_idx_input[1].cpu())
        elif self.opt.dataset_mode == 'single':
            input_A = input['A'].index_select(1, self.chnl_idx_input[0].cpu())
            input_B = input['A'].index_select(1, self.chnl_idx_input[1].cpu())
        else:
            raise NotImplementedError('Dataset mode [%s] is not recognized' % self.opt.dataset_mode)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.fake_B = self.netG.forward(self.real_A, self.noise)

    def sample_noise(self):
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.fake_B = self.netG.forward(self.real_A, self.noise)

    # no backprop gradients
    def test(self):
        self.noise = Variable(self.noise_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                  self.opt.noiseSize, self.opt.noiseSize).normal_(0, 1))
        self.real_A = self.transform(Variable(self.input_A, volatile=True))
        # self.real_B = Variable(self.input_B, volatile=True)
        self.fake_B = self.netG.forward(self.real_A, self.noise)
        print('Random check: {}'.format(self.noise.data[0, 0, 0, 0]))

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        if self.opt.no_cgan:
            fake = self.fake_pool.query(self.fake_B)
        else:
            fake = self.fake_pool.query(torch.cat((self.real_A, self.fake_B), 1))
        self.loss_D_fake = 0
        for netD in self.netD:
            pred_fake = netD.forward(fake.detach())
            self.loss_D_fake += self.criterionGAN(pred_fake, False)

        # Real
        if self.opt.no_cgan:
            real = self.real_B
        else:
            real = torch.cat((self.real_A, self.real_B), 1)
        self.loss_D_real = 0
        for netD in self.netD:
            pred_real = netD.forward(real)
            self.loss_D_real += self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        if self.opt.no_cgan:
            fake = self.fake_B
        else:
            fake = torch.cat((self.real_A, self.fake_B), 1)
        self.loss_G = 0
        for netD, lambda_D in zip(self.netD, self.opt.lambda_D):
            pred_fake = netD.forward(fake)
            if not self.opt.no_logD_trick:
                self.loss_G += self.criterionGAN(pred_fake, True) * lambda_D
            else:
                self.loss_G += -self.criterionGAN(pred_fake, False) * lambda_D

        # L1 loss
        if self.opt.weights is None:
            weight = None
        else:
            # when using weighted L1, must be label --> image
            weight = Variable(self.Tensor(self.opt.batchSize, 1, self.opt.fineSize, self.opt.fineSize).fill_(1.0),
                              requires_grad=False)
            real_A = (self.real_A.detach() + 1) / 2
            for i in range(len(self.opt.weights)):
                weight += real_A.narrow(1, i, 1) * (self.opt.weights[i] - 1.0)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B, weight) * self.opt.lambda_A
        self.loss_G += self.loss_G_L1
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
                    ('G_L1', self.loss_G_L1.data[0]),
                    ('D_real', self.loss_D_real.data[0]),
                    ('D_fake', self.loss_D_fake.data[0])]
        return OrderedDict(err_list)

    def get_current_visuals(self, save_as_single_image=False):
        # visualize label and image (channel: (red, green): label; (blue): image)
        if self.isTrain:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            real_B = util.tensor2im(self.real_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])
        elif save_as_single_image:
            AB = util.tensor2im(torch.cat([self.real_A.data, self.fake_B.data], dim=1))
            return OrderedDict([('AB', AB)])
        else:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, gpu_ids=self.gpu_ids)
        for netD, n in zip(self.netD, range(self.n_netD)):
            self.save_network(netD, 'D_%d' % n, label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr
