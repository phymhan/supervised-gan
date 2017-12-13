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


class TwoStageCycleModel(BaseModel):
    def name(self):
        return 'TwoStageCycleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

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
        self.noise1 = None
        self.noise1_ = self.Tensor(opt.batchSize, opt.noise_nc1, self.opt.noiseSize1, self.opt.noiseSize1)
        self.noise2 = None
        self.noise2_ = self.Tensor(opt.batchSize, opt.noise_nc2, self.opt.noiseSize2, self.opt.noiseSize2)
        if self.isTrain and opt.use_fixed_noise1:
            self.fixed_noise1 = Variable(self.Tensor(opt.noise_pool_size, opt.noise_nc1,
                                                     self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))

        # load/define networks
        self.netG1 = networks.define_G(opt.input_nc, 0, opt.ngf1, opt.which_model_netG1, opt.norm,
                                       not opt.no_dropout1, n_layers_G=self.opt.n_layers_G1, use_residual=False,
                                       use_fcn=opt.noiseSize1 != 1, noise_nc=opt.noise_nc1,
                                       add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                       upsample_mode=opt.upsample_mode1, n_layers_CRN_block=opt.n_layers_CRN_block1,
                                       share_label_weights=not opt.no_share_label_block_weights1, gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf2, opt.which_model_netG2, opt.norm,
                                       not opt.no_dropout2, n_layers_G=self.opt.n_layers_G2,
                                       use_residual=opt.use_residual2, use_fcn=False, noise_nc=opt.noise_nc2,
                                       add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                       upsample_mode=opt.upsample_mode2, n_layers_CRN_block=opt.n_layers_CRN_block2,
                                       share_label_weights=not opt.no_share_label_block_weights2, gpu_ids=self.gpu_ids)
        self.netF2 = networks.define_G(opt.output_nc, opt.input_nc, opt.nff2, opt.which_model_netF2, opt.norm,
                                       not opt.no_dropout2, n_layers_G=self.opt.n_layers_F2,
                                       use_residual=opt.use_residual2, use_fcn=False, noise_nc=opt.noise_nc2,
                                       add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                       upsample_mode=opt.upsample_mode2, n_layers_CRN_block=opt.n_layers_CRN_block2,
                                       share_label_weights=not opt.no_share_label_block_weights2, gpu_ids=self.gpu_ids)
        if 'bilinear' in opt.transform_1to2:
            sc = int(opt.transform_1to2.split('_')[1])
            self.transform = torch.nn.Upsample(scale_factor=sc, mode='bilinear')
            self.transform_inverse = torch.nn.AvgPool2d(kernel_size=sc, stride=sc)
        else:
            self.transform = lambda x: x
            self.transform_inverse = lambda x: x
        if self.isTrain:
            assert (len(self.opt.scale_factor1) == len(self.opt.lambda_D1) == len(self.opt.n_layers_D1))
            assert (len(self.opt.scale_factor2) == len(self.opt.lambda_D2) == len(self.opt.n_layers_D2))
            self.n_netD1 = len(self.opt.scale_factor1)
            self.n_netD2 = len(self.opt.scale_factor2)

            self.netD1 = []
            use_sigmoid = opt.no_lsgan1
            for scale, n_layers in zip(self.opt.scale_factor1, self.opt.n_layers_D1):
                netD_input_nc = opt.input_nc
                self.netD1.append(networks.define_D(netD_input_nc, opt.ndf1, opt.which_model_netD1, n_layers_D=n_layers,
                                                    norm=opt.norm, use_sigmoid=use_sigmoid, scale_factor=scale,
                                                    num_classes=2, gpu_ids=self.gpu_ids))
            self.netD2 = []
            use_sigmoid = opt.no_lsgan2
            num_classes = 3 if opt.use_multi_class_GAN else 2
            if opt.no_cgan:
                netD_input_nc = opt.output_nc
            else:
                netD_input_nc = opt.output_nc + opt.input_nc
            for scale, n_layers in zip(self.opt.scale_factor2, self.opt.n_layers_D2):
                self.netD2.append(networks.define_D(netD_input_nc, opt.ndf2, opt.which_model_netD2, n_layers_D=n_layers,
                                                    norm=opt.norm, use_sigmoid=use_sigmoid, scale_factor=scale,
                                                    num_classes=num_classes, gpu_ids=self.gpu_ids))
        if self.isTrain and opt.sequential_train:
            if 'G1' in self.opt.which_model_to_load:
                self.load_network(self.netG1, 'G1', opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'G2' in self.opt.which_model_to_load:
                self.load_network(self.netG2, 'G2', opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'F2' in self.opt.which_model_to_load:
                self.load_network(self.netF2, 'F2', opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'D1' in self.opt.which_model_to_load:
                for netD, n in zip(self.netD1, range(self.n_netD1)):
                    self.load_network(netD, 'D1_%d' % n, opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'D2' in self.opt.which_model_to_load:
                for netD, n in zip(self.netD2, range(self.n_netD2)):
                    self.load_network(netD, 'D2_%d' % n, opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG1, 'G1', opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            self.load_network(self.netF2, 'F2', opt.which_epoch)
            if self.isTrain:
                for netD, n in zip(self.netD1, range(self.n_netD1)):
                    self.load_network(netD, 'D1_%d' % n, opt.which_epoch)
                for netD, n in zip(self.netD2, range(self.n_netD2)):
                    self.load_network(netD, 'D2_%d' % n, opt.which_epoch)

        if self.isTrain:
            self.fake_pool1 = ImagePool(opt.pool_size)
            if not opt.use_multi_class_GAN:
                self.fake_pool2 = ImagePool(opt.pool_size)
            else:
                self.fake_pool2_1 = ImagePool(opt.pool_size)
                self.fake_pool2_2 = ImagePool(opt.pool_size)
            if opt.use_fixed_noise1:
                self.noise_pool1 = ImagePool(opt.noise_pool_size)
                self.noise_pool1.query(self.fixed_noise1)
            self.old_lr = opt.lr
            self.old_lr1 = opt.lr1
            self.old_lr2 = opt.lr2

            # define loss functions
            self.criterionGAN1 = networks.GANLoss(use_lsgan=not opt.no_lsgan1, tensor=self.Tensor)
            if not opt.use_multi_class_GAN:
                self.criterionGAN2 = networks.GANLoss(use_lsgan=not opt.no_lsgan2, tensor=self.Tensor)
            else:
                self.criterionGAN2 = networks.GANLossMultiClass(use_lsgan=not opt.no_lsgan2, use_gpu=len(self.gpu_ids),
                                                                num_classes=3 if opt.use_multi_class_GAN else 2)
            # self.criterionL1 = torch.nn.L1Loss()
            self.criterionL1 = networks.WeightedL1Loss()

            # define backward function for D2
            if not opt.use_multi_class_GAN:
                self.backward_D2 = self.backward_D2_binary
            else:
                self.backward_D2 = self.backward_D2_multiclass

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'name': 'G1', 'params': self.netG1.parameters(), 'lr': opt.lr1},
                                                 {'name': 'G2', 'params': self.netG2.parameters(), 'lr': opt.lr2},
                                                 {'name': 'F2', 'params': self.netF2.parameters(), 'lr': opt.lr2}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            for netD in self.netD1:
                params = itertools.chain(params, netD.model.parameters())
            self.optimizer_D1 = torch.optim.Adam(params, lr=opt.lr1, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            for netD in self.netD2:
                if opt.which_model_netD == 'dcgan':
                    params = itertools.chain(params, netD.parameters())
                elif opt.which_model_netD == 'n_layers_sep':
                    params = itertools.chain(params, netD.model.parameters(),
                                             netD.netA.parameters(), netD.netB.parameters())
                else:
                    params = itertools.chain(params, netD.model.parameters())
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr2, betas=(opt.beta1, 0.999))

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG1)
        networks.print_network(self.netG2)
        networks.print_network(self.netF2)
        if self.isTrain:
            for netD in self.netD1:
                networks.print_network(netD)
            for netD in self.netD2:
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
        if self.opt.use_fixed_noise1:
            self.noise1 = self.noise_pool1.sample(self.opt.batchSize)
        else:
            self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                        self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
        self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
                                                    self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        self.fake_A = self.netG1.forward(self.noise1)
        self.fake_A_from_real_B = self.netF2.forward(self.real_B, self.noise2)
        self.fake_B_from_real_A = self.netG2.forward(self.real_A, self.noise2)
        if not self.opt.detach_G1_from_G2_x:
            self.fake_B_from_fake_A = self.netG2.forward(self.transform(self.fake_A), self.noise2)
        else:
            self.fake_B_from_fake_A = self.netG2.forward(self.transform(self.fake_A.detach()), self.noise2)
        self.recon_real_A = self.netF2.forward(self.fake_B_from_real_A, self.noise2)
        self.recon_fake_A = self.netF2.forward(self.fake_B_from_fake_A, self.noise2)

    def sample_noise(self):
        self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                    self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
        self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
                                                    self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        self.fake_A = self.netG1.forward(self.noise1)
        self.fake_A_from_real_B = self.netF2.forward(self.real_B, self.noise2)
        self.fake_B_from_real_A = self.netG2.forward(self.real_A, self.noise2)
        if not self.opt.detach_G1_from_G2_x:
            self.fake_B_from_fake_A = self.netG2.forward(self.transform(self.fake_A), self.noise2)
        else:
            self.fake_B_from_fake_A = self.netG2.forward(self.transform(self.fake_A.detach()), self.noise2)
        self.recon_real_A = self.netF2.forward(self.fake_B_from_real_A, self.noise2)
        self.recon_fake_A = self.netF2.forward(self.fake_B_from_fake_A, self.noise2)

    # no backprop gradients
    def test(self):
        self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                    self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
        self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
                                                    self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        # self.real_A = Variable(self.input_A, volatile=True)
        # self.real_B = Variable(self.input_B, volatile=True)
        self.fake_A = self.netG1.forward(self.noise1)
        # self.fake_B_from_real_A = self.netG2.forward(self.real_A, self.noise2)
        self.fake_B_from_fake_A = self.netG2.forward(self.transform(self.fake_A), self.noise2)
        print('Random check: {}, {}'.format(self.noise1.data[0, 0, 0, 0], self.noise2.data[0, 0, 0, 0]))

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D1(self):
        # Fake
        fake = self.fake_pool1.query(self.fake_A)
        self.loss_D1_fake = 0
        for netD in self.netD1:
            pred_fake = netD.forward(fake.detach())
            self.loss_D1_fake += self.criterionGAN1(pred_fake, False)

        # Real
        real = self.transform_inverse(self.real_A)
        self.loss_D1_real = 0
        for netD in self.netD1:
            pred_real = netD.forward(real)
            self.loss_D1_real += self.criterionGAN1(pred_real, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5
        self.loss_D1.backward()

    def backward_D2_binary(self):
        # Fake
        self.loss_D2_fake = 0
        num_fake_pairs = 0
        if 'real_fake' in self.opt.GAN_losses_D2:
            if self.opt.no_cgan:
                fake = self.fake_pool2.query(self.fake_B_from_real_A)
            else:
                fake = self.fake_pool2.query(torch.cat([self.real_A, self.fake_B_from_real_A], 1))
            num_fake_pairs += 1
            for netD in self.netD2:
                pred_fake = netD.forward(fake.detach())
                self.loss_D2_fake += self.criterionGAN2(pred_fake, False)
        if 'fake_fake' in self.opt.GAN_losses_D2:
            if self.opt.no_cgan:
                fake = self.fake_pool2.query(self.fake_B_from_fake_A)
            else:
                fake = self.fake_pool2.query(torch.cat([self.transform(self.fake_A), self.fake_B_from_fake_A], 1))
            num_fake_pairs += 1
            for netD in self.netD2:
                pred_fake = netD.forward(fake.detach())
                self.loss_D2_fake += self.criterionGAN2(pred_fake, False)
        self.loss_D2_fake /= num_fake_pairs

        # Real
        if self.opt.no_cgan:
            real = self.real_B
        else:
            real = torch.cat([self.real_A, self.real_B], 1)
        self.loss_D2_real = 0
        for netD in self.netD2:
            pred_real = netD.forward(real)
            self.loss_D2_real += self.criterionGAN2(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        self.loss_D2.backward()

    def backward_D2_multiclass(self):
        # 0: (real_A, real_B)
        if self.opt.no_cgan:
            real = self.real_B
        else:
            real = torch.cat([self.real_A, self.real_B], 1)
        self.loss_D2_0 = 0
        for netD in self.netD2:
            pred_real = netD.forward(real)
            self.loss_D2_0 += self.criterionGAN2(pred_real, 0)

        # 1: (real_A, fake_B)
        if self.opt.no_cgan:
            fake = self.fake_pool2_1.query(self.fake_B_from_real_A)
        else:
            fake = self.fake_pool2_1.query(torch.cat([self.real_A, self.fake_B_from_real_A], 1))
        self.loss_D2_1 = 0
        for netD in self.netD2:
            pred_fake = netD.forward(fake.detach())
            self.loss_D2_1 += self.criterionGAN2(pred_fake, 1)

        # 2: (fake_A, fake_B)
        if self.opt.no_cgan:
            fake = self.fake_pool2_2.query(self.fake_B_from_fake_A)
        else:
            fake = self.fake_pool2_2.query(torch.cat([self.transform(self.fake_A), self.fake_B_from_fake_A], 1))
        self.loss_D2_2 = 0
        for netD in self.netD2:
            pred_fake = netD.forward(fake.detach())
            self.loss_D2_2 += self.criterionGAN2(pred_fake, 2)

        # Combined loss
        self.loss_D2 = (self.loss_D2_0 + self.loss_D2_1 + self.loss_D2_2) / 3
        self.loss_D2.backward()

    def backward_G(self):
        """
        think of G1 and G2 as sub-nets of a big network (GAN losses are combined), so G1 and G2 are updated
        simultaneously.
        """
        # loss of G1
        self.loss_G1_GAN = 0
        for netD, lambda_D in zip(self.netD1, self.opt.lambda_D1):
            pred_fake = netD.forward(self.fake_A)
            if not self.opt.no_logD_trick:
                self.loss_G1_GAN += self.criterionGAN1(pred_fake, True) * lambda_D
            else:
                self.loss_G1_GAN += -self.criterionGAN1(pred_fake, False) * lambda_D

        # loss of G2
        flipped_label = 0 if self.opt.use_multi_class_GAN else True
        self.loss_G2_GAN = 0
        num_fake_pairs = 0
        if 'real_fake' in self.opt.GAN_losses_G2:
            if self.opt.no_cgan:
                fake = self.fake_B_from_real_A
            else:
                fake = torch.cat([self.real_A, self.fake_B_from_real_A], 1)
            num_fake_pairs += 1
            for netD, lambda_D in zip(self.netD2, self.opt.lambda_D2):
                pred_fake = netD.forward(fake)
                if not self.opt.no_logD_trick:
                    self.loss_G2_GAN += self.criterionGAN2(pred_fake, flipped_label) * lambda_D
                else:
                    self.loss_G2_GAN += -self.criterionGAN2(pred_fake, False) * lambda_D
        if 'fake_fake' in self.opt.GAN_losses_G2:
            if self.opt.no_cgan:
                fake = self.fake_B_from_fake_A
            elif not self.opt.detach_G1_from_G2_y:
                fake = torch.cat([self.transform(self.fake_A), self.fake_B_from_fake_A], 1)
            else:
                fake = torch.cat([self.transform(self.fake_A.detach()), self.fake_B_from_fake_A], 1)
            num_fake_pairs += 1
            for netD, lambda_D in zip(self.netD2, self.opt.lambda_D2):
                pred_fake = netD.forward(fake)
                if not self.opt.no_logD_trick:
                    self.loss_G2_GAN += self.criterionGAN2(pred_fake, flipped_label) * lambda_D
                else:
                    self.loss_G2_GAN += -self.criterionGAN2(pred_fake, False) * lambda_D

        # L1 loss
        if 'real_fake' in self.opt.GAN_losses_G2:
            if self.opt.weights is None:
                weight = None
            else:
                # when using weighted L1, must be label --> image
                weight = Variable(self.Tensor(self.opt.batchSize, 1, self.opt.fineSize, self.opt.fineSize).fill_(1.0),
                                  requires_grad=False)
                real_A = (self.real_A.detach() + 1) / 2
                for i in range(len(self.opt.weights)):
                    weight += real_A.narrow(1, i, 1) * (self.opt.weights[i] - 1.0)
            self.loss_G2_L1 = self.criterionL1(self.fake_B_from_real_A, self.real_B, weight)
        else:
            self.loss_G2_L1 = 0

        # Segmentation loss
        self.loss_F2_CE = torch.nn.BCELoss()((self.fake_A_from_real_B + 1) / 2, (self.real_A + 1) / 2)  # rescale

        # Cycle losses
        self.loss_G2_real_cycle = torch.nn.BCELoss()((self.recon_real_A + 1) / 2, (self.real_A + 1) / 2)
        self.loss_G2_fake_cycle = torch.nn.BCELoss()((self.recon_fake_A + 1) / 2,
                                                     (self.transform(self.fake_A.detach()) + 1) / 2)

        self.loss_G = self.loss_G1_GAN + self.loss_G2_GAN / num_fake_pairs \
                      + self.loss_G2_L1 * self.opt.lambda_A \
                      + self.loss_F2_CE * self.opt.lambda_B \
                      + self.loss_G2_real_cycle * self.opt.lambda_A_cycle \
                      + self.loss_G2_fake_cycle * self.opt.lambda_A_cycle * self.opt.lambda_fake_cycle
        self.loss_G.backward()

    def optimize_parameters(self):
        # forward
        self.forward()

        # D1:
        for _ in range(self.opt.n_update_D1):
            self.optimizer_D1.zero_grad()
            self.backward_D1()
            self.optimizer_D1.step()
            if self.opt.n_update_D1 > 1:
                self.sample_noise()

        # D2:
        for _ in range(self.opt.n_update_D2):
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()
            if self.opt.n_update_D2 > 1:
                self.sample_noise()

        # G:
        for _ in range(self.opt.n_update_G):
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            if self.opt.n_update_G > 1:
                self.sample_noise()

    def get_current_errors(self):
        err_list = [('G2_GAN', self.loss_G2_GAN.data[0]),
                    ('G2_real_cycle', self.loss_G2_real_cycle.data[0]),
                    ('G2_fake_cycle', self.loss_G2_fake_cycle.data[0]),
                    ('D2', self.loss_D2.data[0]),
                    ('G1_GAN', self.loss_G1_GAN.data[0]),
                    ('D1', self.loss_D1.data[0])]
        return OrderedDict(err_list)

    def get_current_visuals(self, save_as_single_image=False):
        if self.isTrain:
            real_A = util.tensor2im(self.real_A.data)
            fake_B_from_real_A = util.tensor2im(self.fake_B_from_real_A.data)
            recon_real_A = util.tensor2im(self.recon_real_A.data)
            recon_fake_A = util.tensor2im(self.recon_fake_A.data)
            fake_A = util.tensor2im(self.transform(self.fake_A).data)
            fake_B_from_fake_A = util.tensor2im(self.fake_B_from_fake_A.data)
            fake_A_from_real_B = util.tensor2im(self.fake_A_from_real_B.data)
            real_B = util.tensor2im(self.real_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B_real_A', fake_B_from_real_A),
                                ('fake_A', fake_A), ('fake_B_fake_A', fake_B_from_fake_A),
                                ('fake_A_real_B', fake_A_from_real_B), ('real_B', real_B),
                                ('recon_real_A', recon_real_A), ('recon_fake_A', recon_fake_A)])
        elif save_as_single_image:
            AB = util.tensor2im(torch.cat([self.transform(self.fake_A).data, self.fake_B_from_fake_A.data], dim=1))
            return OrderedDict([('AB', AB)])
        else:
            fake_A = util.tensor2im(self.transform(self.fake_A).data)
            fake_B_from_fake_A = util.tensor2im(self.fake_B_from_fake_A.data)
            return OrderedDict([('fake_A', fake_A), ('fake_B', fake_B_from_fake_A)])

    def save(self, label):
        self.save_network(self.netG1, 'G1', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netG2, 'G2', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netF2, 'F2', label, gpu_ids=self.gpu_ids)
        for netD, n in zip(self.netD1, range(self.n_netD1)):
            self.save_network(netD, 'D1_%d' % n, label, gpu_ids=self.gpu_ids)
        for netD, n in zip(self.netD2, range(self.n_netD2)):
            self.save_network(netD, 'D2_%d' % n, label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = max(0, self.old_lr - lrd)
        lrd1 = self.opt.lr1 / self.opt.niter_decay
        lr1 = max(0, self.old_lr1 - lrd1)
        lrd2 = self.opt.lr2 / self.opt.niter_decay
        lr2 = max(0, self.old_lr2 - lrd2)
        for param_group in self.optimizer_D1.param_groups:
            param_group['lr'] = lr1
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr2
        for param_group in self.optimizer_G.param_groups:
            if param_group['name'] == 'G1':
                param_group['lr'] = lr1
            elif param_group['name'] == 'G2':
                param_group['lr'] = lr2
            elif param_group['name'] == 'F2':
                param_group['lr'] = lr2
            else:
                param_group['lr'] = lr
        print('update learning rate: %f -> %f, %f -> %f' % (self.old_lr1, lr1, self.old_lr2, lr2))
        self.old_lr = lr
        self.old_lr1 = lr1
        self.old_lr2 = lr2
