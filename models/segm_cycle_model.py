import numpy as np
import torch
import itertools
import random
import util.util as util
from collections import OrderedDict
from torch.autograd import Variable
from util.util import compute_Rand_F_scores
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .loss import CrossEntropyLoss2d


class SegmentationCycleModel(BaseModel):
    def name(self):
        return 'SegmentationCycleModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # Random seed
        self.use_gpu = len(opt.gpu_ids) and torch.cuda.is_available()
        if opt.manualSeed is None:
            opt.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", opt.manualSeed)
        random.seed(opt.manualSeed)
        torch.manual_seed(opt.manualSeed)
        if self.use_gpu:
            torch.cuda.manual_seed_all(opt.manualSeed)

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

        self.num_classes = opt.output_nc + 1 if opt.add_background_onehot else opt.output_nc
        self.class_weights = None if opt.weights is None else self.Tensor(opt.weights)
        # Sigmoid vs Softmax
        self.use_sigmoid_ss = opt.use_sigmoid_ss  # if or not use sigmoid (otherwise softmax)
        self.activation = torch.nn.Sigmoid() if opt.use_sigmoid_ss else torch.nn.Softmax2d()
        self.SoftmaxLoss = CrossEntropyLoss2d(weight=self.class_weights) if self.isTrain else CrossEntropyLoss2d(None)

        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc, opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, self.num_classes, opt.fineSize, opt.fineSize)
        self.label_ = self.LongTensor(opt.batchSize, opt.fineSize, opt.fineSize)
        self.noise1 = None
        self.noise1_ = self.Tensor(opt.batchSize, opt.noise_nc1, self.opt.noiseSize1, self.opt.noiseSize1)
        self.noise2 = None
        self.noise2_ = self.Tensor(opt.batchSize, opt.noise_nc2, self.opt.noiseSize2, self.opt.noiseSize2)
        self.noise_val_ = self.Tensor(opt.batchSize, opt.noise_nc1, self.opt.noiseSizeVal, self.opt.noiseSizeVal)

        # load/define networks
        self.netG1 = networks.define_G(opt.input_nc, self.num_classes, opt.ngf1, opt.which_model_netG1, opt.norm,
                                       not opt.no_dropout1, n_layers_G=opt.n_layers_G1, use_residual=False,
                                       use_fcn=opt.noiseSize1 != 1, noise_nc=opt.noise_nc1,
                                       add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                       upsample_mode=opt.upsample_mode1, n_layers_CRN_block=opt.n_layers_CRN_block1,
                                       share_label_weights=not opt.no_share_label_block_weights1,
                                       n_layers_G_skip=opt.n_layers_G1_skip, gpu_ids=self.gpu_ids)
        self.netG2 = networks.define_G(self.num_classes, opt.input_nc, opt.ngf2, opt.which_model_netG2, opt.norm,
                                       not opt.no_dropout2, n_layers_G=opt.n_layers_G2, use_residual=False,
                                       use_fcn=opt.noiseSize2 != 1, noise_nc=opt.noise_nc2,
                                       add_gaussian_noise=opt.add_gaussian_noise, gaussian_sigma=opt.gaussian_sigma,
                                       upsample_mode=opt.upsample_mode2, n_layers_CRN_block=opt.n_layers_CRN_block2,
                                       share_label_weights=not opt.no_share_label_block_weights2,
                                       n_layers_G_skip=opt.n_layers_G2_skip, gpu_ids=self.gpu_ids)
        if self.isTrain:
            use_sigmoid = opt.no_lsgan2
            assert (len(opt.scale_factor2) == len(opt.lambda_D2) == len(opt.n_layers_D2))
            self.n_netD2 = len(opt.scale_factor2)
            self.netD2 = []
            if opt.no_cgan:
                netD_input_nc = opt.input_nc
            else:
                netD_input_nc = opt.input_nc + self.num_classes
            for scale, n_layers in zip(opt.scale_factor2, opt.n_layers_D2):
                self.netD2.append(networks.define_D(netD_input_nc, opt.ndf2, opt.which_model_netD2, n_layers_D=n_layers,
                                                    norm=opt.norm, use_sigmoid=use_sigmoid, scale_factor=scale,
                                                    gpu_ids=self.gpu_ids))
        if self.isTrain and opt.sequential_train and not opt.continue_train:
            if 'G1' in self.opt.which_model_to_load:
                self.load_network(self.netG1, 'G1', opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'G2' in self.opt.which_model_to_load:
                self.load_network(self.netG2, 'G2', opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
            if 'D2' in self.opt.which_model_to_load:
                for netD, n in zip(self.netD2, range(self.n_netD2)):
                    self.load_network(netD, 'D2_%d' % n, opt.which_epoch_sequential, model_dir=opt.pretrained_model_dir)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG1, 'G1', opt.which_epoch)
            self.load_network(self.netG2, 'G2', opt.which_epoch)
            if self.isTrain:
                for netD, n in zip(self.netD2, range(self.n_netD2)):
                    self.load_network(netD, 'D2_%d' % n, opt.which_epoch)

        if self.isTrain:
            self.fake_pool2 = ImagePool(opt.pool_size)
            self.old_lr = opt.lr
            self.old_lr1 = opt.lr1
            self.old_lr2 = opt.lr2
            # define loss functions
            self.criterionGAN2 = networks.GANLoss(use_lsgan=not opt.no_lsgan2, tensor=self.Tensor)
            self.criterionL1 = networks.WeightedL1Loss()

            # initialize optimizers
            self.optimizer_G = torch.optim.Adam([{'name': 'G1', 'params': self.netG1.parameters(), 'lr': opt.lr1},
                                                 {'name': 'G2', 'params': self.netG2.parameters(), 'lr': opt.lr2}],
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            params = itertools.chain()
            for netD in self.netD2:
                if opt.which_model_netD2 == 'n_layers_sep':
                    params = itertools.chain(params, netD.model.parameters(),
                                             netD.netA.parameters(), netD.netB.parameters())
                else:
                    params = itertools.chain(params, netD.model.parameters())
            self.optimizer_D2 = torch.optim.Adam(params, lr=opt.lr2, betas=(opt.beta1, 0.999))

        print('------------ Networks initialized -------------')
        networks.print_network(self.netG1)
        networks.print_network(self.netG2)
        if self.isTrain:
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

        # do transformation here
        # assume input_B is always one-hot embedding (, may add index embedding later)
        # real_B and fake_B are always one-hot, logit is fake_B before sigmoid or softmax, label is always index label
        input_B = (input_B + 1) / 2.0  # rescale to [0, 1]
        if self.opt.add_background_onehot:
            input_B = torch.cat([input_B, 1.0 - torch.clamp(input_B.sum(dim=1, keepdim=True), 0, 1)], dim=1)
        _, label = input_B.max(dim=1)
        self.label_.resize_(label.size()).copy_(label)
        self.label = Variable(self.label_, requires_grad=False)
        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self, val_mode=False):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        if not val_mode:
            self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                        self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
            self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
                                                        self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        else:
            self.noise1 = Variable(self.noise_val_.resize_(self.opt.batchSize, self.opt.noise_nc,
                                                           self.opt.noiseSizeVal, self.opt.noiseSizeVal).normal_(0, 1))
        self.logit = self.netG1.forward(self.real_A, self.noise1, activation=lambda x: x)
        self.fake_B = self.activation(self.logit)
        if not val_mode:
            self.fake_A = self.netG2.forward(self.real_B, self.noise2)
            self.recon_A = self.netG2.forward(self.fake_B, self.noise2)

    def sample_noise(self):
        self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                    self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
        self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
                                                    self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        self.logit = self.netG1.forward(self.real_A, self.noise1, activation=lambda x: x)
        self.fake_B = self.activation(self.logit)
        self.fake_A = self.netG2.forward(self.real_B, self.noise2)
        self.recon_A = self.netG2.forward(self.fake_B, self.noise2)

    # no backprop gradients
    def test(self):
        self.noise1 = Variable(self.noise1_.resize_(self.opt.batchSize, self.opt.noise_nc1,
                                                    self.opt.noiseSize1, self.opt.noiseSize1).normal_(0, 1))
        # self.noise2 = Variable(self.noise2_.resize_(self.opt.batchSize, self.opt.noise_nc2,
        #                                             self.opt.noiseSize2, self.opt.noiseSize2).normal_(0, 1))
        self.real_A = Variable(self.input_A, volatile=True)
        self.real_B = Variable(self.input_B, volatile=True)
        self.logit = self.netG1.forward(self.real_A, self.noise1, activation=lambda x: x)
        self.fake_B = self.activation(self.logit)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D2(self):
        # Fake
        if self.opt.no_cgan:
            fake = self.fake_pool2.query(self.fake_A)
        else:
            fake = self.fake_pool2.query(torch.cat([self.real_B, self.fake_A], 1))
        self.loss_D2_fake = 0
        for netD in self.netD2:
            pred_fake = netD.forward(fake.detach())
            self.loss_D2_fake += self.criterionGAN2(pred_fake, False)
        # Real
        if self.opt.no_cgan:
            real = self.real_A
        else:
            real = torch.cat([self.real_B, self.real_A], 1)
        self.loss_D2_real = 0
        for netD in self.netD2:
            pred_real = netD.forward(real)
            self.loss_D2_real += self.criterionGAN2(pred_real, True)
        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        self.loss_D2.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        self.loss_G2_GAN = 0
        if self.opt.no_cgan:
            fake = self.fake_A
        else:
            fake = torch.cat([self.real_B, self.fake_A], 1)
        for netD, lambda_D in zip(self.netD2, self.opt.lambda_D2):
            pred_fake = netD.forward(fake)
            self.loss_G2_GAN += self.criterionGAN2(pred_fake, True) * lambda_D

        # (weighted) cross-entropy loss
        if self.use_sigmoid_ss:
            if self.class_weights is None:
                weight = None
            else:
                # compute weight on the fly
                weight = Variable(self.Tensor(self.opt.batchSize, 1, self.opt.fineSize, self.opt.fineSize).fill_(1.0),
                                  requires_grad=False)
                for i in range(len(self.class_weights)):
                    weight += self.real_B.narrow(1, i, 1) * (self.class_weights[i] - 1.0)
            self.loss_G1_CE = torch.nn.BCELoss(weight=weight)(self.fake_B, self.real_B)
        else:
            self.loss_G1_CE = self.SoftmaxLoss.forward(self.logit, self.label)

        # L1 loss for G2 (together with GAN loss)
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B)

        # InfoGAN loss
        self.loss_G_cycle = self.criterionL1(self.recon_A, self.real_A)

        # combine losses
        self.loss_G = self.loss_G1_CE * self.opt.lambda_A + \
                      self.loss_G2_GAN + self.loss_G_L1 * self.opt.lambda_B + \
                      self.loss_G_cycle * self.opt.lambda_A_cycle
        self.loss_G.backward()

    def compute_cross_entropy_loss(self):
        # (weighted) cross-entropy loss
        if self.use_sigmoid_ss:
            self.loss_G1_CE = torch.nn.BCELoss(weight=None)(self.fake_B, self.real_B)
        else:
            self.loss_G1_CE = self.SoftmaxLoss.forward(self.logit, self.label)

    def optimize_parameters(self):
        self.forward()
        for _ in range(self.opt.n_update_D2):
            self.optimizer_D2.zero_grad()
            self.backward_D2()
            self.optimizer_D2.step()
            if self.opt.n_update_D2 > 1:
                self.sample_noise()
        for _ in range(self.opt.n_update_G):
            self.optimizer_G.zero_grad()
            self.backward_G()
            self.optimizer_G.step()
            if self.opt.n_update_G > 1:
                self.sample_noise()

    def get_current_errors(self):
        err_list = [('G_CE', self.loss_G1_CE.data[0])]
        if self.isTrain:
            err_list += [('G_GAN', self.loss_G2_GAN.data[0])]
        return OrderedDict(err_list)

    def get_current_visuals(self):  # TODO: image-->label
        if self.isTrain:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data * 2 - 1)
            real_B = util.tensor2im(self.real_B.data * 2 - 1)
            recon_A = util.tensor2im(self.recon_A.data)
            return OrderedDict([('image', real_A),
                                ('label', real_B),
                                ('prediction', fake_B),
                                ('recon', recon_A)])
        else:
            real_A = util.tensor2im(self.real_A.data)
            fake_B = util.tensor2im(self.fake_B.data * 2 - 1)
            real_B = util.tensor2im(self.real_B.data * 2 - 1)
            return OrderedDict([('image', real_A),
                                ('label', real_B),
                                ('prediction', fake_B)])

    def save(self, label):
        self.save_network(self.netG1, 'G1', label, gpu_ids=self.gpu_ids)
        self.save_network(self.netG2, 'G2', label, gpu_ids=self.gpu_ids)
        for netD, n in zip(self.netD2, range(self.n_netD2)):
            self.save_network(netD, 'D2_%d' % n, label, gpu_ids=self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = max(0, self.old_lr - lrd)
        lrd1 = self.opt.lr1 / self.opt.niter_decay
        lr1 = max(0, self.old_lr1 - lrd1)
        lrd2 = self.opt.lr2 / self.opt.niter_decay
        lr2 = max(0, self.old_lr2 - lrd2)
        for param_group in self.optimizer_D2.param_groups:
            param_group['lr'] = lr2
        for param_group in self.optimizer_G.param_groups:
            if param_group['name'] == 'G1':
                param_group['lr'] = lr1
            elif param_group['name'] == 'G2':
                param_group['lr'] = lr2
            else:
                param_group['lr'] = lr
        print('update learning rate: %f -> %f, %f -> %f' % (self.old_lr1, lr1, self.old_lr2, lr2))
        self.old_lr = lr
        self.old_lr1 = lr1
        self.old_lr2 = lr2

    def reset_accs(self):
        self.confusion = 0
        self.numAveragedImages = 0  # num of averaged images
        self.numAveragedPixels = 0  # num of averaged pixels
        self.RandScore = 0
        self.pixelAcc = 0
        self.meanAcc = 0
        self.meanIU = 0

    def accum_accs(self):
        # Rand F-score
        if 'RandScore' in self.opt.which_metric:
            self.compute_current_Rand_score()
        # mean IU
        if 'meanIU' in self.opt.which_metric:
            self.compute_current_accuracy()

    def compute_current_Rand_score(self):
        assert (self.num_classes == 2)  # currently only support binary classification
        T = self.real_B.data.cpu().numpy()
        S = self.fake_B.data.cpu().numpy()
        RIs = compute_Rand_F_scores(S, T, do_thin=False)
        n = self.numAveragedImages
        m = n + RIs.size
        self.numAveragedImages = m
        self.RandScore = (n * self.RandScore + RIs.sum()) / m

    def compute_current_accuracy(self):
        if self.opt.add_background_onehot_acc:
            labels = self.real_B.data.cpu().float().numpy()
            labels = np.concatenate([labels, 1.0 - np.minimum(1, labels.sum(axis=1, keepdims=True))],
                                    axis=1).argmax(axis=1).astype(np.int32).flatten()
            predictions = self.activation(self.logit).data.cpu().float().numpy()
            predictions = np.concatenate([predictions, 1.0 - np.minimum(1, predictions.sum(axis=1, keepdims=True))],
                                         axis=1).argmax(axis=1).astype(np.int32).flatten()
            plusone = 1
        else:
            labels = self.label_.cpu().float().numpy().astype(np.int32).flatten()
            predictions = self.logit.data.cpu().numpy().argmax(axis=1).astype(np.int32).flatten()
            plusone = 0
        numPixels = labels.size  # no label to ignore
        confusion = np.zeros([self.num_classes + plusone, self.num_classes + plusone])
        for i in range(numPixels):
            confusion[labels[i], predictions[i]] += 1
        self.confusion += confusion
        self.numAveragedPixels += numPixels
        rel = np.sum(self.confusion, axis=1)
        sel = np.sum(self.confusion, axis=0)
        tp = np.diag(self.confusion)
        self.pixelAcc = tp.sum() / np.maximum(1, self.numAveragedPixels)
        self.meanAcc = np.mean(tp / np.maximum(1, rel))
        self.meanIU = np.mean(tp / np.maximum(1, rel + sel - tp))

    def get_current_accs(self):
        acc_list = []
        if 'RandScore' in self.opt.which_metric:
            acc_list.append(('RandScore', self.RandScore))
        if 'meanIU' in self.opt.which_metric:
            acc_list.append(('meanIU', self.meanIU))
        # acc_list = [('meanIU', self.meanIU), ('RandScore', self.RandScore)]
        return OrderedDict(acc_list)
