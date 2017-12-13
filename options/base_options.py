import argparse
import os
from util import util
import torch

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=True, help='path to images (should have subfolders trainA, trainB, valA, valB, etc)')
        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--patchSize', type=int, default=70, help='patch size')
        self.parser.add_argument('--input_nc', type=int, default=3, help='# of input image channels')
        self.parser.add_argument('--noise_nc', type=int, default=8, help='# of input noise channels')
        self.parser.add_argument('--noiseSize', type=int, default=1, help='# of noise image size')
        self.parser.add_argument('--noiseSizeVal', type=int, default=1, help='# of noise image size')
        self.parser.add_argument('--output_nc', type=int, default=3, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks', help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=[3], nargs='+', help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--n_layers_G', type=int, default=5, help='specifies number of layers if which_model_netG==deconv, or number of skip connections if using unet')
        self.parser.add_argument('--scale_factor', type=int, default=[1], nargs='+', help='scale factor for discriminators')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned', help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='cycle_gan', help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--identity', type=float, default=0.0, help='use identity mapping. Setting identity other than 1 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set optidentity = 0.1')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--no_rotate', action='store_true', help='if specified, do not rotate the images for data augmentation')
        self.parser.add_argument('--use_residual', action='store_true', help='add residual shortcut to G')
        self.parser.add_argument('--add_gaussian_noise', action='store_true', help='add Gaussian noise when upsampling')
        self.parser.add_argument('--gaussian_sigma', type=float, default=0.1, help='std of Gaussian noise added')
        self.parser.add_argument('--which_channel', type=str, default='rg', help='selects channels to read as input')
        self.parser.add_argument('--manualSeed', type=int, default=None, help='manual random seed')
        self.parser.add_argument('--display_title', type=str, default='loss over time', help='title of plot')
        self.parser.add_argument('--n_layers_G_skip', type=int, default=-1, help='for compatibility reasons')
        self.parser.add_argument('--use_sigmoid_ss', action='store_true', help='use sigmoid rather than softmax, in semantic segmentation')
        self.parser.add_argument('--weights', type=float, default=None, nargs='+', help='weights for L1 loss in cGAN, or CE loss in segmentation')
        self.parser.add_argument('--upsample_mode', type=str, default='convt', help='upsample mode, convt or bilinear')
        self.parser.add_argument('--no_share_label_block_weights', action='store_true', help='do not share weights of label blocks in CRN')
        self.parser.add_argument('--n_layers_CRN_block', type=int, default=1, help='number of layers of CRN inter blocks')
        self.parser.add_argument('--pretrained_model_dir', type=str, default='', help='pretrained models are saved here, if is empty, use checkpoint_dir/$name instead')

        # for two-stage model only:
        self.parser.add_argument('--scale_factor1', type=int, default=[1], nargs='+', help='scale factor for discriminators')
        self.parser.add_argument('--scale_factor2', type=int, default=[1], nargs='+', help='scale factor for discriminators')
        self.parser.add_argument('--which_model_netD1', type=str, default='n_layers', help='selects model to use for netD1')
        self.parser.add_argument('--which_model_netG1', type=str, default='fcgan', help='selects model to use for netG1')
        self.parser.add_argument('--which_model_netF1', type=str, default='fcgan', help='selects model to use for netF1')
        self.parser.add_argument('--ngf1', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf1', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nff1', type=int, default=64, help='# of reconstructor filters in first conv layer')
        self.parser.add_argument('--n_layers_D1', type=int, default=[3], nargs='+', help='only used if which_model_netD1==n_layers')
        self.parser.add_argument('--n_layers_G1', type=int, default=5, help='only used if which_model_netG1==fcgan')
        self.parser.add_argument('--n_layers_F1', type=int, default=5, help='only used if which_model_netG1==fcgan')
        self.parser.add_argument('--no_dropout1', action='store_true', help='no dropout for the generator1')
        self.parser.add_argument('--noise_nc1', type=int, default=256, help='# of input noise1 channels')
        self.parser.add_argument('--noiseSize1', type=int, default=1, help='# of noise1 channels')
        self.parser.add_argument('--which_model_netD2', type=str, default='n_layers', help='selects model to use for netD2')
        self.parser.add_argument('--which_model_netG2', type=str, default='unet_128', help='selects model to use for netG2')
        self.parser.add_argument('--which_model_netF2', type=str, default='unet_128', help='selects model to use for netF2')
        self.parser.add_argument('--ngf2', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--ndf2', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--nff2', type=int, default=64, help='# of reconstructor filters in first conv layer')
        self.parser.add_argument('--n_layers_D2', type=int, default=[3], nargs='+', help='only used if which_model_netD2==n_layers')
        self.parser.add_argument('--n_layers_G2', type=int, default=5, help='only used if which_model_netG2==fcgan')
        self.parser.add_argument('--n_layers_F2', type=int, default=5, help='only used if which_model_netF2==fcgan')
        self.parser.add_argument('--no_dropout2', action='store_true', help='no dropout for the generator2')
        self.parser.add_argument('--noise_nc2', type=int, default=256, help='# of input noise2 channels')
        self.parser.add_argument('--noiseSize2', type=int, default=1, help='# of noise2 channels')
        self.parser.add_argument('--transform_1to2', type=str, default='None', help='transform from output of G1(z1) to input of G2(y, z2)')
        self.parser.add_argument('--use_residual1', action='store_true', help='add residual shortcut to G1')
        self.parser.add_argument('--use_residual2', action='store_true', help='add residual shortcut to G2')
        self.parser.add_argument('--upsample_mode1', type=str, default='convt', help='upsample mode, convt or bilinear')
        self.parser.add_argument('--no_share_label_block_weights1', action='store_true', help='do not share weights of label blocks in CRN1')
        self.parser.add_argument('--n_layers_CRN_block1', type=int, default=1, help='number of layers of CRN inter blocks')
        self.parser.add_argument('--upsample_mode2', type=str, default='convt', help='upsample mode, convt or bilinear')
        self.parser.add_argument('--no_share_label_block_weights2', action='store_true', help='do not share weights of label blocks in CRN2')
        self.parser.add_argument('--n_layers_CRN_block2', type=int, default=1, help='number of layers of CRN inter blocks')
        self.parser.add_argument('--n_layers_G1_skip', type=int, default=-1, help='for compatibility reasons')
        self.parser.add_argument('--n_layers_G2_skip', type=int, default=-1, help='for compatibility reasons')

        # for segmentation model only:
        self.parser.add_argument('--valSize', type=int, default=0, help='val size')
        self.parser.add_argument('--save_val_visuals', action='store_true', help='save val visuals')
        self.parser.add_argument('--best_metric', type=str, default='None', help='pick and save best model according to which metric')
        self.parser.add_argument('--which_metric', default=['None'], nargs='+', help='which metric to compute during training')
        self.parser.add_argument('--add_background_onehot', action='store_true', help='add background class for one-hot representation')
        self.parser.add_argument('--add_background_onehot_acc', action='store_true', help='add background class for one-hot representation when computing accuracy')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain   # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)
        
        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ---------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ---------------\n')
        return self.opt
