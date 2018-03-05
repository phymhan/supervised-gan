# Supervised GAN #

## Quick Start ###

### Datasets and pre-trained models:
* Download pre-processed [VNC dataset](https://github.com/unidesigner/groundtruth-drosophila-vnc) from <https://drive.google.com/open?id=1PpIu89DgsE1L67yoWWH2cnwGz-8n7Jc5>
* Extract `vnc-rgb.zip` and put the folder under `./datasets/gan` folder
* Download pre-trained models (D and G) from <https://drive.google.com/open?id=1fGTMV6gp3Ud2fRnDlC4WBK8fxTdXvfTZ>
* Extract `twostage_D1G1.zip` and put all `.pth` files under `./pretrained/twostage` folder

### Train a DSGAN model:
Training:
```
python train.py --dataroot ./datasets/gan/vnc-rgb --name dsgan_model --model twostage_cycle --which_direction AtoB --dataset_mode single --loadSize 1024 --fineSize 512 --transform_1to2 bilinear_2 --batchSize 1 --input_nc 2 --output_nc 1 --which_channel rg_b --which_model_netG1 fcgan --n_layers_G1 5 --ngf1 32 --which_model_netD1 n_layers --n_layers_D1 3 3 --ndf1 32 --scale_factor1 1 2 --lambda_D1 0.5 0.4 --which_model_netG2 crn --ngf2 64 --upsample_mode2 bilinear --n_layers_CRN_block2 2 --which_model_netF2 unet_128 --nff2 32 --which_model_netD2 n_layers --n_layers_D2 3 4 3 4 --ndf2 64 --scale_factor2 1 1 2 2 --lambda_D2 0.3 0.3 0.2 0.2 --lambda_A 10 --lambda_B 10 --lambda_A_cycle 5 --lambda_fake_cycle 1 --noise_nc1 8 --noiseSize1 4 --noise_nc2 8 --noiseSize2 8 --norm instance --no_dropout1 --n_update_G 1 --niter 150 --niter_decay 50 --display_freq 40 --save_epoch_freq 200 --no_lsgan1 --no_lsgan2 --sequential_train --manualSeed 0 --GAN_losses_D2 real_fake --GAN_losses_G2 real_fake --sequential_train --which_epoch_sequential seq --which_model_to_load G1 D1 --pretrained_model_dir pretrained/twostage --lr1 0.0002 --lr2 0.0002
```

Testing:
```
python test.py --dataroot ./datasets/null --name dsgan_model --model twostage_cycle --which_direction AtoB --dataset_mode single --loadSize 512 --fineSize 512 --transform_1to2 bilinear_2 --batchSize 1 --input_nc 2 --output_nc 1 --which_channel rg_b --which_model_netG1 fcgan --n_layers_G1 5 --ngf1 32 --which_model_netD1 n_layers --n_layers_D1 3 3 --ndf1 32 --scale_factor1 1 2 --which_model_netG2 crn --ngf2 64 --upsample_mode2 bilinear --n_layers_CRN_block2 2 --which_model_netF2 unet_128 --nff2 32 --which_model_netD2 n_layers --n_layers_D2 3 4 3 4 --ndf2 64 --scale_factor2 1 1 2 2 --noise_nc1 8 --noiseSize1 2 --noise_nc2 8 --noiseSize2 4 --norm instance --no_dropout1 --manualSeed 0 --serial_batches --no_flip --no_rotate --how_many 100
```


### Train a SGAN model

Training a SGAN model involves training two separate models, a GAN and a CGAN.

#### Step 1, training a GAN model:
```
python train.py --dataroot ./datasets/gan/vnc-rgb --name sgan_gan --model fcgan --which_direction A --dataset_mode single --loadSize 512 --fineSize 512 --batchSize 1 --input_nc 2 --which_model_netG deconv --n_layers_G 5 --ngf 32 --which_model_netD n_layers --n_layers_D 3 3 3 --ndf 32 --scale_factor 1 2 4 --lambda_D 0.5 0.4 0.1 --noise_nc 8 --noiseSize 8 --norm instance --no_dropout --n_update_G 2 --niter 100 --niter_decay 100 --display_freq 40 --save_epoch_freq 200 --no_lsgan --which_channel rg --no_dropout
```

#### Step 2, training a CGAN model:
```
python train.py --dataroot ./datasets/gan/vnc-rgb --name sgan_cgan --model cgan --which_direction AtoB --dataset_mode single --loadSize 1024 --fineSize 512 --batchSize 1 --input_nc 2 --output_nc 1 --which_model_netG unet_256 --ngf 64 --which_model_netD n_layers --n_layers_D 3 4 --ndf 64 --scale_factor 1 1 --lambda_D 0.5 0.5 --lambda_A 10 --noise_nc 8 --noiseSize 4 --norm instance --n_update_G 2 --niter 150 --niter_decay 50 --display_freq 50 --save_epoch_freq 200 --weight_L1 2 4 --no_lsgan --manualSeed 0 --add_gaussian_noise --which_channel rg_b
```

Similar to training a label generator in the first step, we can easily train JointGAN and UnsupervisedGAN by simply changing the ```--which_channel``` option.

### Train a JointGAN model
```
python train.py --dataroot ./datasets/gan/vnc-rgb --name jointgan --model fcgan --which_direction A --dataset_mode single --loadSize 512 --fineSize 512 --batchSize 1 --input_nc 2 --which_model_netG deconv --n_layers_G 5 --ngf 32 --which_model_netD n_layers --n_layers_D 3 3 3 --ndf 32 --scale_factor 1 2 4 --lambda_D 0.5 0.4 0.1 --noise_nc 8 --noiseSize 8 --norm instance --no_dropout --n_update_G 2 --niter 100 --niter_decay 100 --display_freq 40 --save_epoch_freq 200 --no_lsgan --which_channel rg_b --no_dropout
```


### Train a UnsupervisedGAN model
```
python train.py --dataroot ./datasets/gan/vnc-rgb --name unsupgan --model fcgan --which_direction A --dataset_mode single --loadSize 512 --fineSize 512 --batchSize 1 --input_nc 2 --which_model_netG deconv --n_layers_G 5 --ngf 32 --which_model_netD n_layers --n_layers_D 3 3 3 --ndf 32 --scale_factor 1 2 4 --lambda_D 0.5 0.4 0.1 --noise_nc 8 --noiseSize 8 --norm instance --no_dropout --n_update_G 2 --niter 100 --niter_decay 100 --display_freq 40 --save_epoch_freq 200 --no_lsgan --which_channel b --no_dropout
```


### Train a Segmentation network
Coming soon...



## Options and Parameters #

Training parameters:
* The structure and organization of the code are largely based on [CycleGAN and pix2pix PyTorch implementation](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix). The basic training options are similar, please refer to their [website](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix#trainingtest-details).
* The training process can similarly be visualized using [visdom](https://github.com/facebookresearch/visdom).
* `--which_model_to_load` defines which pre-trained model(s) to load when training twostage models (DSGANs), it can take: `G1`, `D1`, `G2`, `D2`, `F2`. `F2` is the reconstructor for the second conditional part. The models should be put under folders specified by `--pretrained_model_dir`.
* `--GAN_losses_D2` and `--GAN_losses_G2`: if contains `'real_fake'`, the (realA, fakeB) pair is included in adversarial loss (or the value function); if contains `'fake_fake'`, the (fakeA, fakeB) pair is included.
* We change the definition of `--lambda_A` and `--lambda_B`: in our code `--lambda_A` determines the weight for regression loss from A to B. For example, if we are training a conditional GAN (CGAN) (A -> B,  label to image), then `--lambda_A` is the L1-regression loss on B; if training a segmentation model (A -> B, image to label), `--lambda_A` is the weight for cross-entropy loss on B. The weight for cycle losses are defined by `--lambda_A_cycle` and `--lambda_B_cycle`.
* `--n_update_D` and `--n_update_G` are numbers of updates of D and G in each iteration.

We add lots of options in `base_options.py`, which basically defines the models and structures.
* `noise_nc` defines the number of channels of input noises (latent noise image).
* `noiseSize` is the height and width (a single integer) of the input noise.
* `--scale_factor` is a list specifies the scales for each discriminators (since we are using multi-scale discriminator which is implemented as a list of single discriminators).
* `--n_layers_D` is also a list.
* If `--add_gaussian_noise` is `true`, Gaussian noise will be added when upsampling. The noise level is specified by `--gaussian_sigma`.
* `--transform_1to2` defines the transform applied to the output of the first generator. If the value is `'bilinear_2'`, the output of G1 will be upsampled by a factor of 2 before being fed into G2.
