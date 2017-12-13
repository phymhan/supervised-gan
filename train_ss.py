import time
import torch
import random
import numpy as np
import os
import ntpath
import re
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util.util import save_image, mkdir

opt_train = TrainOptions().parse()
opt_val = TrainOptions().parse()

# Random seed
opt_train.use_gpu = len(opt_train.gpu_ids) and torch.cuda.is_available()
if opt_train.manualSeed is None:
    opt_train.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt_train.manualSeed)
random.seed(opt_train.manualSeed)
np.random.seed(opt_train.manualSeed)
torch.manual_seed(opt_train.manualSeed)
if opt_train.use_gpu:
    torch.cuda.manual_seed_all(opt_train.manualSeed)

data_loader = CreateDataLoader(opt_train)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

opt_val.phase = 'val'
opt_val.nThreads = 1   # test code only supports nThreads = 1
opt_val.batchSize = 1  # test code only supports batchSize = 1
opt_val.serial_batches = True  # no shuffle
opt_val.no_flip = True  # no flip
opt_val.no_rotate = True  # no rotate
if opt_val.valSize == 0:
    opt_val.valSize = opt_val.loadSize
opt_val.loadSize = opt_val.valSize
opt_val.fineSize = opt_val.valSize
data_loader_val = CreateDataLoader(opt_val)
dataset_val = data_loader_val.load_data()
dataset_size_val = len(data_loader_val)
print('#validation images = %d' % dataset_size_val)

## Visualizers
model = create_model(opt_train)
visualizer = Visualizer(opt_train)
opt_train.display_id = 10
opt_train.display_title = 'train accuracy'
visualizer_acc = Visualizer(opt_train)
opt_val.display_id = 20
opt_val.display_title = 'val accuracy'
visualizer_acc_val = Visualizer(opt_val)

total_steps = 0
best_metric = -1

chkpt_dir = os.path.join(opt_train.checkpoints_dir, opt_train.name)

for epoch in range(1, opt_train.niter + opt_train.niter_decay + 1):
    epoch_start_time = time.time()
    model.reset_accs()
    for i, data in enumerate(dataset):
        iter_start_time = time.time()
        total_steps += opt_train.batchSize
        epoch_iter = total_steps - dataset_size * (epoch - 1)
        model.set_input(data)
        model.optimize_parameters()
        model.accum_accs()

        if total_steps % opt_train.display_freq == 0:
            visualizer.display_current_results(model.get_current_visuals(), epoch)

        if total_steps % opt_train.print_freq == 0:
            errors = model.get_current_errors()
            t = (time.time() - iter_start_time) / opt_train.batchSize
            visualizer.print_current_errors(epoch, epoch_iter, errors, t)
            if opt_train.display_id > 0:
                visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt_train, errors)
        if total_steps % opt_train.print_freq == 0:
            accs = model.get_current_accs()
            visualizer_acc.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt_train, accs)

        if total_steps % opt_train.save_latest_freq == 0:
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
            model.save('latest')

    ####################################################################################################################
    # eval val after every epoch
    model.reset_accs()

    if opt_val.save_val_visuals:
        img_dir = os.path.join(chkpt_dir, 'val', 'epoch%03d' % epoch)
        mkdir(img_dir)

    for j, data in enumerate(dataset_val):
        model.set_input(data)
        model.forward(val_mode=True)
        model.accum_accs()
        # if j % opt.display_freq == 0:
        #     visualizer.display_current_results(model.get_current_visuals(), epoch)
        if opt_val.save_val_visuals:
            visuals = model.get_current_visuals()
            name = os.path.splitext(ntpath.basename(model.get_image_paths()[0]))[0]
            for label, image_numpy in visuals.items():
                if re.search('image', label):  # save disk space by skipping raw image
                    continue
                image_name = '%s_%s.png' % (name, label)
                save_path = os.path.join(img_dir, image_name)
                save_image(image_numpy, save_path)

    accs = model.get_current_accs()
    if opt_val.best_metric is not 'None':
        if accs[opt_val.best_metric] > best_metric:
            best_metric = accs[opt_val.best_metric]
            model.save('best')
    visualizer_acc_val.plot_current_errors(epoch, 0.0, opt_val, accs)
    ####################################################################################################################

    if epoch % opt_train.save_epoch_freq == 0:
        print('saving the model at the end of epoch %d, iters %d' % (epoch, total_steps))
        model.save('latest')
        model.save(epoch)

    print('End of epoch %d / %d \t Time Taken: %d sec' %
          (epoch, opt_train.niter + opt_train.niter_decay, time.time() - epoch_start_time))

    if epoch > opt_train.niter:
        model.update_learning_rate()
        # if opt_train.use_dynamic_lambda:
        #     model.update_lambda()
