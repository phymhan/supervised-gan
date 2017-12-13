from tqdm import tqdm
# import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html
import numpy as np

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.no_rotate = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

l2_dist = []
ll_noise = []
ll_noise_init = []

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)

    print('reconstruct image {}...'.format(i))
    this_l2_dist, this_ll_noise, this_ll_noise_init = model.reconstruction()

    l2_dist.append(this_l2_dist)
    ll_noise.append(this_ll_noise)
    ll_noise_init.append(this_ll_noise_init)

    visuals = model.get_current_visuals(True)
    img_path = model.get_image_paths()
    visualizer.save_images(webpage, visuals, img_path)

webpage.save()

l2_dist = np.array(l2_dist).squeeze()
ll_noise = np.array(ll_noise).squeeze()
ll_noise_init = np.array(ll_noise_init).squeeze()
print('BCE: mean {0:0.4f} std {1:0.4f}; noise: mean {2:0.4f} std {3:0.4f}; noise init: mean {4:0.4f} std {5:0.4f}'.
      format(np.mean(l2_dist), np.std(l2_dist), np.mean(ll_noise), np.std(ll_noise), np.mean(ll_noise_init), np.std(ll_noise_init)))
