import time
import os
import numpy as np
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip
opt.no_rotate = True

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
print('#testing images = %d' % len(data_loader))
model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
model.reset_accs()
ce_loss = []

for i, data in enumerate(dataset):
    # if i >= opt.how_many:
    #     break
    model.set_input(data)
    model.test()
    model.compute_cross_entropy_loss()
    model.accum_accs()
    errs = model.get_current_errors()
    ce_loss.append(errs['G_CE'])
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()
    print('process image... %s' % img_path)
    visualizer.save_images(webpage, visuals, img_path)

accs = model.get_current_accs()
ce_loss = np.array(ce_loss)
print('Segmentation results:')
for key in accs.keys():
    print('{}: {}'.format(key, accs[key]))
print('cross entropy loss: mean {}, std {}'.format(np.mean(ce_loss), np.std(ce_loss)))
webpage.save()
