import time
import os
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

model = create_model(opt)
visualizer = Visualizer(opt)

# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

# test
if opt.model.startswith('cgan'):
    # only cgan needs label
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals(
            save_as_single_image=opt.save_as_single_image
        )
        img_path = model.get_image_paths()
        print('process image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)
else:
    # fcgan, twostage models
    for i in range(opt.how_many):
        model.test()
        visuals = model.get_current_visuals(
            save_as_single_image=opt.save_as_single_image
        )
        img_path = ['%04d.png' % (i + 1)]
        print('produce image... %s' % img_path)
        visualizer.save_images(webpage, visuals, img_path)

webpage.save()
