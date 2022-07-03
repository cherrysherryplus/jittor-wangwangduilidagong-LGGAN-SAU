"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
from pathlib import Path
import numpy as np
from collections import OrderedDict
from options.train_options import TrainOptions
import data
from data import FID_val
from util.iter_counter import IterationCounter
from util.visualizer import Visualizer
from util.fid_scores import fid_jittor
from util.util import fix_seed, start_grad, stop_grad
from trainers.pix2pix_trainer import Pix2PixTrainer
import jittor as jt

# parse options
opt = TrainOptions().parse()

# global seed
fix_seed(seed = 628)

# debug 
# opt.display_freq = 5
# opt.use_sau = True
# opt.batchSize = 2
# jt.flags.lazy_execution = 0

# if continue train
# opt.continue_train = True

jt.flags.use_cuda = (jt.has_cuda and opt.gpu_ids != "-1")
# print options to help debugging
print(' '.join(sys.argv))

# load the dataset
dataset = data.create_dataloader(opt)
dataloader = dataset().set_attrs(batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=int(opt.nThreads),
        drop_last=opt.isTrain)
dataloader.initialize(opt)
print("the Dataset contains %d labels" %(len(dataloader)))


# load FID val dataset
data_val = FID_val.FidDataset().set_attrs(batch_size=opt.batchSize, drop_last=False)
# data_val.initialize(opt)

best_fid = 99999999
if opt.continue_train:
    txt_path = f"checkpoints/{opt.name}/best_iter.txt"
    try:
        best_epoch, best_iter, best_fid = np.loadtxt(txt_path, delimiter=',', dtype=int)
        print('Load cur_fid (approx: %d) from epoch %d at iteration %d' % (best_fid, best_epoch, best_iter))
    except:
        print(f'Could not load best fid record at {txt_path}')
fid_test = fid_jittor(opt, data_val, best_fid)


# create trainer for our model
trainer = Pix2PixTrainer(opt)

# create tool for counting iterations
iter_counter = IterationCounter(opt, len(dataloader))

# create tool for visualization
visualizer = Visualizer(opt)

for epoch in iter_counter.training_epochs():
    iter_counter.record_epoch_start(epoch)
    for i, data_i in enumerate(dataloader, start=iter_counter.epoch_iter):
        iter_counter.record_one_iteration()

        # Training
        # train generator
        if i % opt.D_steps_per_G == 0:
            stop_grad(trainer.pix2pix_model.netD)
            start_grad(trainer.pix2pix_model.netG)
            trainer.run_generator_one_step(data_i)

        # train discriminator
        stop_grad(trainer.pix2pix_model.netG)
        start_grad(trainer.pix2pix_model.netD)
        trainer.run_discriminator_one_step(data_i)

        # Visualizations
        if iter_counter.needs_printing():
            losses = trainer.get_latest_losses()
            visualizer.print_current_errors(epoch, iter_counter.epoch_iter,
                                            losses, iter_counter.time_per_iter)
            visualizer.plot_current_errors(losses, iter_counter.total_steps_so_far)

        if iter_counter.needs_displaying():
            visuals = OrderedDict([('input_label', data_i['label']),
                                   ('synthesized_image', trainer.get_latest_generated()),
                                   ('global_image', trainer.get_global_generated()),
                                   ('local_image', trainer.get_local_generated()),
                                   ('global_attention', trainer.get_global_attention()),
                                   ('local_attention', trainer.get_local_attention()),
                                   ('real_image', data_i['image'])])

            visualizer.display_current_results(visuals, epoch, iter_counter.total_steps_so_far)

        if iter_counter.needs_saving():
            is_better, cur_fid = fid_test.update(trainer.pix2pix_model, iter_counter.total_steps_so_far)
            if is_better:
                print(f"saving the currently best model epoch={epoch}, step={iter_counter.total_steps_so_far}")
                trainer.save('best')
                iter_counter.record_best_iter(cur_fid)
            print('saving the latest model (epoch %d, total_steps %d)' % (epoch, iter_counter.total_steps_so_far))
            trainer.save('latest')
            iter_counter.record_current_iter()

    trainer.update_learning_rate(epoch)
    iter_counter.record_epoch_end()

    if epoch % opt.save_epoch_freq == 0 or epoch == iter_counter.total_epochs:
        is_better, cur_fid = fid_test.update(trainer.pix2pix_model, iter_counter.total_steps_so_far)
        if is_better:
            print(f"saving the currently best model epoch={epoch}, step={iter_counter.total_steps_so_far}")
            trainer.save('best')
            iter_counter.record_best_iter(cur_fid)
        print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
        trainer.save('latest')
        trainer.save(epoch)
    
    jt.sync_all()
    jt.gc()
    
# after training
is_better, cur_fid = fid_test.update(trainer.pix2pix_model, iter_counter.total_steps_so_far)
if is_better:
    print(f"saving the currently best model epoch={epoch}, step={iter_counter.total_steps_so_far}")
    trainer.save('best')
    iter_counter.record_best_iter(cur_fid)
print('saving the model at the end of epoch %d, iters %d' % (epoch, iter_counter.total_steps_so_far))
trainer.save('latest')
trainer.save(epoch)

# training end
print('Training was successfully finished.')

