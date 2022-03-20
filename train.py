### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call
import copy

from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model, create_optimizer, init_params, save_models, update_models
import util.util as util
from util.visualizer import Visualizer


def train():
    opt = TrainOptions().parse()
    if opt.debug:
        opt.display_freq = 1
        opt.print_freq = 1    
        opt.nThreads = 1

    ### initialize dataset train and val
    data_loader_train = CreateDataLoader(opt)
    dataset_train = data_loader_train.load_data()
    print('#training videos = %d' % len(data_loader_train))

    if opt.eval_freq != 0:
        opt_val = copy.deepcopy(opt)
        opt_val.isTrain = False
        opt_val.phase = 'val'
        opt_val.start_frame = 0
        data_loader_val = CreateDataLoader(opt_val)
        dataset_val = data_loader_val.load_data()
        print('#val videos = %d' % len(data_loader_val))

    ### initialize models
    models = create_model(opt)
    modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    ### set parameters    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path = init_params(opt, modelG, modelD, data_loader_train)
    visualizer = Visualizer(opt)    

    ### real training starts here  
    for epoch in range(start_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()    
        for idx, data in enumerate(dataset_train, start=epoch_iter):        
            if total_steps % print_freq == 0:
                iter_start_time = time.time()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            n_frames_total, n_frames_load, t_len = data_loader_train.dataset.init_data_params(data, n_gpus, tG)
            fake_B_prev_last, frames_all = data_loader_train.dataset.init_data(t_scales)

            for i in range(0, n_frames_total, n_frames_load):
                input_A, input_B, inst_A = data_loader_train.dataset.prepare_data(data, i, input_nc, output_nc)
                
                ###################################### Forward Pass ##########################
                ####### generator
                if opt.use_amp:
                    with torch.cuda.amp.autocast():                  
                        fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, inst_A, fake_B_prev_last)
                else:
                    fake_B, fake_B_raw, flow, weight, real_A, real_Bp, fake_B_last = modelG(input_A, input_B, inst_A, fake_B_prev_last)

                ####### discriminator            
                ### individual frame discriminator          
                real_B_prev, real_B = real_Bp[:, :-1], real_Bp[:, 1:]   # the collection of previous and current real frames
                
                if opt.use_amp:
                    with torch.cuda.amp.autocast(): 
                        flow_ref, conf_ref = flowNet(real_B, real_B_prev)       # reference flows and confidences 
                else:
                    flow_ref, conf_ref = flowNet(real_B, real_B_prev)       # reference flows and confidences 

                fake_B_prev = modelG.module.compute_fake_B_prev(real_B_prev, fake_B_prev_last, fake_B)
                fake_B_prev_last = fake_B_last
                
                if opt.use_amp:
                    with torch.cuda.amp.autocast():                
                        losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                else:
                    losses = modelD(0, reshape([real_B, fake_B, fake_B_raw, real_A, real_B_prev, fake_B_prev, flow, weight, flow_ref, conf_ref]))
                losses = [ torch.mean(x) if x is not None else 0 for x in losses ]
                loss_dict = dict(zip(modelD.module.loss_names, losses))

                ### temporal discriminator                
                # get skipped frames for each temporal scale
                if opt.use_amp:
                    with torch.cuda.amp.autocast(): 
                        frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                                real_B, fake_B, flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet)   
                else:
                    frames_all, frames_skipped = modelD.module.get_all_skipped_frames(frames_all, \
                            real_B, fake_B, flow_ref, conf_ref, t_scales, tD, n_frames_load, i, flowNet)                         

                # run discriminator for each temporal scale
                loss_dict_T = []
                for s in range(t_scales):                
                    if frames_skipped[0][s] is not None:
                        if opt.use_amp:
                            with torch.cuda.amp.autocast():                
                                losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        else:
                            losses = modelD(s+1, [frame_skipped[s] for frame_skipped in frames_skipped])
                        losses = [ torch.mean(x) if not isinstance(x, int) else x for x in losses ]
                        loss_dict_T.append(dict(zip(modelD.module.loss_names_T, losses)))

                # collect losses
                loss_G, loss_D, loss_D_T, t_scales_act = modelD.module.get_losses(loss_dict, loss_dict_T, t_scales)

                ###################################### Backward Pass #################################                 
                # update generator weights     
                loss_backward(opt, loss_G, optimizer_G, scaler)                

                # update individual discriminator weights                
                loss_backward(opt, loss_D, optimizer_D, scaler)

                # update temporal discriminator weights
                for s in range(t_scales_act):                    
                    loss_backward(opt, loss_D_T[s], optimizer_D_T[s], scaler)

                if i == 0: fake_B_first = fake_B[0, 0]   # the first generated image in this sequence

            if opt.debug:
                call(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"]) 

            ############## Display results and errors ##########
            ### print out errors
            if total_steps % print_freq == 0:
                t = (time.time() - iter_start_time) / print_freq
                errors = {k: v.data.item() if not isinstance(v, int) else v for k, v in loss_dict.items()}
                for s in range(len(loss_dict_T)):
                    errors.update({k+str(s): v.data.item() if not isinstance(v, int) else v for k, v in loss_dict_T[s].items()})            
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)
                visualizer.plot_current_errors(errors, total_steps)

            ### display output images
            if total_steps % opt.display_freq == 0:                
                visuals = util.save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD)                
                visualizer.display_current_results(visuals, epoch, total_steps)

            ### run val
            update_best = False
            if opt.eval_freq != 0 and total_steps % opt.eval_freq == 0:
                pass
                # update_best = eval(dataset_val) 

            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, scaler, update_best=update_best)            
            if epoch_iter > len(data_loader_train) - opt.batchSize:
                epoch_iter = 0
                break
           
        # end of epoch 
        iter_end_time = time.time()
        visualizer.vis_print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))

        ### save model for this epoch and update model params
        save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, scaler, end_of_epoch=True)
        update_models(opt, epoch, modelG, modelD, data_loader_train) 

def eval(dataset):
    for i, data in enumerate(dataset):
        if data['change_seq']:
            model.fake_B_prev = None

        _, _, height, width = data['A'].size()
        A = Variable(data['A']).view(1, -1, input_nc, height, width)
        B = Variable(data['B']).view(1, -1, opt.output_nc, height, width) if len(data['B'].size()) > 2 else None
        inst = Variable(data['inst']).view(1, -1, 1, height, width) if len(data['inst'].size()) > 2 else None
        generated = model.inference(A, B, inst)
        
        if opt.label_nc != 0:
            real_A = util.tensor2label(generated[1], opt.label_nc)
        else:
            c = 3 if opt.input_nc == 3 else 1
            real_A = util.tensor2im(generated[1][:c], normalize=False)    
            
        visual_list = [('real_A', real_A), 
                       ('fake_B', util.tensor2im(generated[0].data[0]))]
        visuals = OrderedDict(visual_list) 
        img_path = data['A_path']
        print('process image... %s' % img_path)
        visualizer.save_images(save_dir, visuals, img_path)

def loss_backward(opt, loss, optimizer, scaler=None):
    optimizer.zero_grad()                
    # if opt.fp16:
    #     from apex import amp
    #     with amp.scale_loss(loss, optimizer) as scaled_loss: 
    #         scaled_loss.backward()
    #     optimizer.step()
    if opt.use_amp:
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        loss.backward()
        optimizer.step()

def reshape(tensors):
    if tensors is None: return None
    if isinstance(tensors, list):
        return [reshape(tensor) for tensor in tensors]    
    _, _, ch, h, w = tensors.size()
    return tensors.contiguous().view(-1, ch, h, w)

if __name__ == "__main__":
   train()