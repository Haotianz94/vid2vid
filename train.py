### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import time
import os
import torch
from subprocess import call
import copy
from tqdm import tqdm
import cv2

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
    print('#training video frames = %d' % len(data_loader_train))

    if opt.eval_freq != 0:
        opt_val = copy.deepcopy(opt)
        opt_val.isTrain = False
        opt_val.phase = 'val'
        opt_val.start_frame = 0
        opt_val.nThreads = 1   # test code only supports nThreads = 1
        opt_val.batchSize = 1  # test code only supports batchSize = 1
        opt_val.serial_batches = True  # no shuffle
        opt_val.no_flip = True  # no flip
        data_loader_val = CreateDataLoader(opt_val)
        dataset_val = data_loader_val.load_data()
        dataset_val.loss_best = 1e9
        print('#val video frames = %d' % len(data_loader_val))

    ### initialize models
    models = create_model(opt)
    modelG, modelD, flowNet, optimizer_G, optimizer_D, optimizer_D_T = create_optimizer(opt, models)
    if opt.use_amp:
        scaler = torch.cuda.amp.GradScaler()
    else:
        scaler = None

    ### set parameters    
    n_gpus, tG, tD, tDB, s_scales, t_scales, input_nc, output_nc, \
        start_epoch, epoch_iter, print_freq, total_steps, iter_path, loss_best = init_params(opt, modelG, modelD, data_loader_train)
    visualizer = Visualizer(opt)
    if opt.eval_freq != 0:
        dataset_val.loss_best = loss_best    

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

                fake_B_prev = modelG.module.compute_fake_B_prev(real_B_prev, fake_B_prev_last, fake_B) # Haotian: need to figure out 
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
            loss_best = None
            if opt.eval_freq != 0 and total_steps % opt.eval_freq == 0:
                update_best = eval(modelG.module, dataset_val, visualizer, total_steps)
                loss_best = dataset_val.loss_best 

            ### save latest model
            save_models(opt, epoch, epoch_iter, total_steps, visualizer, iter_path, modelG, modelD, scaler, 
                update_best=update_best, loss_best=loss_best)            
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

def eval(model, dataset, visualizer, total_steps):
    opt = model.opt
    model.fake_B_prev = None
    model.opt.isTrain = False
    ReconLoss = torch.nn.L1Loss()

    start_of_seq = 3
    loss_sum = 0
    frame_cnt = 0
    frames = []
    visualizer.vis_print("Running evaluation ...")
    for i, data in tqdm(enumerate(dataset)):
        if data['change_seq']:
            model.fake_B_prev = None
            start_of_seq = 3

        _, _, height, width = data['A'].size()
        As = data['A'].view(1, -1, opt.input_nc, height, width)
        Bs = data['B'].view(1, -1, opt.output_nc, height, width) if len(data['B'].size()) > 2 else None
        inst = data['inst'].view(1, -1, 1, height, width) if len(data['inst'].size()) > 2 else None
        generated = model.inference(As, Bs, inst)
        device = generated[0].get_device()

        real_B = Bs[0][-1].detach().to(device) # 3 x h x w    
        fake_B = generated[0][0].detach() # 3 x h x w   

        ### add color shift
        diff = real_B - fake_B
        diff_crop = torch.cat((diff[:, :50], diff[:, -50:]), dim=1)
        diff_avg = torch.mean(diff_crop.view(-1, 3), dim=0).unsqueeze(1).unsqueeze(1)
        fake_B = fake_B + diff_avg

        if start_of_seq == 0:
            recon_loss = ReconLoss(real_B, fake_B)
            loss_sum += recon_loss.item()
            frame_cnt += 1
        else:
            start_of_seq -= 1

        if opt.pose_input_format == 'stick_on_bg':
            real_A = cv2.cvtColor(util.tensor2im(As[0][-1]), cv2.COLOR_RGB2BGR)
        elif opt.pose_input_format == 'stick_with_bg':
            real_A = cv2.cvtColor(util.tensor2im(As[0][-1][3:]), cv2.COLOR_RGB2BGR)
        real_B = cv2.cvtColor(util.tensor2im(real_B), cv2.COLOR_RGB2BGR)
        fake_B = cv2.cvtColor(util.tensor2im(fake_B), cv2.COLOR_RGB2BGR)
        frames.append(util.plot_image_grid([real_A, real_B, fake_B], nrow=3))

    loss = loss_sum / frame_cnt
    update_best = False
    eps = 0.0005
    if loss < dataset.loss_best + eps:
        util.sequence_to_video(frames, os.path.join(opt.checkpoints_dir, opt.name, 'val_best.mp4'), fps=25)
        visualizer.vis_print("Average val loss {:.6f} < current best loss {:.6f}, updating best model ...".format(loss, dataset.loss_best))
        dataset.loss_best = loss
        update_best = True
    else:
        visualizer.vis_print("Average val loss {:.6f} > current best loss {:.6f}, not update best model ...".format(loss, dataset.loss_best))

    errors = {'val': loss}
    visualizer.plot_current_errors(errors, total_steps)

    model.fake_B_prev = None
    model.opt.isTrain = True
    return update_best

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