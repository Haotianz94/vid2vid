python train.py --name djokovic_fg_stickonbg2full \
--dataroot 'datasets/vid2player/djokovic_fg_stickonbg2full' \
--dataset_mode pose \
--densepose_only \
--input_nc 3 --num_D 1 \
--resize_or_crop none \
--gpu_ids 0 --batchSize 4 --max_frames_per_gpu 1 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_t_step 4 \