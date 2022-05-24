python3 train.py --name djokovic_fg_stickonbg2full \
--dataroot '/home/haotian/Projects/vid2player/data/train_input/vid2vid/djokovic_fg_stickonbg2full' \
--dataset_mode pose \
--input_nc 3 --num_D 2 \
--resize_or_crop none \
--gpu_ids 4,5,6,7 --batchSize 8 --max_frames_per_gpu 3 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_t_step 4