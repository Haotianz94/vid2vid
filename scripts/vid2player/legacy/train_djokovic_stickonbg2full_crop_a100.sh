python3 train.py --name djokovic_fg_stickonbg2full_crop \
--dataroot 'datasets/vid2player/djokovic_fg_stickonbg2full' \
--dataset_mode pose \
--pose_stick_input \
--input_nc 3 --num_D 2 \
--resize_or_crop crop --fineSize 256 \
--gpu_ids 0 --batchSize 4 --max_frames_per_gpu 4 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_t_step 4 \
--print_freq 100 --display_freq 100 --save_latest_freq 1000