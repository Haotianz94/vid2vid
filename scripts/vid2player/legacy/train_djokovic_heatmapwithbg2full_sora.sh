python3 train.py --name djokovic_fg_heatmapwithbg2full_sora \
--dataroot '/home/haotian/Projects/vid2player/data/train_input/vid2vid/djokovic_fg_stick2full' \
--dataset_mode pose \
--pose_input_format 'heatmap_with_bg' \
--input_nc 17 --num_D 2 \
--resize_or_crop crop --fineSize 128 \
--gpu_ids 1,0 --batchSize 1 --max_frames_per_gpu 1 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_t_step 4 \
--print_freq 3 --display_freq 3 