python3 train.py --name nadal_fg_dense2full \
--dataroot 'datasets/vid2player/nadal_fg_dense2full' \
--dataset_mode pose \
--densepose_only \
--input_nc 3 --num_D 2 \
--resize_or_crop none \
--gpu_ids 0,1,2,3 --batchSize 8 --max_frames_per_gpu 3 \
--niter 5 --niter_decay 5 \
--no_first_img --n_frames_total 12 --max_t_step 4 \
