python3 train.py --name federer_fg_dense2full \
--dataroot 'datasets/federer_fg_dense2full' \
--input_nc 3 --loadSize 320  --fineSize 320 \
--n_frames_total 30 \
--batchSize 8 \
--max_frames_per_gpu 1 \
--gpu_ids 4,5,6,7 \
--no_first_img
