CUDA_VISIBLE_DEVICES=2 python3 train.py --name video_inpainting_noise \
--dataroot '/mnt/ssd/tmp/hazhang/Vid2Vid/train/NoiseHole' \
--input_nc 3 --loadSize 384 \
--nThreads 0 \
--niter_step 20 \
--n_frames_total 30 \
--max_frames_per_gpu 1 \
