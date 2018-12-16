CUDA_VISIBLE_DEVICES=3 python3 test.py --name video_inpainting_noise \
--dataroot '../../data_home/Vid2Vid/test/NoiseHole' \
--input_nc 3 --loadSize 384 \
--no_first_img --how_many 10000
 
