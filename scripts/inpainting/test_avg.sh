CUDA_VISIBLE_DEVICES=3 python3 test.py --name video_inpainting_g1 \
--dataroot '../../data_home/Vid2Vid/test/AvgHole' \
--input_nc 3 --loadSize 384 \
--no_first_img --how_many 10000
 
