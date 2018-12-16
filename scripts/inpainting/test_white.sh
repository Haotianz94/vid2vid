CUDA_VISIBLE_DEVICES=3 python3 test.py --name video_inpainting_white \
--dataroot '../../data_home/Vid2Vid/test/WhiteHole' \
--input_nc 3 --loadSize 384 \
--no_first_img --how_many 10000
 
