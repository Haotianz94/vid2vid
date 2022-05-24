python3 test.py --name djokovic_fg_stickwithbg2relight \
--dataroot '/home/haotian/Projects/vid2player/data/train_input/vid2vid/djokovic_fg_stick2full_v2' \
--dataset_mode vid2player \
--pose_input_format 'stick_with_bg' --pose_input 'body' --use_relight \
--input_nc 6 \
--resize_or_crop none \
--gpu_ids 1 \
--no_first_img \
--how_many 100000
