python train.py --name djokovic_fg_wimbledon_augment_v1 \
--dataroot ~/Projects/vid2player/data/train_input/vid2vid/djokovic_fg_wimbledon \
--dataset_mode vid2player \
--pose_input_format stick_with_bg --pose_input body,feet,racket \
--input_nc 6 --num_D 2 --resize_or_crop none \
--is_fg --use_relight \
--no_first_img \
--test_only