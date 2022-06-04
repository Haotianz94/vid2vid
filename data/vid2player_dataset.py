import os
import torch
from PIL import Image
import numpy as np
import json
import cv2

from data.base_dataset import BaseDataset, get_img_params, get_transform, get_video_params, concat_frame
from data.image_folder import make_grouped_dataset, check_path_valid
from data.keypoint2img import read_keypoints
from util.util import make_joint_heatmaps, make_stick_figures, crop_image


class Vid2playerDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot 

        self.dir_bg = os.path.join(opt.dataroot, opt.phase + '_bg')                
        self.dir_kp = os.path.join(opt.dataroot, opt.phase + '_keypoints')
        if opt.use_relight:
            self.dir_img = os.path.join(opt.dataroot, opt.phase + '_img_relight')
        else:
            self.dir_img = os.path.join(opt.dataroot, opt.phase + '_img_real')
        print('Using trainB images from', self.dir_img)
        self.img_paths = sorted(make_grouped_dataset(self.dir_img))
        # No longer load pre-rendered stick images
        # if self.opt.pose_input_format == 'stick_on_bg':
        #     self.dir_stick = os.path.join(opt.dataroot, opt.phase + '_stickonbg')                
        #     self.stick_paths = sorted(make_grouped_dataset(self.dir_stick))
        #     check_path_valid(self.stick_paths, self.img_paths)
        # elif self.opt.pose_input_format == 'stick_with_bg':
        #     self.dir_stick = os.path.join(opt.dataroot, opt.phase + '_stickonblack')                
        #     self.stick_paths = sorted(make_grouped_dataset(self.dir_stick))
        #     check_path_valid(self.stick_paths, self.img_paths)
        #     self.bg_paths = sorted(make_grouped_dataset(self.dir_bg))
        #     check_path_valid(self.bg_paths, self.img_paths)
        if self.opt.pose_input_format in ['stick_on_bg', 'stick_with_bg', 'heatmap_with_bg']:
            self.kp_paths = sorted(make_grouped_dataset(self.dir_kp))
            check_path_valid(self.kp_paths, self.img_paths)
            self.bg_paths = sorted(make_grouped_dataset(self.dir_bg))
            check_path_valid(self.bg_paths, self.img_paths)
        else:
            raise Exception("Pose input format not supported {}".format(self.opt.pose_input_format))
        if opt.phase == 'train' and (opt.add_recon_bg_loss or opt.generate_mask or opt.augment_background):
            self.dir_mask = os.path.join(opt.dataroot, opt.phase + '_mask')
            self.mask_paths = sorted(make_grouped_dataset(self.dir_mask))
            check_path_valid(self.mask_paths, self.img_paths)
        if opt.phase == 'train' and opt.augment_background:
            self.dir_crop = os.path.join(opt.dataroot, opt.phase + '_crop')
            self.crop_paths = sorted(make_grouped_dataset(self.dir_crop))
            check_path_valid(self.crop_paths, self.img_paths)
            self.background = np.array(Image.open(
                os.path.join(opt.dataroot, opt.background_path)).convert('RGB'))
            assert self.background is not None
        self.init_frame_idx(self.img_paths)

    def __getitem__(self, index):
        opt = self.opt
        A, B, _, seq_idx = self.update_frame_idx(self.img_paths, index)
        M = None
        img_paths = self.img_paths[seq_idx]        
        n_frames_total, start_idx, t_step = get_video_params(self.opt, self.n_frames_total, len(img_paths), self.frame_idx)
        augment_background = False
        dx, dy, dl = 0, 0, None
        if opt.augment_background and opt.phase == 'train' and np.random.rand() < opt.augment_ratio:
            dx, dy = np.random.randint(-100, 100, 2)
            augment_background = True
            if opt.augment_lighting:
                dl = np.random.randint(-30, 30, 1)[0]
        
        img = Image.open(img_paths[start_idx]).convert('RGB')     
        size = img.size
        params = get_img_params(self.opt, size)
        params['thick'] = self.opt.is_fg == True
        params['dl'] = dl

        frame_range = list(range(n_frames_total)) if (self.opt.isTrain or self.A is None) else [self.opt.n_frames_G-1]
        for i in frame_range:
            # from sequence of seq_idx, sample n_frames_total frames start at start_idx with stride of t_step
            # this sampled seq serve as one item in the batch

            # No longer load pre-rendered stick images
            # if self.opt.pose_input_format == 'stick_on_bg':
            #     stick_path = self.stick_paths[seq_idx][start_idx + i * t_step]
            #     Ai = self.get_image(stick_path, size, params, input_type='img')
            # elif self.opt.pose_input_format == 'stick_with_bg':
            #     stick_path = self.stick_paths[seq_idx][start_idx + i * t_step]
            #     Si = self.get_image(stick_path, size, params, input_type='img')
            #     bg_path = self.bg_paths[seq_idx][start_idx + i * t_step]
            #     Bi = self.get_image(bg_path, size, params, input_type='img')
            #     Ai = torch.cat([Bi, Si])
            if opt.pose_input_format == 'stick_on_bg':
                kp_path = self.kp_paths[seq_idx][start_idx + i * t_step]
                bg_path = self.bg_paths[seq_idx][start_idx + i * t_step]
                params['bg_path'] = bg_path
                Ai = self.get_image(kp_path, size, params, input_type='stick_on_bg')
            elif opt.pose_input_format == 'stick_with_bg':
                kp_path = self.kp_paths[seq_idx][start_idx + i * t_step]
                Si = self.get_image(kp_path, size, params, input_type='stick_on_black')
                bg_path = self.bg_paths[seq_idx][start_idx + i * t_step]
                Gi = self.get_image(bg_path, size, params, input_type='img')
                Ai = torch.cat([Gi, Si])
            elif opt.pose_input_format == 'heatmap_with_bg':
                bg_path = self.bg_paths[seq_idx][start_idx + i * t_step]
                Gi = self.get_image(bg_path, size, params, input_type='img')
                kp_path = self.kp_paths[seq_idx][start_idx + i * t_step]
                Hi = self.get_image(kp_path, size, params, input_type='heatmap')
                Ai = torch.cat([Hi, Gi])

            img_path = img_paths[start_idx + i * t_step]
            Bi = self.get_image(img_path, size, params, input_type='img')
            if augment_background:
                mask_path = self.mask_paths[seq_idx][start_idx + i * t_step]
                Mi = self.get_image(mask_path, size, params, input_type='mask') # binary
                crop_path = self.crop_paths[seq_idx][start_idx + i * t_step]
                x, y = json.load(open(crop_path, 'r'))
                if opt.background_path.startswith('wimbledon'): y -= 100
                crop_box = [x + dx, y + dy, x + dx + size[0], y + dy + size[1]]
                bg_crop = crop_image(self.background, crop_box) # in RGB 
                if 'crop' in self.opt.resize_or_crop:
                    bg_crop = bg_crop[params['crop_pos'][0] : params['crop_pos'][0] + params['crop_size'][0],
                                params['crop_pos'][1] : params['crop_pos'][1] + params['crop_size'][1]]
                if opt.augment_lighting:
                    bg_crop = self.adjust_brightness(bg_crop, dl)
                transform = get_transform(opt, params)
                Gi = transform(Image.fromarray(bg_crop))
                Ai = torch.cat([Gi, Si])
                Bi = Gi * (1 - Mi) + Bi * Mi
            
            # Ai, Bi = self.crop(Ai), self.crop(Bi) # only crop the central half region to save time

            A = concat_frame(A, Ai, n_frames_total)
            B = concat_frame(B, Bi, n_frames_total)
            if opt.phase == 'train' and (opt.add_recon_bg_loss or opt.generate_mask):
                mask_path = self.mask_paths[seq_idx][start_idx + i * t_step]
                Mi = self.get_image(mask_path, size, params, input_type='mask')
                M = concat_frame(M, Mi, n_frames_total)
        
        if not self.opt.isTrain:
            self.A, self.B = A, B
            self.frame_idx += 1            
        change_seq = False if self.opt.isTrain else self.change_seq
        return_list = {'A': A, 'B': B, 'inst': 0, 'mask': M if M is not None else 0, 'A_path': img_path, 'change_seq': change_seq}

        return return_list

    def get_image(self, A_path, size, params, input_type, bg_path=None):
        if input_type == 'heatmap':
            A_img = make_joint_heatmaps(size[0], size[1], np.array(json.load(open(A_path, 'r'))))
            if 'crop' in self.opt.resize_or_crop:
                A_img = A_img[params['crop_pos'][0] : params['crop_pos'][0] + params['crop_size'][0],
                              params['crop_pos'][1] : params['crop_pos'][1] + params['crop_size'][1]]
            return torch.from_numpy(A_img.transpose(2, 0, 1)).float()
        elif input_type in ['stick_on_black', 'stick_on_bg']:
            kps = np.array(json.load(open(A_path, 'r')))
            if self.opt.add_input_noise != 'none':
                if self.opt.add_input_noise == 'sigma5,cap10':
                    kps_noise = np.clip(np.random.normal(0, 5, kps.shape[0] * kps.shape[1]), -10, 10).reshape(kps.shape[0], -1)
                elif self.opt.add_input_noise == 'sigma2,cap5':
                    kps_noise = np.clip(np.random.normal(0, 2, kps.shape[0] * kps.shape[1]), -5, 5).reshape(kps.shape[0], -1)
                else:
                    raise Exception("Not support option: add_input_noise="+self.opt.add_input_noise)
                kps = kps + kps_noise

            background = None
            if input_type == 'stick_on_bg':
                background = cv2.imread(bg_path)
            A_img_np = make_stick_figures(size[0], size[1], kps, background=background, thick=params['thick'],
                draw_feet=True if 'feet' in self.opt.pose_input else False,
                draw_racket=True if 'racket' in self.opt.pose_input else False)
            A_img = Image.fromarray(A_img_np[:, :, ::-1])
        elif input_type == 'mask':
            A_img = Image.open(A_path).convert('L')
            A_img = np.array(A_img) > 127 # store mask where fg in 1 and bg in 0
            if 'crop' in self.opt.resize_or_crop:
                A_img = A_img[params['crop_pos'][0] : params['crop_pos'][0] + params['crop_size'][0],
                              params['crop_pos'][1] : params['crop_pos'][1] + params['crop_size'][1]]
            return torch.from_numpy(A_img).unsqueeze(0).float()
        else:
            A_img = Image.open(A_path).convert('RGB')
            if self.opt.augment_lighting and params.get('dl', None) is not None:
                A_img_np = self.adjust_brightness(np.array(A_img), params['dl'])
                A_img = Image.fromarray(A_img_np)

        is_img = input_type == 'img'
        method = Image.BICUBIC if is_img else Image.NEAREST
        transform_scaleA = get_transform(self.opt, params, method=method)
        A_scaled = transform_scaleA(A_img)
        return A_scaled

    def crop(self, Ai):
        w = Ai.size()[2]
        base = 32
        x_cen = w // 2
        bs = int(w * 0.25) // base * base
        return Ai[:,:,(x_cen-bs):(x_cen+bs)]
               
    def normalize_pose(self, A_img, target_yc, target_len, first=False):
        w, h = A_img.size
        A_np = np.array(A_img)  

        if first == True:          
            part_labels = A_np[:,:,2]            
            part_coords = np.nonzero((part_labels == 1) | (part_labels == 2))
            y, x = part_coords[0], part_coords[1]

            ys, ye = y.min(), y.max()                    
            min_i, max_i = np.argmin(y), np.argmax(y)
            v_min = A_np[y[min_i], x[min_i], 1] / 255
            v_max = A_np[y[max_i], x[max_i], 1] / 255
            ylen = (ye-ys) / (v_max-v_min)
            yc = (0.5-v_min) / (v_max-v_min) * (ye-ys) + ys            
            
            ratio = target_len / ylen
            offset_y = int(yc - (target_yc / ratio))
            offset_x = int(w * (1 - 1/ratio) / 2)        

            padding = int(max(0, max(-offset_y, int(offset_y + h/ratio) - h)))
            padding = int(max(padding, max(-offset_x, int(offset_x + w/ratio) - w)))
            offset_y += padding
            offset_x += padding            
            self.offset_y, self.offset_x = offset_y, offset_x
            self.ratio, self.padding = ratio, padding

        p = self.padding
        A_np = np.pad(A_np, ((p,p),(p,p),(0,0)), 'constant', constant_values=0)
        A_np = A_np[self.offset_y:int(self.offset_y + h/self.ratio), self.offset_x:int(self.offset_x + w/self.ratio):, :]        
        A_img = Image.fromarray(A_np)
        A_img = A_img.resize((w, h))
        return A_img
    
    def adjust_brightness(self, img, value):
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = np.clip((hsv[:,:,2].astype(int) + value), 0, 255)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return img

    def __len__(self):        
        return sum(self.frames_count)

    def name(self):
        return 'Vid2playerDataset'