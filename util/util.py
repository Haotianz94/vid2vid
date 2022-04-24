from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import inspect, re
import numpy as np
import os
import collections
from PIL import Image
import cv2
from collections import OrderedDict


def save_all_tensors(opt, real_A, fake_B, fake_B_first, fake_B_raw, real_B, flow_ref, conf_ref, flow, weight, modelD):
    if opt.label_nc != 0:
        input_image = tensor2label(real_A, opt.label_nc)
    elif opt.dataset_mode in ['pose', 'vid2player']:
        input_image = tensor2im(real_A)
        if real_A.size()[2] == 6:
            input_image = tensor2im(real_A[:, :, 3:])
        elif real_A.size()[2] == 17:
            input_image = tensor2im(sum([real_A[:, :, i:i+1] for i in range(14)]), normalize=False)
    else:
        c = 3 if opt.input_nc >= 3 else 1
        input_image = tensor2im(real_A[0, -1, :c], normalize=False)
    if opt.use_instance:
        edges = tensor2im(real_A[0, -1, -1:], normalize=False)
        input_image += edges[:,:,np.newaxis]
    
    if opt.add_face_disc:
        ys, ye, xs, xe = modelD.module.get_face_region(real_A[0, -1:])
        if ys is not None:
            input_image[ys, xs:xe, :] = input_image[ye, xs:xe, :] = input_image[ys:ye, xs, :] = input_image[ys:ye, xe, :] = 255 

    visual_list = [('input_image', input_image),
                   ('fake_image', tensor2im(fake_B)),
                   ('fake_first_image', tensor2im(fake_B_first)),
                   ('fake_raw_image', tensor2im(fake_B_raw)),
                   ('real_image', tensor2im(real_B)),                                                          
                   ('flow_ref', tensor2flow(flow_ref)),
                   ('conf_ref', tensor2im(conf_ref, normalize=False))]
    if flow is not None:
        visual_list += [('flow', tensor2flow(flow)),
                        ('weight', tensor2im(weight, normalize=False))]
    visuals = OrderedDict(visual_list)
    return visuals

# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8, normalize=True):
    if isinstance(image_tensor, list):
        image_numpy = []
        for i in range(len(image_tensor)):
            image_numpy.append(tensor2im(image_tensor[i], imtype, normalize))
        return image_numpy

    if isinstance(image_tensor, torch.autograd.Variable):
        image_tensor = image_tensor.data
    if len(image_tensor.size()) == 5:
        image_tensor = image_tensor[0, -1]
    if len(image_tensor.size()) == 4:
        image_tensor = image_tensor[0]
    image_tensor = image_tensor[:3]
    image_numpy = image_tensor.cpu().float().numpy()
    if normalize:
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    else:
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    #image_numpy = (np.transpose(image_numpy, (1, 2, 0)) * std + mean)  * 255.0        
    image_numpy = np.clip(image_numpy, 0, 255)
    if image_numpy.shape[2] == 1:        
        image_numpy = image_numpy[:,:,0]
    return image_numpy.astype(imtype)

def tensor2label(output, n_label, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float()    
    if output.size()[0] > 1:
        output = output.max(0, keepdim=True)[1]
    #print(output.size())
    output = Colorize(n_label)(output)
    output = np.transpose(output.numpy(), (1, 2, 0))
    #img = Image.fromarray(output, "RGB")
    return output.astype(imtype)

def tensor2flow(output, imtype=np.uint8):
    if isinstance(output, torch.autograd.Variable):
        output = output.data
    if len(output.size()) == 5:
        output = output[0, -1]
    if len(output.size()) == 4:
        output = output[0]
    output = output.cpu().float().numpy()
    output = np.transpose(output, (1, 2, 0))
    #mag = np.max(np.sqrt(output[:,:,0]**2 + output[:,:,1]**2)) 
    #print(mag)
    hsv = np.zeros((output.shape[0], output.shape[1], 3), dtype=np.uint8)
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(output[..., 0], output[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    return rgb

def add_dummy_to_tensor(tensors, add_size=0):
    if add_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [add_dummy_to_tensor(tensor, add_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        dummy = torch.zeros_like(tensors)[:add_size]
        tensors = torch.cat([dummy, tensors])
    return tensors

def remove_dummy_from_tensor(tensors, remove_size=0):
    if remove_size == 0 or tensors is None: return tensors
    if isinstance(tensors, list):
        return [remove_dummy_from_tensor(tensor, remove_size) for tensor in tensors]    
    
    if isinstance(tensors, torch.Tensor):
        tensors = tensors[remove_size:]
    return tensors

def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)

def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))

def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def uint82bin(n, count=8):
    """returns the binary of integer n, count refers to amount of bits"""
    return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])

def labelcolormap(N):
    if N == 35: # Cityscapes train
        cmap = np.array([(  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (  0,  0,  0), (111, 74,  0), ( 81,  0, 81),
                     (128, 64,128), (244, 35,232), (250,170,160), (230,150,140), ( 70, 70, 70), (102,102,156), (190,153,153),
                     (180,165,180), (150,100,100), (150,120, 90), (153,153,153), (153,153,153), (250,170, 30), (220,220,  0),
                     (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
                     (  0, 60,100), (  0,  0, 90), (  0,  0,110), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,142)], 
                     dtype=np.uint8)
    elif N == 20: # Cityscapes eval
        cmap = np.array([(128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153), (153,153,153), (250,170, 30), 
                         (220,220,  0), (107,142, 35), (152,251,152), ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), 
                         (  0,  0, 70), (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32), (  0,  0,  0)], 
                         dtype=np.uint8)
    else:
        cmap = np.zeros((N, 3), dtype=np.uint8)
        for i in range(N):
            r, g, b = 0, 0, 0            
            id = i
            for j in range(7):
                str_id = uint82bin(id)
                r = r ^ (np.uint8(str_id[-1]) << (7-j))
                g = g ^ (np.uint8(str_id[-2]) << (7-j))
                b = b ^ (np.uint8(str_id[-3]) << (7-j))
                id = id >> 3
            cmap[i, 0], cmap[i, 1], cmap[i, 2] = r, g, b             
    return cmap

def colormap(n):
    cmap = np.zeros([n, 3]).astype(np.uint8)
    for i in np.arange(n):
        r, g, b = np.zeros(3)

        for j in np.arange(8):
            r = r + (1 << (7-j))*((i & (1 << (3*j))) >> (3*j))
            g = g + (1 << (7-j))*((i & (1 << (3*j+1))) >> (3*j+1))
            b = b + (1 << (7-j))*((i & (1 << (3*j+2))) >> (3*j+2))

        cmap[i, :] = np.array([r, g, b])

    return cmap

class Colorize(object):
    def __init__(self, n=35):
        self.cmap = labelcolormap(n)
        self.cmap = torch.from_numpy(self.cmap[:n])

    def __call__(self, gray_image):
        size = gray_image.size()
        color_image = torch.ByteTensor(3, size[1], size[2]).fill_(0)

        for label in range(0, len(self.cmap)):
            mask = (label == gray_image[0]).cpu()
            color_image[0][mask] = self.cmap[label][0]
            color_image[1][mask] = self.cmap[label][1]
            color_image[2][mask] = self.cmap[label][2]

        return color_image


def make_joint_heatmaps(img_width, img_height, kps, sigma=7/4.0, method='edt'):

    def gaussian(img_width, img_height, center, var_x, var_y, theta):
        xv, yv = np.meshgrid(np.array(range(img_width)), np.array(range(img_height)),
                             sparse=False, indexing='xy')

        a = np.cos(theta) ** 2 / (2 * var_x) + np.sin(theta) ** 2 / (2 * var_y)
        b = -np.sin(2 * theta) / (4 * var_x) + np.sin(2 * theta) / (4 * var_y)
        c = np.sin(theta) ** 2 / (2 * var_x) + np.cos(theta) ** 2 / (2 * var_y)

        return np.exp(-(a * (xv - center[0]) * (xv - center[0]) +
                        2 * b * (xv - center[0]) * (yv - center[1]) +
                        c * (yv - center[1]) * (yv - center[1])))

    def distance_transform(img_width, img_height, center, radius=10):
        heatmap = np.zeros((img_height, img_width), dtype=np.float)
        for i in range(-radius, radius+1, 1):
            for j in range(-radius, radius+1, 1):
                x = int(np.round(center[0])) + i
                y = int(np.round(center[1])) + j
                if x >= 0 and x < img_width and y >= 0 and y < img_height:
                    d = np.sqrt(i*i + j*j)
                    if i == 0 and j == 0:
                        heatmap[y, x] = 1
                    elif d > radius:
                        heatmap[y, x] = 0
                    else:
                        heatmap[y, x] = np.sqrt(1. / d)
        return heatmap

    var = sigma ** 2
    n_joints = kps.shape[1]
    canvas = np.zeros((img_height, img_width, n_joints))
    for i in range(n_joints):
        if method == 'gaussian':
            canvas[:, :, i] = gaussian(img_width, img_height, kps[:2, i], var, var, 0)
        elif method == 'edt':
            canvas[:, :, i] = distance_transform(img_width, img_height, kps[:2, i])
    return canvas


def make_stick_figures(img_width, img_height, kps, draw_feet=False, draw_racket=False):
    class Pose:
        """
        DCPose Body 17
        """
        Nose = 0
        Neck = 1
        Head = 2
        LEar = 3
        REar = 4
        LShoulder = 5
        RShoulder = 6
        LElbow = 7
        RElbow = 8
        LWrist = 9
        RWrist = 10
        LHip = 11
        RHip = 12
        LKnee = 13
        RKnee = 14
        LAnkle = 15
        RAnkle = 16
        LToe = 17
        RToe = 18
        LHeel = 19
        RHeel = 20
        RacketHead = 21
        RacketTail = 22


    shift = np.array([0, 0])
    line_thick = 4
    dot_radius = 4

    kp_lines = [
        [Pose.RShoulder, Pose.RElbow],
        [Pose.RElbow, Pose.RWrist],
        [Pose.LShoulder, Pose.LElbow],
        [Pose.LElbow, Pose.LWrist],
        [Pose.RHip, Pose.RKnee],
        [Pose.RKnee, Pose.RAnkle],
        [Pose.LHip, Pose.LKnee],
        [Pose.LKnee, Pose.LAnkle],
        [Pose.RShoulder, Pose.LShoulder],
        [Pose.RHip, Pose.LHip],
    ]
    if draw_feet:
        kp_lines += [
        [Pose.LAnkle, Pose.LToe],
        [Pose.LAnkle, Pose.LHeel],
        [Pose.RAnkle, Pose.RToe],
        [Pose.RAnkle, Pose.RHeel],
        ]
    if draw_racket:
        kp_lines += [
        [Pose.RacketHead, Pose.RacketTail],
        ]
   
    # Hand crafted colors that are different from court background
    colors = [
        [255, 0,   255],
        [0,   127, 255],
        [127, 0,   0],
        [127, 127, 0],
        [127, 0,   127],
        [0,   127, 0],
        [255, 255, 127],
        [0,   255, 0],
        [255, 127, 127],
        [255, 0,   0],
        [0,   0,   255],
        [0,   0,   255],
       
        [0, 0, 200],
        [0, 0, 200],
        [0, 127, 255],
        [0, 127, 255],
        [127, 127, 127],
    ]

    kp_mask = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    # Draw mid shoulder / mid hip first for better visualization.
    mid_shoulder = (
        kps[:2, Pose.RShoulder] +
        kps[:2, Pose.LShoulder]) // 2 + shift
    mid_hip = (
        kps[:2, Pose.RHip] +
        kps[:2, Pose.LHip]) // 2 + shift
    nose_idx = Pose.Nose
    cv2.line(
        kp_mask, tuple(mid_shoulder.astype(np.int)), tuple((kps[:2, nose_idx] + shift).astype(np.int)),
        color=colors[0], thickness=line_thick, lineType=cv2.LINE_AA)
    cv2.line(
        kp_mask, tuple(mid_shoulder.astype(np.int)), tuple(mid_hip.astype(np.int)),
        color=colors[1], thickness=line_thick, lineType=cv2.LINE_AA)

    # Draw the keypoints.
    for l in range(len(kp_lines)):
        i1 = kp_lines[l][0]
        i2 = kp_lines[l][1]
        p1 = tuple((kps[:2, i1] + shift).astype(np.int))
        p2 = tuple((kps[:2, i2] + shift).astype(np.int))
        if p1[0] > 0 and p2[0] > 0:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l+2], thickness=line_thick, lineType=cv2.LINE_AA)
        if p1[0] > 0:
            cv2.circle(
                kp_mask, p1,
                radius=dot_radius, color=colors[l+2], thickness=-1, lineType=cv2.LINE_AA)
        if p2[0] > 0:
            cv2.circle(
                kp_mask, p2,
                radius=dot_radius, color=colors[l+2], thickness=-1, lineType=cv2.LINE_AA)
    return kp_mask

###############################################################################
# Image visualization utils
###############################################################################

def make_image_grid(images, nrow=1, padding=2, padding_color=(255, 255, 255)):
    """
    nrow: how many images will be in one row
    """
    def get_up_left_loc(idx):
        x, y =  idx % nrow, idx // nrow
        return x * (w + padding) + padding, y * (h + padding) + padding

    ncol = int(np.ceil(len(images) / nrow))
    h, w = images[0].shape[:2]
    H, W = ncol * (h + padding) + padding, nrow * (w + padding) + padding
    canvas = np.zeros((H, W, 3), dtype=np.uint8) + np.array(padding_color)
    for idx, img in enumerate(images):
        assert (h, w) == img.shape[:2]
        x, y = get_up_left_loc(idx)
        canvas[y : y + h, x : x + w] = img.copy()
    return canvas.astype(np.uint8)

def plot_image_grid(images_np, nrow=1, padding=2, padding_color=(255, 255, 255), 
        factor=1, resize=None, output_path=None, post_process=None):
    """Draws images in a grid
    
    Args:
        images_np: list of images, each image is np.array of size 3xHxW of 1xHxW
        nrow: how many images will be in one row
        factor: size if the plt.figure 
        interpolation: interpolation used in plt.imshow
    """
    import torch
    import torchvision

    if resize is not None:
        images_np = [cv2.resize(img, resize) for img in images_np] 

    # images_torch = [torch.from_numpy(img.transpose(2, 0, 1)) for img in images_np]
    # grid = torchvision.utils.make_grid(images_torch, nrow, padding, pad_value=255).numpy().transpose(1, 2, 0).copy() 
    grid = make_image_grid(images_np, nrow, padding, padding_color)

    # without copy, cv2 functions does not work
    if post_process:
        grid = post_process(grid)

    if output_path is not None:
        cv2.imwrite(output_path, grid)

    return grid

def sequence_to_video(images, output_path, fps=25, remove_avi=True):
    out = cv2.VideoWriter(output_path.replace('mp4', 'avi'), cv2.VideoWriter_fourcc('X','V','I','D'), fps, 
                      (images[0].shape[1], images[0].shape[0]))
    for image in images:
        out.write(image)
    out.release()

    convert_cmd = "ffmpeg -y -i {} -c:v libx264 -preset slow -crf 17 -c:a copy {} -loglevel quiet".format(
        output_path.replace('mp4', 'avi'), output_path)
    ret = os.system(convert_cmd)
    if ret == 0:
        print("Dumped {} frames to {}".format(len(images), output_path))
    else:
        print("Dump video failed!!!".format(len(images)))
    if remove_avi:
        os.remove(output_path.replace('mp4', 'avi'))