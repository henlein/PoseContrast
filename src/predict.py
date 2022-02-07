import os
import torch
import math
import numpy as np
from scipy.spatial.transform import Rotation as R
from math import radians
from PIL import Image, ImageFilter
from dataset.data_utils import process_viewpoint_label, TransLightning, resize_pad, random_crop
import torchvision.transforms as transforms
from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator, Estimator


#img_name = "../test_img/HICO_train2015_00000001.jpg"
img_name = "../test_img/Screenshot 2022-01-26 120213.png"
#img_name = "../test_img/simple-knife.jpg"
im = Image.open(img_name).convert('RGB')
im_pos = im.copy()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
im_transform = transforms.Compose([transforms.ToTensor(), normalize])

#im = im.crop((left, upper, right, lower))
im_flip = im.transpose(Image.FLIP_LEFT_RIGHT)
im = resize_pad(im, 224)
im_flip = resize_pad(im_flip, 224)
im = im_transform(im)
im_flip = im_transform(im_flip)
im = im_flip

# ================CREATE NETWORK============================ #
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

net_feat = resnet50(num_classes=128)
net_vp = BaselineEstimator(img_feature_dim=2048)

net_feat.cuda()
net_vp.cuda()

state = torch.load("../exps/PoseContrast_ObjectNet3d_ZeroShot/ckpt.pth")
net_feat.load_state_dict(state['net_feat'])
net_vp.load_state_dict(state['net_vp'])
net_feat.eval()
net_vp.eval()

im = im[None, :].cuda()
feat, _ = net_feat(im)
out = net_vp(feat)
vp_pred, score = net_vp.compute_vp_pred(out, True)
print(vp_pred)
print(score)

azi, ele, inp = vp_pred[0]
print(azi, ele, inp)
ele = ele - 90
rol = inp - 180
print(azi, ele, rol)
azi = radians(azi)
ele = radians(ele)
rol = radians(rol)
r = 3
loc_y = r * math.cos(ele) * math.cos(azi)
loc_x = r * math.cos(ele) * math.sin(azi)
loc_z = r * math.sin(ele)

print(loc_x, loc_y, loc_z + 0.5)

distance = np.sqrt(loc_x ** 2 + loc_y ** 2 + loc_z ** 2)

rotate_val = rol
rotate_axes = (loc_x / distance, loc_y / distance, loc_z / distance)

r = R.from_quat([rol, loc_x / distance, loc_y / distance, loc_z / distance])
print("..............")
up = np.array([0, 1, 0])
front = np.array([1, 0, 0])
print(r.apply(up))
print(r.apply(front))
#print(quat * up)
#print(quat * front)