import bpy
from PIL import ImageDraw
import os
import torch
import cv2
import math
import numpy as np
from math import radians
from PIL import Image, ImageFilter
from dataset.data_utils import process_viewpoint_label, TransLightning, resize_pad, random_crop
import torchvision.transforms as transforms
from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator, Estimator


# Crop and Resize the image and paste it to the bg
def crop_resize_paste(im, bg, left, upper, right, lower):
    # crop the RGBA image according to alpha channel
    bbox = im.getbbox()
    im = im.crop(bbox)

    # resize & padding the rendering then paste on the bg
    w, h = im.size
    target_w, target_h = right - left, lower - upper
    ratio = min(float(target_w) / w, float(target_h / h))
    new_size = (int(w * ratio), int(h * ratio))
    im = im.resize(new_size, Image.BILINEAR)
    bg.paste(im, (left + (target_w - new_size[0]) // 2, upper + (target_h - new_size[1]) // 2))


# create a lamp with an appropriate energy
def makeLamp(lamp_name, rad):
    # Create new lamp data block
    lamp_data = bpy.data.lamps.new(name=lamp_name, type='POINT')
    lamp_data.energy = rad
    # modify the distance when the object is not normalized
    # lamp_data.distance = rad * 2.5

    # Create new object with our lamp data block
    lamp_object = bpy.data.objects.new(name=lamp_name, object_data=lamp_data)

    # Link lamp object to the scene so it'll appear in this scene
    scene = bpy.context.scene
    scene.objects.link(lamp_object)
    return lamp_object


def parent_obj_to_camera(b_camera):
    # set the parenting to the origin
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty

    scn = bpy.context.scene
    scn.objects.link(b_empty)
    scn.objects.active = b_empty
    return b_empty


def clean_obj_lamp_and_mesh(context):
    scene = context.scene
    objs = bpy.data.objects
    meshes = bpy.data.meshes
    for obj in objs:
        if obj.type == "MESH" or obj.type == 'LAMP':
            scene.objects.unlink(obj)
            objs.remove(obj)
    for mesh in meshes:
        meshes.remove(mesh)


def angles_to_matrix(angles):
    """Compute the rotation matrix from euler angles for a mini-batch"""
    azi = angles[:, 0]
    ele = angles[:, 1]
    rol = angles[:, 2]
    element1 = (torch.cos(rol) * torch.cos(azi) - torch.sin(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element2 = (torch.sin(rol) * torch.cos(azi) + torch.cos(rol) * torch.cos(ele) * torch.sin(azi)).unsqueeze(1)
    element3 = (torch.sin(ele) * torch.sin(azi)).unsqueeze(1)
    element4 = (-torch.cos(rol) * torch.sin(azi) - torch.sin(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element5 = (-torch.sin(rol) * torch.sin(azi) + torch.cos(rol) * torch.cos(ele) * torch.cos(azi)).unsqueeze(1)
    element6 = (torch.sin(ele) * torch.cos(azi)).unsqueeze(1)
    element7 = (torch.sin(rol) * torch.sin(ele)).unsqueeze(1)
    element8 = (-torch.cos(rol) * torch.sin(ele)).unsqueeze(1)
    element9 = (torch.cos(ele)).unsqueeze(1)
    return torch.cat((element1, element2, element3, element4, element5, element6, element7, element8, element9), dim=1)

def render_obj(obj, output_dir, azi, ele, rol, name, shape=[512, 512], forward=None, up=None):
    clean_obj_lamp_and_mesh(bpy.context)

    # Setting up the environment
    scene = bpy.context.scene
    scene.render.resolution_x = shape[1]
    scene.render.resolution_y = shape[0]
    scene.render.resolution_percentage = 100
    scene.render.alpha_mode = 'TRANSPARENT'

    # Camera setting
    cam = scene.objects['Camera']
    cam_constraint = cam.constraints.new(type='TRACK_TO')
    cam_constraint.track_axis = 'TRACK_NEGATIVE_Z'
    cam_constraint.up_axis = 'UP_Y'
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # Light setting
    lamp_object = makeLamp('Lamp1', 5)
    lamp_add = makeLamp('Lamp2', 1)

    if forward is not None and up is not None:
        bpy.ops.import_scene.obj(filepath=obj, axis_forward=forward, axis_up=up)
    else:
        bpy.ops.import_scene.obj(filepath=obj)

    # normalize it and set the center
    for object in bpy.context.scene.objects:
        if object.name in ['Camera', 'Lamp'] or object.type == 'EMPTY':
            continue
        bpy.context.scene.objects.active = object
        max_dim = max(object.dimensions)
        object.dimensions = object.dimensions / max_dim if max_dim != 0 else object.dimensions

    # Output setting
    scene.render.image_settings.file_format = 'PNG'
    scene.render.filepath = os.path.join(output_dir, name)

    # scene.render.filepath = os.path.join(output_dir, name + '_render_%03d_%03d_%03d' % (int(azi), int(ele), int(rol)))

    # transform Euler angles from degrees into radians
    azi = radians(azi)
    ele = radians(ele)
    rol = radians(rol)

    r = 3
    loc_y = r * math.cos(ele) * math.cos(azi)
    loc_x = r * math.cos(ele) * math.sin(azi)
    loc_z = r * math.sin(ele)
    print(loc_x, loc_y, loc_z)

    print("===========")
    cam.location = (loc_x, loc_y, loc_z + 0.5)
    bpy.ops.render.render(write_still=True)



    cam.location = (loc_x, loc_y, loc_z + 0.5)
    lamp_object.location = (loc_x, loc_y, 10)
    lamp_add.location = (loc_x, loc_y, -10)

    bpy.ops.render.render(write_still=True)

    # Change the in-plane rotation
    cam_ob = bpy.context.scene.camera
    bpy.context.scene.objects.active = cam_ob  # select the camera object
    distance = np.sqrt(loc_x ** 2 + loc_y ** 2 + loc_z ** 2)
    bpy.ops.transform.rotate(value=90, axis=(loc_x / distance, loc_y / distance, loc_z / distance),
                             constraint_axis=(False, False, False), constraint_orientation='GLOBAL', mirror=False,
                             proportional='DISABLED', proportional_edit_falloff='SMOOTH',
                             proportional_size=1)
    print(loc_x / distance, loc_y / distance, loc_z / distance)
    print(rol)
    print(rol * 180 / math.pi)
    bpy.ops.render.render(write_still=True)




img_name = "../test_img/Screenshot 2022-01-26 120213.png"
img_name = "../test_img/horse.png"

obj_path = "../data/Pix3D/model/chair/IKEA_BERNHARD/model.obj"
save_name = "PoseImg.png"
out_dir = ".../test_img/out"

img = Image.open(img_name).convert('RGB')
im_pos = img.copy()

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
im_transform = transforms.Compose([transforms.ToTensor(), normalize])

#im = im.crop((left, upper, right, lower))
im_flip = img.transpose(Image.FLIP_LEFT_RIGHT)
im = resize_pad(img, 224)
im_flip = resize_pad(im_flip, 224)
im = im_transform(im)
im_flip = im_transform(im_flip)
#im = im_flip

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

def RotMat(axis, ang):
    s = np.sin(ang)
    c = np.cos(ang)
    res = np.zeros((3, 3))
    if axis == 'Z':
        res[0, 0] = c
        res[0, 1] = -s
        res[1, 0] = s
        res[1, 1] = c
        res[2, 2] = 1
    elif axis == 'Y':
        res[0, 0] = c
        res[0, 2] = s
        res[1, 1] = 1
        res[2, 0] = -s
        res[2, 2] = c
    elif axis == 'X':
        res[0, 0] = 1
        res[1, 1] = c
        res[1, 2] = -s
        res[2, 1] = s
        res[2, 2] = c
    return res

def angle2dcm(azi, ele, theta):
    azi = radians(azi)
    ele = radians(ele)
    theta = radians(theta)
    PI = np.arccos(-1)
    return np.dot(RotMat('Z', theta), np.dot(RotMat('X', - (PI / 2 - ele)), RotMat('Z', - azi)))

#vp_pred[:, 1] = vp_pred[:, 1] - 180.
#vp_pred[:, 2] = vp_pred[:, 2] - 180.
#vp_pred = vp_pred * np.pi / 180.

    # render viewpoint predictions
[azi, ele, inp] = vp_pred[0].detach().cpu().numpy()

mat = angle2dcm(azi, ele - 90, inp - 180)
mat = torch.FloatTensor(mat)
front = torch.FloatTensor([0,0,1])
right = torch.FloatTensor([0,1,0])

up = torch.FloatTensor([1,0,0])
print(mat)
print("....")
print("Front", torch.matmul(front, mat))
print("Up", torch.matmul(up, mat))
print("Right", torch.matmul(right, mat))

render_obj(obj_path,
           "out",
           azi, ele - 90, inp - 180,
           "test.png",
           [521, 512], None, None)

#render_obj(obj_path,
#            "out",
#            azi, ele, inp,
#            "test.png",
#            [521, 512], None, None)

# blend viewpoint results into the original image
#img = cv2.imread(img_name)
#w, h = img.shape[1], img.shape[0]
#bg = Image.new("RGBA", (w, h))
#vp_rend = Image.open(os.path.join(out_dir, save_name))
#crop_resize_paste(vp_rend, bg, left, upper, right, lower)

#draw = ImageDraw.Draw(bg)
#draw.rectangle([(left, upper), (right, lower)], outline=(0, 255, 0, 255),
#               width=int(0.02 * min(img.shape[0], img.shape[1])))
#print("bgsave")
#bg.save(os.path.join(out_dir, save_name))

