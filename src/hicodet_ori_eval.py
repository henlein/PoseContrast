import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import json
import torch
import numpy as np
from PIL import Image
from dataset.data_utils import angles_to_matrix
from tqdm import tqdm
from dataset.data_utils import resize_pad
import torchvision.transforms as transforms
from model.resnet import resnet50
from model.vp_estimator import BaselineEstimator


class HicoDetDataset:
    def __init__(self, img_folder, anno_file):
        self.img_folder = img_folder
        self.anno_file = anno_file

        self.anno_list = []
        with open(self.anno_file, 'r') as f:
            anno_ori = json.load(f)

            for anno_idx, (anno_key, anno) in enumerate(anno_ori["_via_img_metadata"].items()):
                for region in anno["regions"]:
                    bbox = [region["shape_attributes"]["x"], region["shape_attributes"]["y"],
                            region["shape_attributes"]["x"] + region["shape_attributes"]["width"],
                            region["shape_attributes"]["y"] + region["shape_attributes"]["height"]]

                    front_vec = self._ori_dict_to_vec(region["region_attributes"]["front"])
                    up_vec = self._ori_dict_to_vec(region["region_attributes"]["up"])

                    if "category" not in region["region_attributes"]:
                        continue
                    elif region["region_attributes"]["category"] == "human":
                        name = "person"
                    elif region["region_attributes"]["category"] == "object":
                        name = region["region_attributes"]["obj name"]
                    else:
                        print("???????????????????")
                        exit()

                    self.anno_list.append({"bbox": bbox, "name": name, "front": front_vec, "up": up_vec, "img": anno["filename"]})

    def __len__(self):
        return len(self.anno_list)

    def __getitem__(self, item):
        anno = self.anno_list[item]

        img = Image.open(self.img_folder + anno["img"]).convert('RGB')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        im_transform = transforms.Compose([transforms.ToTensor(), normalize])

        bbox = anno["bbox"]
        pil_img = img.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        #img.show()
        img = resize_pad(pil_img, 224)
        img = im_transform(img)
        return img, anno, pil_img

    def _ori_dict_to_vec(self, ori_dict):
        vector = np.zeros(3)
        keys = ori_dict.keys()
        if "n/a" in keys or len(keys) == 0:
            return vector
        elif "+x" in keys:
            vector[0] = 1
        elif "-x" in keys:
            vector[0] = -1
        elif "+y" in keys:
            vector[2] = 1
        elif "-y" in keys:
            vector[2] = -1
        elif "+z" in keys:
            vector[1] = 1
        elif "-z" in keys:
            vector[1] = -1
        else:
            print(ori_dict)
            print("!!!!!!!!!!!!!!!!!")
        return vector


class OriModel:
    def __init__(self, pretrained):
        self.net_feat = resnet50(num_classes=128)
        self.net_vp = BaselineEstimator(img_feature_dim=2048)

        self.net_feat.cuda()
        self.net_vp.cuda()

        state = torch.load(pretrained)
        self.net_feat.load_state_dict(state['net_feat'])
        self.net_vp.load_state_dict(state['net_vp'])
        self.net_feat.eval()
        self.net_vp.eval()

    def process_img(self, img):
        with torch.no_grad():
            feat, _ = self.net_feat(img)
            out = self.net_vp(feat)
            vp_pred = self.net_vp.compute_vp_pred(out)

        vp_pred[:, 1] = vp_pred[:, 1] - 90.
        vp_pred[:, 2] = vp_pred[:, 2] - 180.

        # change degrees to radians
        vp_pred_rad = vp_pred * np.pi / 180.
        return vp_pred_rad, vp_pred


    def results_to_rot_vect(self, vp_pred):
        r_pred = angles_to_matrix(vp_pred)
        r_pred = r_pred[0].view(-1, 3)
        r_pred = r_pred.cpu().numpy()

        left = np.array([1., 0., 0.])
        front = np.array([0., 1., 0.])
        up = np.array([0., 0., 1.])

        left = r_pred.dot(left)
        front = r_pred.dot(front)
        up = r_pred.dot(up)

        #left, front = front, left
        return up, front, left


if __name__ == "__main__":
    # Z: geht nach oben, x nach rechts und z nach vorne ....
    img_folder = "D:/Corpora/HICO-DET/hico_20160224_det/images/merge2015/"
    #ori_annotation_file = "D:/Corpora/HICO-DET/via234_1000 items_jan 8.json"
    ori_annotation_file = "D:/Corpora/HICO-DET/via234_1200 items_train verified.json"
    #model_path = "../exps/PoseContrast_ObjectNet3d_ZeroShot/ckpt.pth"
    model_path = "../exps/PoseContrast_ObjectNet3d_FewShot/ckpt.pth"
    dataset = HicoDetDataset(img_folder, ori_annotation_file)
    model = OriModel(model_path)

    correct_up_per_obj = {}
    all_up_per_obj = {}

    correct_front_per_obj = {}
    all_front_per_obj = {}
    for (img, annotation, pil_img) in tqdm(dataset):
        gold_front = annotation["front"]
        gold_up = annotation["up"]
        obj_label = annotation["name"]

        if obj_label not in correct_up_per_obj:
            correct_up_per_obj[obj_label] = 0
            all_up_per_obj[obj_label] = 0
            correct_front_per_obj[obj_label] = 0
            all_front_per_obj[obj_label] = 0

        img = img[None, :].cuda()
        results_rad, results_deg = model.process_img(img)
        up, front, left = model.results_to_rot_vect(results_rad)
        #print(up)

        #print(left)
        if np.count_nonzero(gold_front) > 0:
            all_front_per_obj[obj_label] += 1
            front_abs = np.abs(front)
            front_amax = np.argmax(front_abs)
            if gold_front[front_amax].item() != 0:
                if front[front_amax] > 0 and gold_front[front_amax] > 0:
                    correct_front_per_obj[obj_label] += 1
                elif front[front_amax] < 0 and gold_front[front_amax] < 0:
                    correct_front_per_obj[obj_label] += 1


        if np.count_nonzero(gold_up) > 0:
            all_up_per_obj[obj_label] += 1
            up_abs = np.abs(up)
            up_amax = np.argmax(up_abs)
            if gold_up[up_amax].item() != 0:
                if up[up_amax] > 0 and gold_up[up_amax] > 0:
                    correct_up_per_obj[obj_label] += 1
                elif up[up_amax] < 0 and gold_up[up_amax] < 0:
                    correct_up_per_obj[obj_label] += 1


    for key in all_up_per_obj.keys():
        if all_up_per_obj[key] < 10:
            continue
        print("....")
        print(key)
        if all_up_per_obj[key] > 0:
            print("up", correct_up_per_obj[key] / all_up_per_obj[key])
        if all_front_per_obj[key] > 0:
            print("front", correct_front_per_obj[key] / all_front_per_obj[key])