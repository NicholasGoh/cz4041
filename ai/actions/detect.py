import os, shlex, time
import subprocess
import cv2
import gc
import copy
import torch
import argparse
import numpy as np
from tqdm import tqdm

from loaders import PictureDataLoader

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor


def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_paths', type=str, default='/data/plant-seedlings-classification/train/Charlock')
    parser.add_argument('-o', '--outdir', type=str, default='/data/cropped')
    args = parser.parse_args()
    return args

class Detector:
    def __init__(self):
        self.args = make_args()
        self.images = PictureDataLoader(self.args.image_paths, shuffle=False)
        os.makedirs(self.args.outdir, exist_ok=True)
        self.actions = {
            'detection': 'COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml',
            'segmentation': 'COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml'
        }
            #  'segmentation': 'Cityscapes/mask_rcnn_R_50_FPN.yaml'
            #  'segmentation': 'LVISv0.5-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_1x.yaml'

        # detectron2 inits
        cfg = get_cfg()
        action = self.actions.get('segmentation')
        cfg.merge_from_file(model_zoo.get_config_file(action))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(action)
        self.predictor = DefaultPredictor(cfg)

    def lossy_image(self, segment=False):
        '''wrapper function to process image into selected people'''
        for image_path, image in tqdm(self.images):
            self.image_path = image_path
            if image is None:
                os.remove(image_path)
                continue
            out = self._lossy_image(image, segment)
            self.imwrite(image_path, out)
            #  self.imwrite(
            #      image_path,
            #      self._age(self._gen(self._lossy_image(image, segment))))
            #  os.remove(image_path)

    def _lossy_image(self, image, segment=False):
        '''detects or segments image into multiple people'''
        outputs = self.predictor(image)
        if segment:
            people = self._segment(image, outputs)
        else:
            people = self._detect(image, outputs, outputs['instances'].pred_classes)
        return people

    def _detect(self, image, outputs, classes):
        boxes = outputs['instances'].pred_boxes.tensor.to(torch.device('cpu')).tolist()
        people = []
        unique_boxes = [[int(b) for b in boxes[i]] for i, c in enumerate(classes)
                                                   if c == 0] # c == 0 is person class
        for box in unique_boxes:
            xmin, ymin, xmax, ymax = box
            people.append(image[ymin: ymax + 1, xmin: xmax + 1])
        return people

    def imwrite(self, path, people):
        if not people:
            self.log(f'{self.image_path}: nothing', fail=True)
        base = os.path.basename(path)
        parent, ext = os.path.splitext(base)

        for counter, person in enumerate(people):
            outdir = self.args.outdir
            os.makedirs(outdir, exist_ok=True)
            des = os.path.join(outdir, f'{parent}_{str(counter).zfill(2)}{ext}')
            cv2.imwrite(des, person)

    def _segment(self, image, outputs):
        try:
        # todo
            mask = outputs['instances'].pred_masks.to(torch.device('cpu')).numpy()
            print(self.image_path, mask.shape)
            people = []
            for i in range(mask.shape[0]):
                image = copy.deepcopy(image)
                image[mask[i] == False] = 0
                people.append(image)
        except Exception as e:
            print(e)
            return []
        return people

    def log(self, string, stdout=False, fail=False):
        file = '/data/fail.txt' if fail else '/data/logs.txt'
        with open(file, 'a') as f:
            f.write(f'{string}\n')
        if stdout:
            print(f'{string}\n')

det = Detector()
det.lossy_image(segment=True)
