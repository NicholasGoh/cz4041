import glob
import cv2
import random
import shutil
import argparse

import sys, os
sys.path.insert(0, os.path.abspath('..'))
from utils import read_file, write_line
sys.path.remove(os.path.abspath('..'))

class PictureDataLoader:
    def __init__(self, path, shuffle = True):
        self.image_paths = glob.glob(os.path.join(path, '*.png'))
        assert len(self.image_paths) > 0, f'No images found in {path}'
        if shuffle:
            random.shuffle(self.image_paths)
        else:
            self.image_paths = sorted(self.image_paths)
    def __len__(self):
        return len(self.image_paths)
    def __getitem__(self, index):
        path = self.image_paths[index]
        return path, cv2.imread(path)
    def find(self, path):
        assert os.path.isfile(path)
        return cv2.imread(path)

def make_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--image_paths', type=str, default='in')
    parser.add_argument('-r', '--results', type=str, default='out')
    args = parser.parse_args()
    return args
