import cv2, os, glob, sys, tqdm
import matplotlib.pyplot as plt
import numpy as np
sys.path.insert(0, os.path.abspath('/home/nic/cz4041'))

from utils import get_hsv_masked

def visualize(paths):
    size = 224
    n_cols = 8; n_rows = 8
    canvas = np.zeros((n_rows * size, n_cols * size, 3))
    counter = 0
    for row in range(n_rows):
        for col in range(0, n_cols, 2):
            image = cv2.resize(cv2.imread(paths[counter])[:, :, ::-1], (size, size))
            # fill canvas up row-wise in pairs; image then hue-masked image
            canvas[
                row * size: (row + 1) * size,
                col * size: (col + 1) * size, :
            ] = image
            col += 1
            canvas[
                row * size: (row + 1) * size,
                col * size: (col + 1) * size, :
            ] = get_hsv_masked(image)
            counter += 1

    canvas = np.uint8(canvas)
    title = f'Hue-masked for {os.path.basename(os.path.dirname(paths[0]))}'
    plt.figure(figsize=(9, 9), dpi=200)
    plt.imshow(canvas, aspect='auto')
    plt.axis('off')
    plt.title(title)
    plt.savefig(os.path.join(outdir, title.replace(' ', '_') + '.jpg'))

root = '/data/train'
outdir = 'results/hsv_mask'
os.makedirs(outdir, exist_ok=True)
for subroot in tqdm.tqdm(sorted(glob.glob(f'{root}/*'))):
    paths = glob.glob(f'{subroot}/*')
    visualize(paths[:32])
