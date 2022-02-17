import cv2, os
import numpy as np
import matplotlib.pyplot as plt

outdir = 'results/hsv_range'
os.makedirs(outdir, exist_ok=True)
lower_b = np.array([30,20,20])
upper_b = np.array([90,255,255])

s_gradient = np.ones((500,1), dtype=np.uint8)*np.linspace(lower_b[1], upper_b[1], 500, dtype=np.uint8)
v_gradient = np.rot90(np.ones((500,1), dtype=np.uint8)*np.linspace(lower_b[1], upper_b[1], 500, dtype=np.uint8))
h_array = np.arange(lower_b[0], upper_b[0]+1, 5)
canvas = np.zeros((500, len(h_array) * 500, 3))

for i, hue in enumerate(h_array):
    h = hue*np.ones((500,500), dtype=np.uint8)
    hsv_color = cv2.merge((h, s_gradient, v_gradient))
    rgb_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2RGB)
    canvas[:, 500*i: 500*(i+1), :] = rgb_color
canvas = canvas.astype(np.uint8)
plt.figure(figsize=(9, 9), dpi=200)
plt.imshow(canvas)
plt.axis('off')
plt.savefig(os.path.join(outdir, 'hsv_range.jpg'))
