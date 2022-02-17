import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow.keras.backend as K

def compute_heatmap(model, image, colormap=cv2.COLORMAP_JET):
    last_conv = [l for l in model.layers if isinstance(l, layers.Conv2D)][-1]

    # log variables for automatic differentiation later
    with tf.GradientTape() as tape:
        conv_feature_maps, preds = tf.keras.Model(model.input, [last_conv.output, model.output])(image)
        preds = tf.nn.softmax(preds) # final dense layer did not have softmax
        loss = tf.reduce_max(preds, axis=-1) # take -ve max prediction probability as loss

    # compute useful gradients (which features result in increase in prediction probability)
    grads = tape.gradient(loss, conv_feature_maps) # dloss wrt dconv_feature_maps
    feature_mask = tf.cast(conv_feature_maps > 0, 'float32') # useful features
    grads_mask = tf.cast(grads > 0, 'float32') # increase in loss (probability)
    masked_grads = grads * grads_mask * feature_mask

    # discard batch dimension
    masked_grads = masked_grads[0]
    conv_feature_maps = conv_feature_maps[0]

    weights = tf.reduce_mean(masked_grads, axis=(0, 1)) # GlobalAveragePooling2D
    # weight each channel in feature_map then mean to get 1 feature_map
    feature_map = tf.reduce_sum(weights * conv_feature_maps, axis=-1)
    h, w = image.shape[1: 3]
    heatmap = cv2.resize(feature_map.numpy(), (w, h))
    
    # make heatmap range from 0 to 255
    numer = heatmap - np.min(heatmap)
    denom = (np.max(heatmap) - np.min(heatmap)) + K.epsilon()
    heatmap = numer / denom
    heatmap = (heatmap * 255).astype("uint8")
    # generate bgr from black and white (cv2) then convert to rgb
    heatmap = cv2.applyColorMap(heatmap, colormap).squeeze()[..., ::-1] # generate rgb from black and white
    return heatmap

def overlay_heatmap(image, heatmap, undo_preprocess, alpha=.2):
    overlay = undo_preprocess(image).astype('float32') * (1 - alpha) + heatmap.astype('float32') * alpha
    overlay = np.uint8(overlay)
    return overlay

def save_gradcam(model, images, outdir, undo_preprocess):
    assert images.shape[0] == 8
    os.makedirs(outdir, exist_ok=True)
    n_rows = 4; n_cols = 6; size = 224
    canvas = np.zeros((n_rows * size, n_cols * size, 3))
    counter = 0
    for row in range(n_rows):
        for col in range(0, n_cols, 3):
            image = np.expand_dims(images[counter], axis=0)
            heatmap = compute_heatmap(model, image)
            overlay = overlay_heatmap(image, heatmap, undo_preprocess)
            # fill canvas up row-wise
            canvas[
                row * size: (row + 1) * size,
                col * size: (col + 1) * size
            ] = undo_preprocess(image)
            col += 1
            canvas[
                row * size: (row + 1) * size,
                col * size: (col + 1) * size
            ] = heatmap
            col += 1
            canvas[
                row * size: (row + 1) * size,
                col * size: (col + 1) * size
            ] = overlay
            counter += 1
    plt.figure(figsize = (9, 9), dpi=200)
    canvas = canvas.astype(np.uint8)
    plt.imshow(canvas)
    plt.axis('off')
    plt.savefig(os.path.join(outdir, f'gradcam.jpg'))
