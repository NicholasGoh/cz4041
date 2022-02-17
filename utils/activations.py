import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import backend as K

def postprocess_feature_map(image):
    image -= tf.reduce_mean(image)
    image /= (tf.math.reduce_std(image) + K.epsilon())
    # this distribution results in cleaner feature maps
    image *= 64
    image += 128
    # p(x) > 255 or p(x) < 0 is 0.02361
    return tf.cast(tf.clip_by_value(image, 0, 255), tf.uint8)
    
def save_activations(model, image, outdir, num_layers):
    os.makedirs(outdir, exist_ok=True)
    conv_layers = [l for l in model.layers if isinstance(l, layers.Conv2D)]
    conv_layers = [l for l in conv_layers if l.output.shape[-1] >= 32][:num_layers]
    conv_feature_maps = tf.keras.Model(model.input, [l.output for l in conv_layers])(image)
    
    n_cols = 8; n_rows = 4
    for layer, feature_maps in zip(conv_layers, conv_feature_maps):
        n_maps = feature_maps.shape[-1]
        size = feature_maps.shape[1]
        canvas = np.zeros((n_rows * size, n_cols * size))
        
        for row in range(n_rows):
            for col in range(n_cols):
                # (batch, w, h, c)
                feature_map = feature_maps[0, ..., row * n_cols + col]
                feature_map = postprocess_feature_map(feature_map)
                # fill canvas up row-wise
                canvas[
                    row * size: (row + 1) * size,
                    col * size: (col + 1) * size
                ] = feature_map
        
        scale = 2 / size # scale canvas to lower resolution
        canvas = np.uint8(canvas)
        plt.figure(figsize=(scale * canvas.shape[1], scale * canvas.shape[0]))
        plt.imshow(canvas, aspect='auto', cmap='gray')
        plt.axis('off')
        plt.title(layer.name)
        plt.savefig(os.path.join(outdir, f'{layer.name}.jpg'))
