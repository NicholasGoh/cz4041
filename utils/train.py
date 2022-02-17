import os, cv2, sys, pickle
import numpy as np
import matplotlib.pyplot as plt
from IPython import display
import tensorflow_addons as tfa
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
import tensorflow.keras.backend as K
from tensorflow.keras import layers
import tensorflow as tf
sys.path.insert(0, os.path.abspath('/home/nic/cz4041'))
plt.style.use('seaborn-whitegrid')

from utils import get_hsv_masked

class DatasetWrapper:
    def __init__(self, dataset_path, img_size, batch_size):
        self.dataset_path = dataset_path
        self.categories = sorted(os.listdir(dataset_path)) 
        self.num_classes = len(self.categories)
        self.img_size = img_size
        self.input_shape = (img_size, img_size, 3)
        self.batch_size = batch_size

    def visualize_train_dataset(self):

        _, axes = plt.subplots(len(self.categories), 6, figsize=(30, 30))
        for i, category in enumerate(self.categories):
            path = os.path.join(self.dataset_path, category)
            images = os.listdir(path)
            for j in range(6):
                image = cv2.imread(path + '/' + images[j])[..., ::-1]
                axes[i, j].imshow(image)
                axes[i, j].set(xticks=[], yticks=[])
                axes[i, j].set_title(category, color = 'tomato').set_size(16)
        plt.tight_layout()

    def visualize_valid_dataset(self):
        _, axes = plt.subplots(len(self.categories), 6, figsize = (30, 30))
        for i in range(len(self.categories)):
            for j in range(6):
                index = (i * 6 + j) % self.batch_size
                if index % self.batch_size == 0:
                    for x, _ in self.valid_generator:
                        break
                image = self.undo_preprocess(x[index])
                axes[i, j].imshow(image)
                axes[i, j].set(xticks=[], yticks=[])
        plt.tight_layout()

    def make_datagens(self, augment=False, hsv_mask=False):
        func = lambda x: (preprocess_input(x) if not hsv_mask
                                              else preprocess_input(get_hsv_masked(x)))
        datagen_kwargs = dict(
            validation_split=.3,
            preprocessing_function=func)
        if augment:
            augment_kwargs = dict(rotation_range=90,
                width_shift_range=0.2,
                height_shift_range=0.2,
                horizontal_flip=True,
                vertical_flip=True)
            datagen_kwargs.update(augment_kwargs)
        datagen = ImageDataGenerator(**datagen_kwargs)

        self.train_generator = datagen.flow_from_directory(
                self.dataset_path,
                target_size=(self.img_size, self.img_size),
                color_mode='rgb',
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                subset='training',
                class_mode="categorical")
        self.valid_generator = datagen.flow_from_directory(
                self.dataset_path,
                target_size=(self.img_size, self.img_size),
                color_mode='rgb',
                batch_size=self.batch_size,
                shuffle=True,
                seed=42,
                subset='validation',
                class_mode="categorical")

        # 1.5 times the (augmented) data
        multiplier = 1.5 if augment else 1
        self.step_size_train= int(self.train_generator.n // self.train_generator.batch_size * multiplier)
        self.step_size_valid= int(self.valid_generator.n // self.valid_generator.batch_size * multiplier)

    def infinite_generator(self, generator):
        while True:
            yield next(generator)

    def undo_preprocess(self, image):
        '''undoes mobilenet preprocess_input'''
        return np.uint8((image.squeeze() + 1) * 127.5)

class ModelWrapper:
    def __init__(self, dataset):
        self.dataset = dataset
        self.plotdir = 'results/plots'
        self.weightdir = 'results/weights'
        self.historydir = 'results/pickles'
        os.makedirs(self.plotdir, exist_ok=True)
        os.makedirs(self.weightdir, exist_ok=True)
        os.makedirs(self.historydir, exist_ok=True)

    def _get_base(self, weights):
        K.clear_session() # frees up memory
        mobilenet = MobileNetV2(include_top=True,
                                weights=weights,
                                input_shape=self.dataset.input_shape)
        mobilenet.trainable = False if weights == 'imagenet' else True
        return mobilenet, mobilenet.layers[-2].output # GlobalAveragePooling2D layer

    def make_model(self,
                   figname='baseline',
                   weights='imagenet',
                   hidden_layers=0,
                   dropout=0,
                   regularizers=dict(kernel_regularizer=None,
                                    bias_regularizer=None,
                                    activity_regularizer=None)):
        self.figname = figname
        self.weights = weights
                
        base_model, pooling = self._get_base(weights)
        x = layers.Dropout(dropout)(pooling)
        
        for i in reversed(range(1, hidden_layers + 1)):
            dense_kwargs = dict(units=i * 128, activation='relu')
            dense_kwargs.update(regularizers)
            
            x = layers.Dense(**dense_kwargs)(x)
            x = layers.Dropout(dropout)(x)
        
        dense_kwargs = dict(units=self.dataset.num_classes)
        dense_kwargs.update(regularizers)
        x = layers.Dense(**dense_kwargs)(x)
        self.model = tf.keras.Model(base_model.input, x)

    # todo
    # import tensorflow_addons as tfa
    def compile(self, lr, cyclic=False):
        assert not cyclic, 'no tfa'
        if cyclic: # lr oscillates between 2 values and decays at the same time
    #         clr = tfa.optimizers.CyclicalLearningRate(initial_learning_rate=lr / 10,
    #         maximal_learning_rate=lr,
    #         scale_fn=lambda x: 1/(2.**(x-1)),
    #         step_size=2 * step_size_train
    #         )
    #         optimizer = tf.keras.optimizers.Adam(clr)
            pass
        else:
            optimizer = tf.keras.optimizers.Adam(lr)
            self.clr = None

        self.model.compile(
            optimizer=optimizer,
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=[tfa.metrics.F1Score(
                num_classes=self.dataset.num_classes,
                average='micro')])

    def visualize_cyclic(self, clr, epochs):
        step = np.arange(0, epochs * self.dataset.step_size_train)
        lr = clr(step)
        plt.plot(step, lr)
        plt.xlabel('Steps')
        plt.ylabel('Learning Rate')
        plt.title('Cyclic Learning Rate with Decay')
        plt.show()
        
    def fit(self, epochs):
        history = self.model.fit(
            x=self.dataset.infinite_generator(self.dataset.train_generator),
            validation_data=self.dataset.infinite_generator(self.dataset.valid_generator),
            steps_per_epoch=self.dataset.step_size_train,
            validation_steps=self.dataset.step_size_valid,
            callbacks=[PlotLosses(self.plotdir, self.figname)],
            epochs=epochs)
        self.history = history.history
        self.model.save_weights(os.path.join(self.weightdir, self.figname + '.h5'))
        self._save_history(history.history)

    def _save_history(self, history):
        with open(os.path.join(self.historydir, self.figname + '.pkl'), 'wb') as f:
            pickle.dump(history, f)

class PlotLosses(tf.keras.callbacks.Callback):
    def __init__(self, plotdir, figname):
        self.plotdir = plotdir
        self.figname = figname
    
    def on_train_begin(self, logs=None):
        self.keys = ['loss', 'val_loss', 'f1_score', 'val_f1_score']
        self.history = {k: [] for k in self.keys}
    
    def on_epoch_end(self, epoch, logs=None):
        # update history of each metric for this epoch
        for key in self.keys:
            self.history.get(key).append(logs.get(key))
        display.clear_output(wait=True)
        
        f, axes = plt.subplots(1, 2, figsize=(12, 4))
        x = np.arange(len((self.history.get('loss'))))
        for i, key in enumerate(self.history):
            axes[i // 2].plot(x, self.history.get(key), label=key)
            axes[i // 2].legend()
        axes[0].set_xlabel('Epochs')
        axes[0].set_ylabel('Loss')
        axes[1].set_xlabel('Epochs')
        axes[1].set_ylabel('Micro F1 Score')
        f.suptitle(self.figname)
        plt.show()
        f.savefig(os.path.join(self.plotdir, self.figname + '.jpg'))
