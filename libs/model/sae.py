from keras.layers import Input, Dense
from keras.layers import BatchNormalization, Dropout, Flatten, Reshape, Lambda
from keras.layers import concatenate
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.optimizers import Adam, RMSprop
from keras.layers import Conv2D, MaxPooling2D, Dense, UpSampling2D, Reshape, Flatten, BatchNormalization, Lambda
from keras.layers import LeakyReLU, Input, Dropout, Conv2DTranspose
from keras.models import Model
from keras.objectives import binary_crossentropy
from keras.layers.advanced_activations import LeakyReLU
from keras.applications.resnet50 import ResNet50
from keras import backend as K
from keras import losses
from keras.models import load_model
from keras.regularizers import L1L2
import tensorflow as tf
import os
import re


def step_ae(input_shape, output_shape, latent_dim,
                start_lr=0.001, loss="binary_crossentropy"):

    input_img = Input(shape=input_shape)

    x = Flatten()(input_img)

    encoded = Dense(latent_dim, activation='relu')(x)

    input_encoded = Input(shape=(latent_dim,))
    x = Dense(output_shape[0] * output_shape[1], activation='sigmoid')(input_encoded)
    decoded = Reshape((output_shape[0], output_shape[1], 1))(x)

    models = {}
    models["encoder"] = Model(input_img, encoded, name="encoder")
    models["decoder"] = Model(input_encoded, decoded, name="decoder")
    models["ae"] = Model(input_img, models["decoder"](
        models["encoder"](input_img)), name="ae")


    models["ae"].compile(optimizer=Adam(lr=start_lr), loss=loss)
    return models

def load_last_weights(model, path, logger=None):
    if path is None:
        return 0

    if not os.path.isdir(path):
        os.makedirs(path)

    weights_files_list = [
        matching_f.group()
        for matching_f in map(
            lambda x: re.match('vae-\d+-\d+-\d+-\d+.h5', x),
            os.listdir(path)
        ) if matching_f if not None
    ]

    if len(weights_files_list) == 0:
        return 0

    weights_files_list.sort(key=lambda x: -int(x.split('-')[1]))

    model.load_weights(os.path.join(path, weights_files_list[0]))

    if logger is not None:
        logger.debug('LOAD MODEL PATH: {}'.format(
            os.path.join(path, weights_files_list[0])
        ))
    else:
        print('LOAD MODEL PATH: {}'.format(
            os.path.join(path, weights_files_list[0])
        ))

    return int(weights_files_list[0].split('-')[1])
