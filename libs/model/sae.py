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
    x = Dense(output_shape[0] * output_shape[1] * output_shape[2], activation='sigmoid')(input_encoded)
    decoded = Reshape((output_shape[0], output_shape[1], output_shape[2]))(x)

    models = {}
    models["encoder"] = Model(input_img, encoded, name="encoder")
    models["decoder"] = Model(input_encoded, decoded, name="decoder")
    models["ae"] = Model(input_img, models["decoder"](
        models["encoder"](input_img)), name="ae")


    models["ae"].compile(optimizer=Adam(lr=start_lr), loss=loss)
    return models


def step_vae(input_shape, output_shape, latent_dim,
                start_lr=0.001, loss="binary_crossentropy",
             dropout_rate=0.4, batch_size=32):
    models = {}

    def apply_bn_and_dropout(x):
        return Dropout(dropout_rate)(BatchNormalization()(x))

    input_img = Input(shape=input_shape)

    x = Flatten()(input_img)

    z_mean = Dense(latent_dim)(x)
    z_log_var = Dense(latent_dim)(x)

    def sampling(args):
        z_mean, z_log_var = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim), mean=0.,
                                  stddev=1.0)
        return z_mean + K.exp(z_log_var / 2) * epsilon

    l = Lambda(sampling, output_shape=(latent_dim,))([z_mean, z_log_var])

    z = Input(shape=(latent_dim,))

    x = Dense(128)(z)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(256)(x)
    x = LeakyReLU()(x)
    x = apply_bn_and_dropout(x)
    x = Dense(output_shape[0] * output_shape[1] * output_shape[2], activation='sigmoid')(x)
    decoded = Reshape((output_shape[0], output_shape[1], output_shape[2]))(x)

    models["encoder"] = Model(input_img, l, 'Encoder')
    models["z_meaner"] = Model(input_img, z_mean, 'Enc_z_mean')
    models["z_lvarer"] = Model(input_img, z_log_var, 'Enc_z_log_var')

    def vae_loss(x, decoded):
        x = K.reshape(x, shape=(batch_size, output_shape[0] * output_shape[1]*output_shape[2]))
        decoded = K.reshape(decoded, shape=(batch_size, output_shape[0] * output_shape[1]*output_shape[2]))
        xent_loss = output_shape[0] * output_shape[1]*output_shape[2]*binary_crossentropy(x, decoded)
        kl_loss = -0.5 * K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return (xent_loss + kl_loss)/2/output_shape[0]/output_shape[1]/output_shape[2]

    models["decoder"] = Model(z, decoded, name='Decoder')
    models["ae"] = Model(input_img,
                          models["decoder"](models["encoder"](input_img)),
                          name="VAE")

    models["ae"].compile(optimizer=Adam(lr=start_lr), loss=vae_loss)

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
