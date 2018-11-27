import argparse
import logging
import os
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint

from libs.model.vae import create_conv_vae, load_last_weights
from libs.utils.dataset_processing import BatchDataLoader


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--batch-size', default=32, type=int,
                            help='Size of batch of images.')
    arg_parser.add_argument('--epochs', default=5, type=int,
                            help='Number of epochs.')
    arg_parser.add_argument('--checkpoints', help='Path to save model weights.')
    arg_parser.add_argument('--loglevel', required=False, default='info',
                            choices=['info', 'debug', 'error'], type=str,
                            help=
                            'Choice logging level. Can be: info,debug,error.')
    arg_parser.add_argument('--latentdim', required=False,
                            type=int,
                            default=2)
    return arg_parser.parse_args()


def set_logger(loglevel='info'):
    if loglevel == 'info':
        logging.basicConfig(level=logging.INFO)

    if loglevel == 'debug':
        logging.basicConfig(level=logging.DEBUG)

    if loglevel == 'error':
        logging.basicConfig(level=logging.ERROR)


if __name__ == '__main__':
    app_args = parse_arguments()

    set_logger(app_args.loglevel)

    models, loss = create_conv_vae(
        input_shape=(32, 32, 3),
        latent_dim=app_args.latentdim,
        dropout_rate=0.4,
        batch_size=app_args.batch_size
    )

    vae = models["vae"]

    start_epoch = load_last_weights(vae, app_args.checkpoints)
    logging.info('START EPOCH NUMBER: {}'.format(start_epoch))

    logging.info(vae.summary())

    callbacks = []

    if app_args.checkpoints:
        callbacks.append(ModelCheckpoint(
            os.path.join(
                app_args.checkpoints,
                'vae-{epoch}-{latent}-{batch}-{img_width}.h5'.format(
                    epoch='{epoch}',
                    latent=app_args.latentdim,
                    batch=app_args.batch_size,
                    img_width=32
                )
            ),
            save_weights_only=True
        ))

    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.

    print(x_train.shape)
    x_train = np.reshape(x_train, (len(x_train), 32, 32, 3))
    x_test = np.reshape(x_test, (len(x_test), 32, 32, 3))

    train_data_gen = BatchDataLoader(
        x_train,
        x_train,
        app_args.batch_size
    )

    val_data_gen = BatchDataLoader(
        x_test,
        x_test,
        app_args.batch_size
    )

    vae.fit_generator(
        train_data_gen,
        len(train_data_gen),
        callbacks=callbacks,
        epochs=app_args.epochs, initial_epoch=start_epoch,
        validation_data=val_data_gen,
        validation_steps=len(val_data_gen),
        workers=4,
        use_multiprocessing=True
    )
