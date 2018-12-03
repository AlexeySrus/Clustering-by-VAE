import argparse
import logging
import os
import numpy as np
from random import shuffle
from keras.datasets import mnist
from keras.callbacks import ModelCheckpoint

from libs.model.vae import create_conv_vae, load_last_weights
from libs.utils.dataset_processing import BatchDataLoaderByImagesPaths
from nist19.dataset_genrator import generate_paths, get_one_hot_label


def parse_arguments():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data-path', required=True, type=str,
                            help='Path to NIST19 dataset.')
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
    arg_parser.add_argument('--shuffle', action='store_true', required=False,
                            default=False,
                            help='Shuffle dataset.')
    arg_parser.add_argument('--prepare-dataset-list', required=False, type=str,
                            help='Path to prepare dataset list.')
    arg_parser.add_argument('--validation-split', required=False, type=float,
                            help='Validation split rate in 0..1 range.')
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
        input_shape=(28, 28, 1),
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
                    img_width=28
                )
            ),
            save_weights_only=True
        ))

    val_data_gen = None
    val_len = None

    origin_dataset = generate_paths(app_args.data_path)

    if app_args.shuffle:
        shuffle(origin_dataset)

    dataset_paths = [
        [d[0], d[1]]
        for d in origin_dataset
    ]

    if app_args.validation_split is not None:
        x_train = np.array(dataset_paths)[
                  :int(len(dataset_paths)*(1 - app_args.validation_split)), 1]
        x_val = np.array(dataset_paths)[
                -int(len(dataset_paths)*app_args.validation_split):, 1]

        val_data_gen = BatchDataLoaderByImagesPaths(
            x_val,
            x_val,
            (28, 28, 1),
            app_args.batch_size
        )

        val_len = len(val_data_gen)
    else:
        x_train = np.array(dataset_paths)[:, 1]

    train_data_gen = BatchDataLoaderByImagesPaths(
        x_train,
        x_train,
        (28, 28, 1),
        app_args.batch_size
    )

    vae.fit_generator(
        train_data_gen,
        len(train_data_gen),
        callbacks=callbacks,
        epochs=app_args.epochs, initial_epoch=start_epoch,
        workers=4,
        use_multiprocessing=True,
        max_queue_size=10,
        validation_data=val_data_gen,
        validation_steps=val_len
    )
