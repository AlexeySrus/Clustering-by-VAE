import argparse
import logging
import os
import numpy as np
from keras.datasets import cifar10
from keras.callbacks import ModelCheckpoint
from libs.model.vae import create_conv_vae, load_last_weights
from libs.model.sae import step_ae, step_vae
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
    arg_parser.add_argument('--last-latent-path', required=False, type=str,
                            help='Path to save last latent data dimension.')
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

    sae = [step_ae(
        input_shape=(32, 32, 3),
        output_shape=(32, 32, 3),
        latent_dim=app_args.latentdim,
        loss='mse'
    )]

    deep = 4

    for i in range(deep):
        sae.append(step_ae(
            input_shape=(app_args.latentdim // (2 ** i), 1),
            output_shape=(32, 32, 3),
            latent_dim=app_args.latentdim // (2 ** (i + 1)),
            loss='mse'
        ))

    sae.append(step_vae(
        input_shape=(app_args.latentdim // (2 ** deep), 1),
        output_shape=(32, 32, 3),
        latent_dim=app_args.latentdim // (2 ** (deep + 1)),
        batch_size=app_args.batch_size
    ))

    start_epoch = 0
    logging.info('START EPOCH NUMBER: {}'.format(start_epoch))

    for ae in sae:
        logging.info(ae["ae"].summary())

    callbacks = []

    if not os.path.isdir(app_args.checkpoints):
        os.makedirs(app_args.checkpoints)

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

    if app_args.last_latent_path is not None:
        if not os.path.isdir(app_args.last_latent_path):
            os.makedirs(app_args.last_latent_path)

    x_train = x_train.astype('float32') / 255.
    x_test = x_test.astype('float32') / 255.
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

    for i, ae in enumerate(sae):
        if i > 0:
            train_data_gen.eval()
            val_data_gen.eval()

            d = sae[i - 1]["encoder"].predict_generator(train_data_gen)
            train_data_gen = BatchDataLoader(
                d.reshape(
                    len(d), app_args.latentdim // (2 ** (i - 1)), 1
                ),
                x_train,
                app_args.batch_size
            )

            if i == len(sae) - 1 and app_args.last_latent_path is not None:
                np.save(
                    app_args.last_latent_path + 'latent_from_{}.npy'.format(
                        app_args.latentdim // (2 ** (i - 1))
                    ),
                    d
                )

            d = sae[i - 1]["encoder"].predict_generator(val_data_gen)
            val_data_gen = BatchDataLoader(
                d.reshape(
                    len(d), app_args.latentdim // (2 ** (i - 1)), 1
                ),
                x_test,
                app_args.batch_size
            )

        print('Train generator #{}:'.format(i + 1))
        ae["ae"].fit_generator(
            train_data_gen,
            len(train_data_gen),
            callbacks=callbacks if i == len(sae) - 1 else [],
            epochs=app_args.epochs, initial_epoch=start_epoch,
            validation_data=val_data_gen,
            validation_steps=len(val_data_gen),
            workers=4,
            use_multiprocessing=True
        )
