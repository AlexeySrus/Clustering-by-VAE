from keras.preprocessing.sequence import Sequence
import cv2
import numpy as np


def resize_coeff(x, new_x):
    """
    Evaluate resize coefficient from image shape
    Args:
        x: original value
        new_x: expect value

    Returns:
        Resize coefficient
    """
    return new_x / x


def resize_image(img, resize_shape=(128, 128), interpolation=cv2.INTER_AREA):
    """
    Resize single image
    Args:
        img: input image
        resize_shape: resize shape in format (height, width)
        interpolation: interpolation method

    Returns:
        Resize image
    """
    return cv2.resize(img, None, fx=resize_coeff(img.shape[1], resize_shape[1]),
                      fy=resize_coeff(img.shape[0], resize_shape[0]),
                      interpolation=interpolation)


class BatchDataLoader(Sequence):
    def __init__(self, x_set, y_set, batch_size, drop_last=True):
        self.x, self.y = list(x_set), list(y_set)
        self.is_train = True
        assert (len(self.x) == len(self.y)) or \
               (drop_last and (abs(len(self.x) - len(self.y)) < batch_size))
        self.batch_size = batch_size

        if not drop_last:
            self._pad()

    def _pad(self):
        for i in range(len(self.x) - (len(self.x) // self.batch_size)):
            self.x.append(self.x[-1])
            self.y.append(self.y[-1])

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        if not self.is_train:
            return np.array(batch_x)
        return np.array(batch_x), np.array(batch_y)

    def eval(self):
        self.is_train = False

    def train(self):
        self.is_train = True


class BatchDataLoaderByImagesPaths(Sequence):
    def __init__(self, x_set, y_set, shape, batch_size, drop_last=True):
        """
        Class constructor
        Args:
            x_set: images paths list
            y_set: y_true list
            batch_size: batch size
            drop_last: drop last batch
        """
        assert batch_size > 0

        self.x, self.y = list(x_set), list(y_set)
        self.is_train = True

        assert (len(self.x) == len(self.y)) or \
               (drop_last and (abs(len(self.x) - len(self.y)) < batch_size))

        self.batch_size = batch_size
        self.shape = shape

        if not drop_last:
            self._pad()

    def _pad(self):
        for i in range(len(self.x) - (len(self.x) // self.batch_size)):
            self.x.append(self.x[-1])
            self.y.append(self.y[-1])

    def __len__(self):
        return len(self.x) // self.batch_size

    def __getitem__(self, idx):
        batch_x = [
            resize_image(
                cv2.imread(p, 0), self.shape
            ).reshape(self.shape) / 255.
            for p in self.x[idx * self.batch_size:(idx + 1) * self.batch_size]
        ]

        if not self.is_train:
            return np.array(batch_x)
        return np.array(batch_x), np.array(batch_x)

    def eval(self):
        self.is_train = False

    def train(self):
        self.is_train = True
