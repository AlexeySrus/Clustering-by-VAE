from keras.preprocessing.sequence import Sequence
import numpy as np


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
