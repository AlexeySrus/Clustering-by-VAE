import random
from threading import Thread
from queue import Queue
import tqdm
import numpy as np


class BatchDataLoader:
    def __init__(self, dataset, batch_size, queue_len=2, shuffle=True):
        self.data = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.data_batches = Queue(queue_len)
        self.ready_flag = True
        self.work_thread = None

    def _updated_indexes(self):
        indexes = list(range(len(self.data)))
        if self.shuffle:
            random.shuffle(indexes)
        return indexes

    def __len__(self):
        data_size = len(self.data)
        return data_size // self.batch_size + (data_size % self.batch_size != 0)

    def _run(self):
        batch = []
        for i in self._updated_indexes():
            if len(batch) >= self.batch_size:
                self.data_batches.put(batch.copy())
                batch = []
            batch.append(self.data[i])

        if 0 < self.batch_size < len(batch):
            batch += [batch[-1]*(self.batch_size - len(batch))]
            self.data_batches.put(batch.copy())

        self.ready_flag = True

    def refresh(self):
        assert self.ready_flag

        if self.work_thread is not None:
            self.work_thread.join()

        self.ready_flag = False
        self.work_thread = Thread(target=self._run)
        self.work_thread.start()

    def ready(self):
        return not self.data_batches.empty()

    def finish(self):
        return self.ready_flag

    def get_batch(self):
        while not self.ready():
            pass

        return self.data_batches.get()

    @staticmethod
    def separate_batch(batch):
        return np.array(batch)[:, 0], np.array(batch)[:, 1]

    def generator(self, verbose=0, infinite=True):
        while True:
            print('START REFRESH')

            while not self.finish():
                pass

            self.refresh()

            batches_count = len(self)

            print('REFRESH')

            _range = range(batches_count) \
                if verbose == 0 else \
                tqdm.tqdm(range(batches_count))

            for i in _range:
                yield self.separate_batch(self.get_batch())

            if not infinite:
                break
