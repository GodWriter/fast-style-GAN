import os
import numpy as np
import PIL.Image as im

class shipData(object):

    def __init__(self,
                 shape,
                 folder_path):
        self.batch_id = 0
        self.shape = shape
        self.data = self.load_image(folder_path)
        self.data_len = self.data.shape[0]

    def load_image(self, folder_path):
        image_list = os.listdir(folder_path)
        print("folder_path: ", folder_path)
        data_shape = [len(image_list)] + self.shape + [3]
        print("data_shape: ", data_shape)

        data = np.zeros(data_shape)
        for id_, img in enumerate(image_list):
            image = im.open(os.path.join(folder_path, img))
            image = image.resize(self.shape, im.LANCZOS)
            data[id_,:,:,:] = np.float32(image)

        return data

    def next_batch(self, batch_size):
        if self.batch_id == len(self.data):
            self.batch_id = 0

        batch_data = (self.data[self.batch_id:min(self.batch_id +
                                                  batch_size, self.data_len)])
        self.batch_id = min(self.batch_id + batch_size, self.data_len)

        return batch_data

