import numpy as np

from Utility.ArrayConvertor import *


def load_data(file_path):
    with open(file_path, 'rb') as file_handler:
        magic_number = byte_array_to_int32(file_handler.read(4), '>')
        if magic_number != 2051:
            raise f'not valid magic number:{magic_number}'

        counts = byte_array_to_int32(file_handler.read(4), '>')
        rows = byte_array_to_int32(file_handler.read(4), '>')
        columns = byte_array_to_int32(file_handler.read(4), '>')
        images_data_size = counts * rows * columns

        images_data = file_handler.read(images_data_size)

        data = np.frombuffer(images_data, dtype=np.uint8)

        data.shape = (counts, rows, columns)
        data = data.astype(np.float32)
        data /= 255.0

        return data


def load_labels(file_path):
    with open(file_path, 'rb') as file_handler:
        magic_number = byte_array_to_int32(file_handler.read(4), '>')
        if magic_number != 2049:
            raise f'not valid magic number:{magic_number}'

        count = byte_array_to_int32(file_handler.read(4), '>')

        y_data = np.zeros((count, 10))

        for i in range(count):
            read_label = file_handler.read(1)
            index_label = int.from_bytes(read_label, byteorder='big', signed=True)
            y_data[i, index_label] = 1.0

        return y_data