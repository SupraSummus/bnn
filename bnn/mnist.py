from mnist import MNIST
import numpy


mndata = MNIST('./data', return_type='numpy', mode='rounded_binarized')
image_size = 28
train_images_count = 60000


train_images, train_labels = mndata.load_training()
assert train_images.shape == (train_images_count, image_size ** 2)


def load_training(workplace, spatial_offset=0, data_offset=0):
    assert len(workplace.shape) == 4
    assert workplace.shape[0] >= 1
    assert workplace.shape[1] >= image_size + spatial_offset
    assert workplace.shape[2] >= image_size + spatial_offset
    assert workplace.shape[3] + data_offset <= train_images_count

    images = train_images[data_offset:data_offset + workplace.shape[3]]
    images = numpy.moveaxis(images, 0, -1)
    images = images.reshape(image_size, image_size, workplace.shape[3])
    workplace[0, spatial_offset:spatial_offset+image_size, spatial_offset:spatial_offset+image_size] = images


def str_bool_array(arr):
    assert len(arr.shape) == 2
    ret = []
    for row in arr:
        for el in row:
            ret.append('#' if el else '.')
        ret.append('\n')
    return ''.join(ret)


batch_size = 1024
for i in range(train_images_count // batch_size):
    workplace = numpy.zeros(shape=(1024, image_size + 2, image_size + 2, batch_size), dtype=bool)
    load_training(workplace, spatial_offset=1, data_offset=batch_size*i)
    print(f'batch {i}, mem {workplace.size / 2**20}MB')
    print(str_bool_array(workplace[0, :, :, 0]))
    print(train_labels[batch_size*i])
