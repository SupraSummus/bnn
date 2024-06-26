import numpy

from mnist import MNIST

from . import ConvBNN

mndata = MNIST('./data', return_type='numpy', mode='rounded_binarized')
image_size = 28
train_images_count = 60000


train_images, train_labels = mndata.load_training()
assert train_images.shape == (train_images_count, image_size ** 2)
assert train_labels.shape == (train_images_count,)


def load_training(workplace, spatial_offset=0, data_offset=0):
    assert len(workplace.shape) == 4
    assert workplace.shape[0] >= 1
    assert workplace.shape[1] >= image_size + spatial_offset
    assert workplace.shape[2] >= image_size + spatial_offset
    assert workplace.shape[3] + data_offset <= train_images_count

    images = train_images[data_offset:data_offset + workplace.shape[3]]
    images = numpy.moveaxis(images, 0, -1)
    images = images.reshape(image_size, image_size, workplace.shape[3])
    workplace[
        0,
        spatial_offset:spatial_offset + image_size,
        spatial_offset:spatial_offset + image_size,
    ] = images


def str_bool_array(arr):
    assert len(arr.shape) == 2
    ret = []
    for row in arr:
        for el in row:
            ret.append('#' if el else '.')
        ret.append('\n')
    return ''.join(ret)


batch_size = 128
dtype = numpy.uint16

bnn = ConvBNN(
    dim=2,
    margin=1,
    sample_depth=1,
    neuron_count=1024,
)

for i in range(train_images_count // batch_size):
    print(f'batch {i}')

    signal = bnn.make_workplace(batch_size=batch_size, sample_size=(image_size, image_size), dtype=bool)
    load_training(signal, spatial_offset=1, data_offset=batch_size * i)

    bnn.infer(signal, dtype=dtype)

    output_slice = (slice(-10, None), image_size // 2 + bnn.margin, image_size // 2 + bnn.margin)
    predictions = signal[output_slice]
    labels = numpy.array([
        train_labels[batch_size * i:batch_size * (i + 1)] == v
        for v in range(10)
    ], dtype=bool)
    assert predictions.shape == labels.shape
    print(f'wrong {numpy.sum(predictions != labels)}/{10 * batch_size}')

    too_much = bnn.make_workplace(batch_size=batch_size, sample_size=(image_size, image_size), dtype=dtype)
    not_enough = bnn.make_workplace(batch_size=batch_size, sample_size=(image_size, image_size), dtype=dtype)
    too_much[output_slice] = (predictions > labels) * 1024
    not_enough[output_slice] = (predictions < labels) * 1024
    bnn.compute_error(signal, too_much, not_enough, dtype=dtype)

    break
