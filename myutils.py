from scipy.misc import imread, imsave
import tensorflow as tf
import numpy as np
import os


def load_one_image(fpath):
    img = imread(fpath)
    return img


def load_images(input_dir, batch_shape):
    images = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]

    for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
        image = load_one_image(filepath) / 255.0
        images[idx, :, :, :] = image * 2.0 - 1.0
        filenames.append(os.path.basename(filepath))
        idx += 1

        if idx == batch_size:
            yield filenames, images
            filenames = []
            images = np.zeros(batch_shape)
            idx = 0

    if idx > 0:
        yield filenames, images


def save_images(images, filenames, output_dir):
    for i, filename in enumerate(filenames):
        basename = os.path.basename(filename)
        img = (((images[i, :, :, :] + 1.0) * 0.5) * 255.0).astype(np.uint8)
        imsave(os.path.join(output_dir, basename), img)
