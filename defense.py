from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os

import numpy as np
from scipy.misc import imread
from scipy.ndimage import median_filter

import tensorflow as tf
import random

#from inception_preprocessing import preprocess_image
import mymodels

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_file', '', 'Output file to save labels.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def load_images(input_dir, batch_shape):
  """Read png images from input directory in batches.

  Args:
    input_dir: input directory
    batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

  Yields:
    filenames: list file names without path of each image
      Lenght of this list could be less than batch_size, in this case only
      first few images of the result are elements of the minibatch.
    images: array with all images from this batch
  """
  images = np.zeros(batch_shape)
  filenames = []
  idx = 0
  batch_size = batch_shape[0]
  for filepath in tf.gfile.Glob(os.path.join(input_dir, '*.png')):
    with tf.gfile.Open(filepath, "rb") as f:
      image = imread(f, mode='RGB').astype(np.float) / 255.0
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


def main(_):
  batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
  num_classes = 1001

  tf.logging.set_verbosity(tf.logging.INFO)

  with tf.Graph().as_default():
    # Prepare graph
    x_input = tf.placeholder(tf.float32, shape=batch_shape)

    models = [mymodels.PNasNetLarge,
              mymodels.EnsAdvInceptionResNetV2]
    models = [model(num_classes) for model in models]
    endpoints = []

    for i, model in enumerate(models):
      select_rows = tf.random_uniform((FLAGS.image_height,)) > 0.3
      select_cols = tf.random_uniform((FLAGS.image_width,)) > 0.3
      drop_rows = tf.boolean_mask(x_input, select_rows, axis=1)
      drop_cols = tf.boolean_mask(drop_rows, select_cols, axis=2)
      resized = tf.image.resize_images(drop_cols, (FLAGS.image_height, FLAGS.image_width),
                                       method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
      endpoints.append(model(resized))

    with tf.Session() as sess:
      for item in models:
        item.restore(sess)

      top_n = [5, 5, 8, 8]
      weight = [1.00, 0.75, 0.75, 1.00]
      with tf.gfile.Open(FLAGS.output_file, 'w') as out_file:
        for filenames, images in load_images(FLAGS.input_dir, batch_shape):
          endpts = []
          endpts.extend(sess.run(endpoints, feed_dict={x_input: images}))
          images2 = median_filter(images.reshape(-1, FLAGS.image_width, 3), 5).reshape(-1, FLAGS.image_height, FLAGS.image_width, 3)
          endpts.extend(sess.run(endpoints, feed_dict={x_input: images2}))

          for i, filename in enumerate(filenames):
            score = np.zeros(num_classes)
            for j, item in enumerate(endpts):
              prob = endpts[j][i, :]
              prob /= prob.max()
              top = np.argsort(prob)[-top_n[j]:]
              for label in top:
                score[label] += prob[label] * random.normalvariate(1, 0.2) * weight[j]

            label = score.argmax()
            out_file.write('{0},{1}\n'.format(filename, label))


if __name__ == '__main__':
  tf.app.run()
# vim: softtabstop=2 shiftwidth=2 expandtab
