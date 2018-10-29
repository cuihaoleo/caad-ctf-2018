#!/usr/bin/env python3

from cleverhans.attacks import FastGradientMethod, MomentumIterativeMethod
import numpy as np
import tensorflow as tf

from myutils import load_images, save_images
import mymodels

slim = tf.contrib.slim


tf.flags.DEFINE_string(
    'input_dir', '', 'Input directory with images.')

tf.flags.DEFINE_string(
    'output_dir', '', 'Output directory with images.')

tf.flags.DEFINE_integer(
    'target', -1, 'target')

tf.flags.DEFINE_float(
    'max_epsilon', 16.0, 'Maximum size of adversarial perturbation.')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size', 16, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS


def main(_):
    eps = 2.0 * FLAGS.max_epsilon / 255.0
    batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]
    num_classes = 1001

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():
        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)

        #model = mymodels.InceptionV3(num_classes)
        #model = mymodels.EnsAdvInceptionResNetV2(num_classes)
        #model = mymodels.NasNetLarge(num_classes)
        model = mymodels.EnsembleModel(num_classes,
                                       [mymodels.InceptionV3(num_classes),
                                        mymodels.Ens3AdvInceptionV3(num_classes),
                                        mymodels.Ens4AdvInceptionV3(num_classes),
                                        mymodels.EnsAdvInceptionResNetV2(num_classes),
                                        mymodels.AdvInceptionV3(num_classes)],
                                       weight=[ 4.0, 1.0, 1.0, 1.0, 4.0 ])

        if FLAGS.target < 0:
            target = None
        else:
            target = tf.constant(np.zeros([FLAGS.batch_size]) + FLAGS.target, tf.int32)
            target = tf.one_hot(target, num_classes)

        with tf.Session() as sess:
            #attacker = FastGradientMethod(model)
            attacker = MomentumIterativeMethod(model)
            x_adv = attacker.generate(x_input,
                                      eps=eps, eps_iter=eps/5.0,
                                      clip_min=-1., clip_max=1.,
                                      y_target=target, nb_iter=80)

            model.restore(sess)
            for filenames, images in load_images(FLAGS.input_dir, batch_shape):
                adv_images = sess.run(x_adv, feed_dict={x_input: images})
                save_images(adv_images, filenames, FLAGS.output_dir)


if __name__ == '__main__':
    tf.app.run()
