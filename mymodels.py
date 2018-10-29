from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import inception_resnet_v2
import nasnet
import pnasnet
import numpy as np

from cleverhans.model import Model
from tensorflow.contrib.slim.nets import inception

import os

slim = tf.contrib.slim
BASEDIR = os.path.dirname(os.path.abspath(__file__))


class MyImageNetModel(Model):
    def __init__(self, nb_classes):
        Model.__init__(self, None, nb_classes)

    def _build_internal(self, ckpt, scope_func, model_func):
        self.ckpt = ckpt
        self.scope_func = scope_func
        self.model_func = model_func

        dummy = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        with slim.arg_scope(scope_func()):
            with tf.variable_scope(self.ckpt, reuse=tf.AUTO_REUSE):
                model_func(dummy, num_classes=self.nb_classes,
                           is_training=False)

        var_dict = {}
        for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                     scope=self.ckpt):
            var_dict[var.op.name[len(self.ckpt) + 1:]] = var
        self.saver = tf.train.Saver(var_dict)

    def fprop(self, x_input):
        with slim.arg_scope(self.scope_func()):
            with tf.variable_scope(self.ckpt, reuse=tf.AUTO_REUSE):
                # x_input = x_input * 2.0 - 1.0
                _, end_points = self.model_func(x_input,
                                                num_classes=self.nb_classes,
                                                is_training=False)

        return {self.O_LOGITS: end_points["Logits"],
                self.O_PROBS: end_points["Predictions"]}

    def restore(self, sess):
        ckpt_path = os.path.join(BASEDIR, "checkpoints", self.ckpt)
        self.saver.restore(sess, ckpt_path)


class InceptionV3(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("inception_v3_2016_08_28", "inception_v3.ckpt")
        self._build_internal(ckpt,
                             inception.inception_v3_arg_scope,
                             inception.inception_v3)


class AdvInceptionV3(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("adv_inception_v3_2017_08_18",
                            "adv_inception_v3.ckpt")
        self._build_internal(ckpt,
                             inception.inception_v3_arg_scope,
                             inception.inception_v3)


class Ens3AdvInceptionV3(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("ens3_adv_inception_v3_2017_08_18",
                            "ens3_adv_inception_v3.ckpt")
        self._build_internal(ckpt,
                             inception.inception_v3_arg_scope,
                             inception.inception_v3)


class Ens4AdvInceptionV3(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("ens4_adv_inception_v3_2017_08_18",
                            "ens4_adv_inception_v3.ckpt")
        self._build_internal(ckpt,
                             inception.inception_v3_arg_scope,
                             inception.inception_v3)


class EnsAdvInceptionResNetV2(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("ens_adv_inception_resnet_v2_2017_08_18",
                            "ens_adv_inception_resnet_v2.ckpt")
        self._build_internal(ckpt,
                             inception_resnet_v2.inception_resnet_v2_arg_scope,
                             inception_resnet_v2.inception_resnet_v2)


class NasNetLarge(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("nasnet-a_large_04_10_2017", "model.ckpt")
        self._build_internal(ckpt,
                             nasnet.nasnet_large_arg_scope,
                             nasnet.build_nasnet_large)


class PNasNetLarge(MyImageNetModel):
    def __init__(self, nb_classes):
        MyImageNetModel.__init__(self, nb_classes)
        ckpt = os.path.join("pnasnet-5_large_2017_12_13", "model.ckpt")
        self._build_internal(ckpt,
                             pnasnet.pnasnet_large_arg_scope,
                             pnasnet.build_pnasnet_large)


class EnsembleModel(Model):
    def __init__(self, nb_classes, models, weight=None):
        Model.__init__(self, None, nb_classes)
        self.models = list(models)
        if weight is None:
            weight = np.full(len(models), 1.0)
        self.weight = np.array(weight)

    def fprop(self, x_input):
        logits = self.models[0].get_logits(x_input) * self.weight[0]

        for i, mod in enumerate(self.models[1:]):
            logits += mod.get_logits(x_input) * self.weight[i + 1]

        return {self.O_LOGITS: logits}

    def restore(self, sess):
        for mod in self.models:
            mod.restore(sess)
