import tensorflow as tf
from cleverhans.model import Model as CleverHansModel

import inception_v4_base

slim = tf.contrib.slim


class InceptionV4(CleverHansModel):
    default_image_size = inception_v4_base.default_image_size

    def __init__(self, x, checkpoint_path, num_classes=1001, is_training=False):
        super(InceptionV4, self).__init__()

        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path

        with slim.arg_scope(inception_v4_base.inception_v4_arg_scope()):
            net, end_points = inception_v4_base.inception_v4(x, num_classes=num_classes,
                                                             is_training=is_training, reuse=None)

        self.variables_to_restore = slim.get_variables_to_restore(exclude=[])
        saver = tf.train.Saver(self.variables_to_restore)
        self.load_weights = lambda sess: saver.restore(sess, checkpoint_path)

        end_points['logits'] = end_points['Logits']
        end_points['probs'] = end_points['Predictions']

        self.x = x
        self.net = net
        self.end_points = end_points

    def fprop(self, x):
        assert x is self.x
        return self.end_points