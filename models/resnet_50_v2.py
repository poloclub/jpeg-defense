import tensorflow as tf

from cleverhans.model import Model as CleverHansModel

from utils.slim.nets.resnet_v2 import resnet_arg_scope, resnet_v2_50

slim = tf.contrib.slim


class ResNet50v2(CleverHansModel):
    default_image_size = 299

    def __init__(self, x, checkpoint_path, num_classes=1001, is_training=False):
        super(ResNet50v2, self).__init__()

        self.num_classes = num_classes
        self.checkpoint_path = checkpoint_path

        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2_50(x, num_classes=num_classes,
                                           is_training=is_training, reuse=None)

        end_points = dict(end_points)

        self.variables_to_restore = slim.get_variables_to_restore(exclude=[])
        saver = tf.train.Saver(self.variables_to_restore)
        self.load_weights = lambda sess: saver.restore(sess, checkpoint_path)

        end_points['logits'] = end_points['resnet_v2_50/spatial_squeeze']
        end_points['probs'] = tf.nn.softmax(end_points['logits'])

        self.x = x
        self.net = net
        self.end_points = end_points

    def fprop(self, x):
        if x is self.x:
            return self.end_points
        else:
            with slim.arg_scope(resnet_arg_scope()):
                net, end_points = resnet_v2_50(x, num_classes=self.num_classes,
                                               is_training=False, reuse=tf.AUTO_REUSE)

            end_points = dict(end_points)
            end_points['logits'] = end_points['resnet_v2_50/spatial_squeeze']
            end_points['probs'] = tf.nn.softmax(end_points['logits'])

            return end_points