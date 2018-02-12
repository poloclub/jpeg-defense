import tensorflow as tf
from tensorflow.contrib.slim.nets import vgg

from cleverhans.model import Model as CleverHansModel

slim = tf.contrib.slim


class VGG16(CleverHansModel):
    def __init__(self, x, checkpoint_file, num_classes=1000, is_training=False):
        super(VGG16, self).__init__()

        self.num_classes = num_classes

        with slim.arg_scope(vgg.vgg_arg_scope()):
            net, end_points = vgg.vgg_16(x, num_classes=num_classes, is_training=is_training)

        variables_to_restore = slim.get_variables_to_restore(exclude=[])
        saver = tf.train.Saver(variables_to_restore)
        self.load_weights = lambda sess: saver.restore(sess, checkpoint_file)

        end_points['logits'] = net
        end_points['probs'] = tf.nn.softmax(net)

        self.x = x
        self.net = net
        self.end_points = end_points

    # Cleverhans methods
    def get_layer_names(self):
        return ['logits', 'probs']

    def fprop(self, x):
        assert x is self.x
        return self.end_points
