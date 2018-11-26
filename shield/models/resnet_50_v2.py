from cleverhans.model import Model as CleverHansModel
import tensorflow as tf
from tensorflow.contrib.slim.python.slim.nets.resnet_v2 import \
    resnet_arg_scope, resnet_v2_50


slim = tf.contrib.slim


def _get_updated_endpoints(original_end_points):
    """Adds the keys 'logits' and 'probs' to the
    end points dictionary of ResNet50-v2.

    Args:
        original_end_points (dict): Original dictionary of end points

    Returns:
        dict: Dictionary of end points with the new keys.
    """

    end_points = dict(original_end_points)
    end_points['logits'] = tf.squeeze(end_points['resnet_v2_50/logits'], [1, 2])
    end_points['probs'] = tf.nn.softmax(end_points['logits'])

    return end_points


class ResNet50v2(CleverHansModel):
    """Wrapper class for the ResNet50-v2 model loaded from tensorflow slim.

    Attributes:
        x (tf.Variable): The variable in the tensorflow graph
            that feeds into the model nodes.
        num_classes (int): Number of predicted classes for classification tasks.
            If 0 or None, the features before the logit layer are returned.
        default_image_size (int): The image size accepted by
            the model by default.
        end_points (dict): Dictionary with pointers to
            intermediate variables of the model.
        variables_to_restore (list): Names of tensorflow variables
            to be restored in the graph.
    """

    default_image_size = 299

    def __init__(self, x, num_classes=1001, is_training=False):
        """Initializes the tensorflow graph for the ResNet50-v2 model.

        Args:
            x (tf.Variable): The variable in the tensorflow graph
                that feeds into the model nodes.
            num_classes (int):
                Number of predicted classes for classification tasks.
                If 0 or None, the features before the logit layer are returned.
            is_training (bool): Whether batch_norm layers are in training mode.
        """

        super(ResNet50v2, self).__init__()

        self.x = x
        self.num_classes = num_classes

        # populating the tensorflow graph
        with slim.arg_scope(resnet_arg_scope()):
            net, end_points = resnet_v2_50(
                x, num_classes=num_classes,
                is_training=is_training, reuse=None)

        self.end_points = _get_updated_endpoints(end_points)
        self.variables_to_restore = slim.get_variables_to_restore(exclude=[])

    def load_weights(self, checkpoint_path, sess=None):
        """Load weights from a checkpoint file into the tensorflow graph.

        Args:
            checkpoint_path (str): Path to the checkpoint file.
            sess (tf.Session): The tensorflow session holding the model graph.
        """

        if sess is None:
            sess = tf.get_default_session()
        assert sess is not None

        saver = tf.train.Saver(self.variables_to_restore)
        saver.restore(sess, checkpoint_path)

    def get_params(self):
        """Lists the model's parameters.

        Returns:
            list: A list of the model's parameters.
        """

        return None

    def fprop(self, x):
        """Exposes all the layers of the model.

        Args:
            x (tf.Variable): Tensor which is input to the model.

        Returns:
            dict: A dictionary mapping layer names to the corresponding
                 node in the tensorflow graph.
        """

        if x is self.x:
            return self.end_points

        else:
            with slim.arg_scope(resnet_arg_scope()):
                net, end_points = resnet_v2_50(
                    x, num_classes=self.num_classes,
                    is_training=False, reuse=tf.AUTO_REUSE)

            return _get_updated_endpoints(end_points)
