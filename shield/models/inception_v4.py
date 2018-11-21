from cleverhans.model import Model as CleverHansModel
import tensorflow as tf

from shield.models.inception_v4_base import inception_v4, inception_v4_arg_scope

slim = tf.contrib.slim


def _get_updated_endpoints(original_end_points):
    """Adds the keys 'logits' and 'probs' to the
    end points dictionary of Inception-v4.

    Args:
        original_end_points (dict): Original dictionary of end points

    Returns:
        dict: Dictionary of end points with the new keys.
    """

    end_points = dict(original_end_points)
    end_points['logits'] = end_points['Logits']
    end_points['probs'] = end_points['Predictions']

    return end_points


class InceptionV4(CleverHansModel):
    """Wrapper class for the Inception-v4 model created using tensorflow slim.

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
        """Initializes the tensorflow graph for the Inception-v4 model.

        Args:
            x (tf.Variable): The variable in the tensorflow graph
                that feeds into the model nodes.
            num_classes (int):
                Number of predicted classes for classification tasks.
                If 0 or None, the features before the logit layer are returned.
            is_training (bool): Whether batch_norm layers are in training mode.
        """

        super(InceptionV4, self).__init__()

        self.x = x
        self.num_classes = num_classes

        # populating the tensorflow graph
        with slim.arg_scope(inception_v4_arg_scope()):
            net, end_points = inception_v4(
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
            with slim.arg_scope(inception_v4_arg_scope()):
                net, end_points = inception_v4(
                    x, num_classes=self.num_classes,
                    is_training=False, reuse=tf.AUTO_REUSE)

            return _get_updated_endpoints(end_points)
