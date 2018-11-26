import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shield.constants import \
    ACCURACY_NPZ_FILENAME, \
    NUM_SAMPLES_VALIDATIONSET, \
    TOP5_ACCURACY_NPZ_FILENAME
from shield.opts import model_checkpoint_map, model_class_map
from shield.utils.slim.preprocessing.inception_preprocessing import \
    preprocess_image
from shield.utils.io import load_image_data_from_tfrecords
from shield.utils.metering import AccuracyMeter, TopKAccuracyMeter


def evaluate(tfrecord_paths_expression,
             model_name,
             output_dir,
             model_checkpoint_path=None,
             load_jpeg=False,
             decode_pixels=False,
             cropping=True):
    """Evaluates and saves the performance of a model on the given tfrecords.

    Args:
        tfrecord_paths_expression (str):
            Wildcard expression for path to the tfrecord files.
        model_name (str):
            Name of the model to be evaluated.
            It should correspond to one of the models in `opts.py`.
        output_dir (str):
            The results are saved to this directory.
        model_checkpoint_path (str):
            If not None, the model weights are loaded from this path.
        load_jpeg (bool):
            Whether the tfrecord contains images in JPEG binary format.
        decode_pixels (bool):
            Whether the tfrecord contains image data
            in pixel space or contains normalized values.
        cropping (bool):
            Whether a central cropping of 0.875 is to be
            applied during evaluation.
    """

    # Define model class
    Model = model_class_map[model_name]

    # Define meters we want to track
    accuracy = AccuracyMeter()
    top5_accuracy = TopKAccuracyMeter(k=5)

    # Define preprocessing function
    img_size = Model.default_image_size
    preprocessing_fn = \
        (lambda x: preprocess_image(
            x, img_size, img_size,
            cropping=cropping,
            is_training=False)) \
        if decode_pixels else lambda x: x

    with tf.Graph().as_default():
        # Initialize the data loader node in the tensorflow graph
        ids, X, y_true = load_image_data_from_tfrecords(
            tfrecord_paths_expression,
            preprocessing_fn=preprocessing_fn,
            load_jpeg=load_jpeg,
            decode_pixels=decode_pixels,
            image_size=Model.default_image_size)

        # Create rest of the tensorflow graph
        model = Model(X)
        top_k_confidences, top_k_preds = \
            tf.nn.top_k(model.fprop(X)['probs'], k=5)

        # Initialize the tensorflow session
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.,
            allow_growth=True)
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options))

        with sess.as_default():
            # Initialize and load model weights
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            if model_checkpoint_path is None:
                model_checkpoint_path = model_checkpoint_map[model_name]
            model.load_weights(model_checkpoint_path, sess=sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET, unit='imgs') as pbar:
                    while not coord.should_stop():
                        # Get predictions for a batch
                        ids_, y_true_, top_k_preds_ = \
                            sess.run([ids, y_true, top_k_preds])

                        top_k_preds_ = np.squeeze(top_k_preds_)
                        y_pred_ = top_k_preds_[:, 0]

                        # Update meters
                        accuracy.offer(y_pred_, y_true_, ids=ids_)
                        top5_accuracy.offer(top_k_preds_, y_true_, ids=ids_)

                        pbar.set_postfix(
                            top_1_accuracy=accuracy.evaluate(),
                            top_5_accuracy=top5_accuracy.evaluate())
                        pbar.update(len(ids_))

            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)

            finally:
                accuracy.save(
                    os.path.join(output_dir, ACCURACY_NPZ_FILENAME))
                top5_accuracy.save(
                    os.path.join(output_dir, TOP5_ACCURACY_NPZ_FILENAME))
