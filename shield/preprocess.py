import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shield.constants import \
    NUM_SAMPLES_VALIDATIONSET, PREPROCESSED_TFRECORD_FILENAME
from shield.opts import defense_fn_map, tf_defenses
from shield.utils.io import encode_tf_examples, load_image_data_from_tfrecords


def preprocess(tfrecord_paths_expression,
               defense_name,
               defense_options,
               output_dir,
               image_size=None,
               load_jpeg=False,
               decode_pixels=False):
    """Applies preprocessing to images and saves them.

    Args:
        tfrecord_paths_expression (str):
            Wildcard expression for path to the tfrecord files.
        defense_name (str):
            Name of the preprocessing technique to be applied.
            It should correspond to one of the defenses in `opts.py`.
        defense_options(dict):
            Options for the preprocessing method.
            This dictionary is passed as keyword arguments to the `defense_fn`
        output_dir (str):
            The results are saved to this directory.
        load_jpeg (bool):
            Whether the tfrecord contains images in JPEG binary format.
        decode_pixels (bool):
            Whether the tfrecord contains image data
            in pixel space or contains normalized values.
    """

    # Define model and defense function
    defense_fn = defense_fn_map[defense_name]

    # Define preprocessing function
    preprocessing_fn = \
        (lambda x: tf.cast(
            255. * (x + 1.) / 2.,
            tf.uint8)) \
        if not decode_pixels else lambda x: x

    # Define the writer that will save the output after preprocessing
    writer = tf.python_io.TFRecordWriter(
        os.path.join(output_dir, PREPROCESSED_TFRECORD_FILENAME))

    with tf.Graph().as_default():
        # Initialize the data loader node in the tensorflow graph
        ids, images, labels = load_image_data_from_tfrecords(
            tfrecord_paths_expression,
            image_size=image_size,
            preprocessing_fn=preprocessing_fn,
            load_jpeg=load_jpeg,
            decode_pixels=decode_pixels)

        # Create rest of the tensorflow graph for TF defenses
        if defense_name in tf_defenses:
            images = tf.map_fn(
                lambda x: defense_fn(x, **defense_options), images)

        # Initialize the tensorflow session
        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=1.,
            allow_growth=True)
        sess = tf.Session(
            config=tf.ConfigProto(
                allow_soft_placement=True,
                gpu_options=gpu_options))

        with sess.as_default():
            # Initialize variables for the input loader
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET,
                          unit='imgs', ncols=100) as pbar:
                    while not coord.should_stop():
                        # Load the images
                        ids_, images_, labels_ = sess.run([ids, images, labels])

                        # Apply preprocessing for non-TF defenses
                        if defense_name not in tf_defenses:
                            images_ = np.array(
                                [defense_fn(image_, **defense_options)
                                 for image_ in images_])

                        # Save the preprocessed images
                        for example in encode_tf_examples(
                                ids_, images_, labels_):
                            writer.write(example.SerializeToString())

                        pbar.update(len(ids_))

            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)
