import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from shield.constants import \
    ACCURACY_NPZ_FILENAME, \
    NUM_SAMPLES_VALIDATIONSET, \
    PREPROCESSED_TFRECORD_FILENAME, \
    TOP5_ACCURACY_NPZ_FILENAME
from shield.opts import \
    defense_fn_map, model_checkpoint_map, model_class_map, tf_defenses
from shield.utils.slim.preprocessing.inception_preprocessing import \
    preprocess_image
from shield.utils.io import encode_tf_examples, load_image_data_from_tfrecords
from shield.utils.metering import AccuracyMeter, TopKAccuracyMeter


def preprocess(tfrecord_paths_expression,
               model_name,
               defense_name,
               defense_options,
               output_dir,
               model_checkpoint_path=None,
               load_jpeg=False,
               decode_pixels=False,
               cropping=True):
    """Applies preprocessing to images and saves them.

    Args:
        tfrecord_paths_expression (str):
            Wildcard expression for path to the tfrecord files.
        model_name (str):
            Name of the model to be evaluated.
            It should correspond to one of the models in `opts.py`.
        defense_name (str):
            Name of the preprocessing technique to be applied.
            It should correspond to one of the defenses in `opts.py`.
        defense_options(dict):
            Options for the preprocessing method.
            This dictionary is passed as keyword arguments to the `defense_fn`
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
            applied before evaluation.
    """

    # Define model and defense function
    Model = model_class_map[model_name]
    defense_fn = defense_fn_map[defense_name]

    # Define meters we want to track
    accuracy = AccuracyMeter()
    top5_accuracy = TopKAccuracyMeter(k=5)

    # Define preprocessing functions
    img_size = Model.default_image_size
    preprocessing_fn = \
        (lambda x: tf.cast(
            255. * (x + 1.) / 2.,
            tf.uint8)) \
        if not decode_pixels else lambda x: x
    preprocessing_fn_model = \
        lambda x: preprocess_image(
            x, img_size, img_size,
            cropping=cropping,
            is_training=False)

    # Define the writer that will save the output after preprocessing
    writer = tf.python_io.TFRecordWriter(
        os.path.join(output_dir, PREPROCESSED_TFRECORD_FILENAME))

    with tf.Graph().as_default():
        # Initialize the data loader node in the tensorflow graph
        ids, images, y_true = load_image_data_from_tfrecords(
            tfrecord_paths_expression,
            image_size=img_size,
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
            # Create rest of the tensorflow graph
            X = tf.placeholder(
                shape=(None, img_size, img_size, 3),
                dtype=tf.float32)
            Xp = tf.map_fn(lambda x: preprocessing_fn_model(x), X)
            model = Model(Xp)

            y_pred_prep = tf.argmax(model.fprop(Xp)['probs'], 1)
            _, top_k_preds_prep = \
                tf.nn.top_k(model.fprop(Xp)['probs'], k=5)

            # Initialize and load model weights
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            if model_checkpoint_path is None:
                model_checkpoint_path = model_checkpoint_map[model_name]
            model.load_weights(model_checkpoint_path, sess=sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            n = 500
            i = 0
            try:
                with tqdm(total=n, unit='imgs') as pbar:
                    while not coord.should_stop():
                        # Load the images
                        ids_, images_, y_true_ = sess.run([ids, images, y_true])

                        # Apply preprocessing for non-TF defenses
                        if defense_name not in tf_defenses:
                            images_ = np.array(
                                [defense_fn(image_, **defense_options)
                                 for image_ in images_])

                        # Get model predictions on the preprocessed images
                        y_pred_prep_, top_k_preds_prep_ = sess.run(
                            [y_pred_prep, top_k_preds_prep],
                            feed_dict={X: images_ / 255.})

                        top_k_preds_prep_ = np.squeeze(top_k_preds_prep_)

                        # Update meter
                        accuracy.offer(
                            y_pred_prep_, y_true_, ids=ids_)
                        top5_accuracy.offer(
                            top_k_preds_prep_, y_true_, ids=ids_)

                        # Save the preprocessed images
                        for example in encode_tf_examples(
                                ids_, images_, y_true_):
                            writer.write(example.SerializeToString())

                        pbar.set_postfix(
                            top_1_accuracy=accuracy.evaluate(),
                            top_5_accuracy=top5_accuracy.evaluate())
                        pbar.update(len(ids_))

                        i += len(ids_)
                        if i >= n:
                            break

            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)

            finally:
                writer.close()

                accuracy.save(
                    os.path.join(output_dir, ACCURACY_NPZ_FILENAME))
                top5_accuracy.save(
                    os.path.join(output_dir, TOP5_ACCURACY_NPZ_FILENAME))
