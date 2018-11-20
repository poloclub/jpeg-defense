import os

import tensorflow as tf
from tqdm import tqdm

from shield.constants import \
    ATTACKED_TFRECORD_FILENAME, \
    NORMALIZED_L2_DISTANCE_NPZ_FILENAME, \
    NUM_SAMPLES_VALIDATIONSET
from shield.opts import attack_class_map, model_checkpoint_map, model_class_map
from utils.slim.preprocessing.inception_preprocessing import preprocess_image
from shield.utils.io import encode_tf_examples, load_image_data_from_tfrecords
from shield.utils.metering import AverageNormalizedL2DistanceMeter


def attack(tfrecord_paths_expression,
           model_name,
           attack_name,
           attack_options,
           output_dir,
           model_checkpoint_path=None,
           load_jpeg=False,
           decode_pixels=False):
    """Computes an attack on the given model and saves the output.

    Args:
        tfrecord_paths_expression (str):
            Wildcard expression for path to the tfrecord files.
        model_name (str):
            Name of the model to be evaluated.
            It should correspond to one of the models in `opts.py`.
        attack_name (str):
            Name of the attack to be performed.
            It should correspond to one of the attacks in `opts.py`.
        attack_options (dict):
            Options for setting up the attack.
            This dictionary is passed as keyword arguments
            to the `generate` method of the attack.
        output_dir (str):
            The results are saved to this directory.
        model_checkpoint_path (str):
            If not None, the model weights are loaded from this path.
        load_jpeg (bool):
            Whether the tfrecord contains images in JPEG binary format.
        decode_pixels (bool):
            Whether the tfrecord contains image data
            in pixel space or contains normalized values.
    """

    # Define model and attack classes
    Model = model_class_map[model_name]
    Attack = attack_class_map[attack_name]

    # Define meters we want to track
    normalized_l2_distance = AverageNormalizedL2DistanceMeter()

    # Define preprocessing function
    img_size = Model.default_image_size
    preprocessing_fn = \
        (lambda x: preprocess_image(
            x, img_size, img_size,
            cropping=False,
            is_training=False)) \
        if decode_pixels else lambda x: x

    # Define the writer that will save the output of the attack
    writer = tf.python_io.TFRecordWriter(
        os.path.join(output_dir, ATTACKED_TFRECORD_FILENAME))

    with tf.Graph().as_default():
        # Initialize the data loader node in the tensorflow graph
        ids, X_ben, y_true = load_image_data_from_tfrecords(
            tfrecord_paths_expression,
            preprocessing_fn=preprocessing_fn,
            load_jpeg=load_jpeg,
            decode_pixels=decode_pixels,
            image_size=Model.default_image_size)

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
            model = Model(X_ben)
            X_adv = Attack(model).generate(
                X_ben, **attack_options)

            # Initialize and load model weights
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()
            if model_checkpoint_path is None:
                model_checkpoint_path = model_checkpoint_map[model_name]
            model.load_weights(model_checkpoint_path, sess=sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET,
                          unit='imgs', ncols=100) as pbar:
                    while not coord.should_stop():
                        # Get attacked images for a batch
                        ids_, X_ben_, X_adv_, y_true_ = sess.run(
                            [ids, X_ben, X_adv, y_true])

                        # Update meter
                        normalized_l2_distance.offer(X_ben_, X_adv_)

                        # Save the attacked images
                        for example in encode_tf_examples(
                                ids_, X_adv_, y_true_):
                            writer.write(example.SerializeToString())

                        pbar.set_postfix(
                            average_normalized_l2=
                            normalized_l2_distance.evaluate())
                        pbar.update(len(ids_))

            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)

            finally:
                writer.close()

                normalized_l2_distance.save(os.path.join(
                    output_dir, NORMALIZED_L2_DISTANCE_NPZ_FILENAME))
