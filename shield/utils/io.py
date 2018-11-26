from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import glob

import tensorflow as tf


def encode_tf_example(image_id, image, label):
    """Encodes raw data into a serialized tensorflow example.
    This format is good for storing the data for an instance as an atomic unit.

    Args:
        image_id (str): ID of the image.
        image (np.ndarray): Raw image pixels, can be in
            pixel space [0, 255] ints or normalized [-1, 1] floats.
        label (int): Class label for the image.

    Returns:
        tf.train.Example: The serialized tensorflow example.
    """

    def _bytes_feature(value):
        return tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[value]))

    def _int64_feature(value):
        return tf.train.Feature(
            int64_list=tf.train.Int64List(value=[value]))

    h, w, c = image.shape

    return tf.train.Example(
        features=tf.train.Features(feature={
            'image/filename': _bytes_feature(image_id),
            'image/encoded': _bytes_feature(image.tostring()),
            'image/height': _int64_feature(h),
            'image/width': _int64_feature(w),
            'image/class/label': _int64_feature(label)}))


def encode_tf_examples(image_ids, images, labels):
    """Encodes multiple instances into serialized tensorflow examples.

    Args:
        image_ids (list): List of image IDs.
        images (images): List of images.
        labels (list): List of class labels corresponding to the IDs.

    Returns:
        generator: Generates serialized examples.
    """

    n = images.shape[0]
    assert len(image_ids) == labels.shape[0] == n

    for i in range(n):
        yield encode_tf_example(image_ids[i], images[i], labels[i])


def decode_tf_example(serialized_example):
    """Parses and decodes a serialized tensorflow example.

    Args:
        serialized_example (tf.train.Example):
            A single serialized tensorflow example.

    Returns:
        dict: A dictionary mapping feature keys to tensors.
    """

    feature_set = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    }

    return tf.parse_single_example(serialized_example, features=feature_set)


def load_image_data_from_tfrecords(
        tfrecord_paths_expression,
        preprocessing_fn=lambda x: x,
        load_jpeg=False,
        decode_pixels=False,
        image_size=None,
        batch_size=20,
        num_preprocessing_threads=4):
    """Loads image data from tfrecord files.

    Args:
        tfrecord_paths_expression (str):
            Wildcard expression for path to the
            tfrecord files, e.g., "/path/to/*.tfrecord".
        preprocessing_fn (lambda):
            A lambda function that
            is applied to each image batch
        load_jpeg (bool):
            True if the tfrecords contain binary JPEG data.
        decode_pixels (bool):
            Whether the tfrecord contains image data
            in pixel space or contains normalized values.
        image_size (int):
            If not None, all images are resized
            to a square of this size.
        batch_size (int):
            Size of a minibatch.
        num_preprocessing_threads (int):
            Number of threads that run in parallel
            to pre-load the data.

    Returns:
        tuple: Image IDs, images and labels as batched tensors.
    """

    print('reading %s' % tfrecord_paths_expression)
    filename_queue = tf.train.string_input_producer(
        glob.glob(tfrecord_paths_expression), num_epochs=1)

    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = decode_tf_example(serialized_example)

    image_id = features['image/filename']
    label = features['image/class/label']
    height, width = tf.cast(features['image/height'], tf.int32), \
                    tf.cast(features['image/width'], tf.int32)

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3) \
        if load_jpeg \
        else tf.decode_raw(
            features['image/encoded'],
            tf.uint8 if decode_pixels else tf.float32)

    image = tf.reshape(image, [height, width, 3])
    if not load_jpeg and image_size is not None:
        image = tf.reshape(image, [image_size, image_size, 3])

    image = preprocessing_fn(image)

    image_ids, images, labels = tf.train.batch(
        [image_id, image, label],
        batch_size=batch_size,
        num_threads=num_preprocessing_threads,
        capacity=5 * batch_size,
        allow_smaller_final_batch=True)

    return image_ids, images, labels
