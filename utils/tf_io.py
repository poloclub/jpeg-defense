import glob

import tensorflow as tf


def encode_example_for_instance(image_id, image, label):
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


def decode_features_from_example(serialized_example):
    feature_set = {
        'image/filename': tf.FixedLenFeature([], tf.string),
        'image/encoded': tf.FixedLenFeature([], tf.string),
        'image/height': tf.FixedLenFeature([], tf.int64),
        'image/width': tf.FixedLenFeature([], tf.int64),
        'image/class/label': tf.FixedLenFeature([], tf.int64)
    }

    return tf.parse_single_example(serialized_example, features=feature_set)


def get_examples_for_instances(image_ids, images, labels):
    n = images.shape[0]

    assert len(image_ids) == labels.shape[0] == n

    for i in range(n):
        yield encode_example_for_instance(image_ids[i], images[i], labels[i])


def load_image_data_from_tfrecords(
        tfrecord_paths_expression,
        preprocessing_fn=lambda x: x,
        load_jpeg=False,
        decode_pixels=False,
        image_size=None,
        batch_size=16,
        num_preprocessing_threads=4):
    assert type(tfrecord_paths_expression) in [str, list]
    print 'reading', tfrecord_paths_expression
    filename_queue = tf.train.string_input_producer(
        glob.glob(tfrecord_paths_expression) \
            if type(tfrecord_paths_expression) is str \
            else tfrecord_paths_expression,
        num_epochs=1)

    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = decode_features_from_example(serialized_example)

    image_id = features['image/filename']

    image = tf.image.decode_jpeg(features['image/encoded'], channels=3) \
        if load_jpeg \
        else tf.decode_raw(
            features['image/encoded'],
            tf.uint8 if decode_pixels else tf.float32)

    height, width = tf.cast(features['image/height'], tf.int32), \
                    tf.cast(features['image/width'], tf.int32)

    image = tf.reshape(image, [height, width, 3])

    if not load_jpeg and image_size is not None:
        image = tf.reshape(image, [image_size, image_size, 3])

    image = preprocessing_fn(image)

    label = features['image/class/label']

    image_ids, images, labels = tf.train.batch([image_id, image, label],
                                                batch_size=batch_size,
                                                num_threads=num_preprocessing_threads,
                                                capacity=5 * batch_size,
                                                allow_smaller_final_batch=True)

    return image_ids, images, labels
