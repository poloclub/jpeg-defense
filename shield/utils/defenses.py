import numpy as np
from PIL import Image as Image
from scipy.ndimage import median_filter as _median_filter
from skimage.restoration import denoise_tv_bregman as _denoise_tv_bregman
import tensorflow as tf


def _get_image_from_arr(img_arr):
    return Image.fromarray(
        np.asarray(img_arr, dtype='uint8'))


def median_filter(img_arr, size=3):
    return _median_filter(img_arr, size=size)


def denoise_tv_bregman(img_arr, weight=30):
    denoised = _denoise_tv_bregman(img_arr, weight=weight) * 255.
    return np.array(denoised, dtype=img_arr.dtype)


def jpeg_compress(x, quality=75):
    return tf.image.decode_jpeg(
        tf.image.encode_jpeg(
            x, format='rgb', quality=quality),
        channels=3)


def slq(x, qualities=(20, 40, 60, 80), patch_size=8):
    num_qualities = len(qualities)

    with tf.name_scope('slq'):
        one = tf.constant(1, name='one')
        zero = tf.constant(0, name='zero')

        x_shape = tf.shape(x)
        n, m = x_shape[0], x_shape[1]

        patch_n = tf.cast(n / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)
        patch_m = tf.cast(m / patch_size, dtype=tf.int32) \
            + tf.cond(n % patch_size > 0, lambda: one, lambda: zero)

        R = tf.tile(tf.reshape(tf.range(n), (n, 1)), [1, m])
        C = tf.reshape(tf.tile(tf.range(m), [n]), (n, m))
        Z = tf.image.resize_nearest_neighbor(
            [tf.random_uniform(
                (patch_n, patch_m, 3),
                0, num_qualities, dtype=tf.int32)],
            (patch_n * patch_size, patch_m * patch_size),
            name='random_layer_indices')[0, :, :, 0][:n, :m]
        indices = tf.transpose(
            tf.stack([Z, R, C]),
            perm=[1, 2, 0],
            name='random_layer_indices')

        x_compressed_stack = tf.stack(
            list(map(
                lambda q: tf.image.decode_jpeg(tf.image.encode_jpeg(
                    x, format='rgb', quality=q), channels=3),
                qualities)),
            name='compressed_images')

        x_slq = tf.gather_nd(x_compressed_stack, indices, name='final_image')

    return x_slq
