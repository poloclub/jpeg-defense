import tensorflow as tf
from tqdm import tqdm

from constants import *
from params import *
from utils.running_stats import RunningAccuracy
from utils.slim.preprocessing import inception_preprocessing
from utils.tf_io import load_image_data_from_tfrecords, get_examples_for_instances


_TF_DEFENSES = ['jpeg', 'slq_tf']


def defend_and_save_images(opt, decode_pixels=False, debug=True, experiment_scope=None):
    Model = model_class_map[opt['model']]

    defense_fn = defense_fn_map[opt['defense']]

    preprocessing_fn = (lambda x: inception_preprocessing.preprocess_image(
            x, Model.default_image_size, Model.default_image_size, is_training=False)) \
        if decode_pixels \
        else (lambda x: x)

    identifier_prefix = '-'.join([
        'imagenet_val',
        opt['model'],
        opt['attack'],
        opt['attack_identifier']])

    experiment_identifier = '-'.join([
        identifier_prefix,
        opt['defense'],
        opt['defense_identifier']])

    if not debug:
        writer = tf.python_io.TFRecordWriter(
            PREPROCESSED_OUT_DIR + experiment_identifier +'.tfrecord')
    else:
        print 'WARN: Images not saved!'

    with tf.Graph().as_default():
        ids, images, labels = load_image_data_from_tfrecords(
            ADVERSARIAL_OUT_DIR + identifier_prefix+'.tfrecord',
            preprocessing_fn=preprocessing_fn,
            decode_pixels=decode_pixels,
            image_size=Model.default_image_size)

        if not decode_pixels:
            images = tf.cast((255 * (images + 1) / 2), tf.uint8)

        if opt['defense'] in _TF_DEFENSES:
            images = tf.map_fn(
                lambda x: defense_fn(x, **opt['defense_params']), images)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1., allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        with sess.as_default():
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET, unit=' imgs', ncols=100) as pbar:
                    while not coord.should_stop():
                        ids_, images_, labels_ = sess.run([ids, images, labels])

                        if opt['defense'] not in _TF_DEFENSES:
                            images_ = np.array([defense_fn(image_, **opt['defense_params']) for image_ in images_])

                        if not debug:
                            for example in get_examples_for_instances(ids_, images_, labels_):
                                writer.write(example.SerializeToString())

                        pbar.update(len(ids_))
            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)
            finally:
                if not debug:
                    writer.close()
