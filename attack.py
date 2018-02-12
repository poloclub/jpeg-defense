import os

import tensorflow as tf
from tqdm import tqdm

from constants import *
from params import *
from utils.running_stats import RunningAverageL2DistanceNormalized, RunningAccuracy
from utils.slim.preprocessing import inception_preprocessing
from utils.tf_io import load_image_data_from_tfrecords, get_examples_for_instances

slim = tf.contrib.slim


def attack_and_save_images(opt, debug=True, experiment_scope=None):
    Model = model_class_map[opt['model']]
    Attack = attack_class_map[opt['attack']]

    l2_distance = RunningAverageL2DistanceNormalized()

    preprocessing_fn = lambda x: inception_preprocessing.preprocess_image(
        x, Model.default_image_size, Model.default_image_size, is_training=False)

    experiment_identifier = '-'.join([
        'imagenet_val',
        opt['model'],
        opt['attack'],
        opt['attack_identifier']
    ])

    if not debug:
        writer = tf.python_io.TFRecordWriter(
            ADVERSARIAL_OUT_DIR + experiment_identifier + '.tfrecord')
    else:
        print 'WARN: Images not saved!'

    with tf.Graph().as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1., allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        with sess.as_default():
            ids, X_ben, y_true = load_image_data_from_tfrecords(
                VALIDATIONSET_RECORDS_EXPRESSION,
                preprocessing_fn=preprocessing_fn,
                load_jpeg=True,
                decode_pixels=True)

            model = Model(X_ben, model_checkpoint_map[opt['model']])
            attack = Attack(model)

            X_adv = attack.generate(X_ben, **opt['attack_params'])

            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            model.load_weights(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET, unit=' imgs', ncols=100) as pbar:
                    while not coord.should_stop():
                        ids_, X_ben_, X_adv_, y_true_ = sess.run([ids, X_ben, X_adv, y_true])

                        l2_distance.offer(X_ben_, X_adv_)

                        if not debug:
                            for example in get_examples_for_instances(ids_, X_adv_, y_true_):
                                writer.write(example.SerializeToString())

                        adv_batch_norm = np.mean(np.linalg.norm(X_adv_, ord='fro', axis=(0, 3)))
                        pbar.set_postfix(
                            average_l2_normalized=l2_distance.get(),
                            adv_batch_norm=adv_batch_norm)
                        pbar.update(len(ids_))
            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)
            finally:
                if not debug:
                    writer.close()

                    l2_distance.save(L2_OUT_DIR + experiment_identifier + '.npz')
