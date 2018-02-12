from collections import defaultdict
import os
import threading

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from constants import *
from utils.ensemble import get_mode, get_mode_of_random_votes
from params import *
from utils.running_stats import RunningConfidenceScores
from utils.slim.preprocessing import inception_preprocessing
from utils.tf_io import load_image_data_from_tfrecords


class RunningAccuracy:
    def __init__(self):
        self.lock = threading.Lock()

        self.reset()

    def reset(self):
        self.ids = np.zeros((0), dtype=str)
        self.y_true = np.zeros((0))
        self.y_pred = np.zeros((0))

        self.correct = 0.
        self.total = 0.

    def offer(self, y_true, y_pred, ids=None):
        with self.lock:
            if ids is not None:
                self.ids = np.append(self.ids, ids)

            self.y_true = np.append(self.y_true, y_true)
            self.y_pred = np.append(self.y_pred, y_pred)

            self.total += y_true.shape[0]
            self.correct += len(np.where(y_true == y_pred)[0])

    def get(self):
        with self.lock:
            acc = self.correct / float(self.total)

        return acc

    def save(self, npz_save_path):
        npzfile = open(npz_save_path, 'w')
        np.savez(npzfile, ids=self.ids, y_true=self.y_true, y_pred=self.y_pred)

    def load(self, npz_load_path):
        npzfile = np.load(npz_load_path)
        self.reset()
        self.offer(
            npzfile['y_true'],
            npzfile['y_pred'],
            ids=npzfile['ids'] if 'ids' in npzfile else np.zeros((0), dtype=str))

        return self.get()


def evaluate_record(
        tfrecord_paths_expression, model_name,
        model_checkpoint_path=None,
        load_jpeg=False, decode_pixels=False,
        eval_top_5=False,
        save_eval_to_path=None,
        debug=True, experiment_scope=None):
    Model = model_class_map[model_name]

    accuracy = RunningAccuracy()
    confidences = RunningConfidenceScores()

    preprocessing_fn = (lambda x: inception_preprocessing.preprocess_image(
            x, Model.default_image_size, Model.default_image_size, is_training=False)) \
        if decode_pixels \
        else (lambda x: x)

    experiment_identifier = 'evaluate'
    if save_eval_to_path is not None:
        experiment_identifier = save_eval_to_path.split('/')[-1].replace('.npz', '')

    with tf.Graph().as_default():
        ids, X, y_true = load_image_data_from_tfrecords(
            tfrecord_paths_expression,
            preprocessing_fn=preprocessing_fn,
            load_jpeg=load_jpeg,
            decode_pixels=decode_pixels,
            image_size=Model.default_image_size)

        model = Model(X, model_checkpoint_map[model_name] if model_checkpoint_path is None else model_checkpoint_path)
        y_pred = tf.argmax(model.fprop(X)['probs'], 1)
        top_k_confidences, top_k_preds = tf.nn.top_k(model.fprop(X)['probs'], k=5)

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1., allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options))
        with sess.as_default():
            tf.local_variables_initializer().run()
            tf.global_variables_initializer().run()

            model.load_weights(sess)

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)

            try:
                with tqdm(total=NUM_SAMPLES_VALIDATIONSET, unit=' imgs', ncols=100) as pbar:
                    while not coord.should_stop():
                        ids_, X_, y_true_, y_pred_, top_k_confidences_, top_k_preds_ = sess.run(
                            [ids, X, y_true, y_pred, top_k_confidences, top_k_preds])

                        accuracy.offer(y_true_, y_pred_, ids=ids_)
                        confidences.offer(y_true_, top_k_confidences_, top_k_preds_, ids=ids_)

                        pbar.set_postfix(top_1_accuracy=accuracy.get(), top_5_accuracy=confidences.get())
                        pbar.update(y_true_.shape[0])
            except tf.errors.OutOfRangeError:
                coord.request_stop()
                coord.join(threads)
            finally:
                if save_eval_to_path is not None and not debug:
                    accuracy.save(save_eval_to_path)
                    confidences.save(save_eval_to_path.replace('eval', 'top_k'))

                if experiment_scope is not None:
                    notify('%s DONE (%0.4f)' % (experiment_identifier, accuracy.get()), scope=experiment_scope)

    return accuracy


def evaluate_ensemble(model_name, attack, attack_ablation, defense='jpeg', num_random_judges=None,
        debug=True, experiment_scope=None):
    get_labels_sorted_by_ids = \
        lambda labels, ids: map(lambda t: t[1], sorted(enumerate(labels), key=lambda t: ids[t[0]]))

    identifier_prefix1 = '-'.join([
        'imagenet_val',
        model_name,
        attack,
        attack_identifiers[attack](attack_ablation)])
    identifier_prefix2 = '-'.join([
        identifier_prefix1,
        defense])
    experiment_identifier = '-'.join([
        identifier_prefix2,
        'ensemble'])

    print experiment_identifier

    preds_by_ids = defaultdict(list)
    labels_by_ids = defaultdict(lambda: None)

    for i, defense_identifier in enumerate(
            map(lambda abln: defense_identifiers[defense](abln), defense_ablations[defense])):
        acc = RunningAccuracy()
        acc.load(EVAL_OUT_DIR + '-'.join([identifier_prefix2, defense_identifier]) + '.npz')

        print acc.get()

        for j, id_ in enumerate(acc.ids):
            preds_by_ids[id_].append(acc.y_pred[j])

            if labels_by_ids[id_] is None:
                labels_by_ids[id_] = acc.y_true[j]
            else:
                assert labels_by_ids[id_] == acc.y_true[j]

    n = len(preds_by_ids)
    k = len(defense_ablations[defense])

    ids = np.zeros((0,), dtype=str)
    y_true = np.zeros((0,))
    Y_pred = np.zeros((0, k),)

    for id_ in preds_by_ids:
        if len(preds_by_ids[id_]) == k:
            ids = np.append(ids, id_)
            y_true = np.append(y_true, labels_by_ids[id_])
            Y_pred = np.append(Y_pred, np.array(preds_by_ids[id_]).reshape((1, -1)), axis=0)

    print Y_pred.shape

    y_pred = get_mode(labels=Y_pred) \
        if num_random_judges is None \
        else get_mode_of_random_votes(labels=Y_pred, m=num_random_judges)

    ensemble_accuracy = RunningAccuracy()
    ensemble_accuracy.offer(y_true=y_true, y_pred=y_pred, ids=ids)
    print '*', ensemble_accuracy.get()
    print '~~~'

    if not debug:
        ensemble_accuracy.save(EVAL_OUT_DIR + experiment_identifier + '.npz')

    return ensemble_accuracy
