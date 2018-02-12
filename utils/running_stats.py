from threading import RLock as _Lock

import numpy as np


def thread_safe(fn):
    def wrapper(*args, **kwargs):
        self = args[0]

        with self._lock:
            return_value = fn(*args, **kwargs)

        return return_value

    return wrapper


class RunningMetric(object):
    def __init__(self):
        self._lock = _Lock()

        self._data_items = {}
        self._reducers = {}

    def _add_data_item(self, label, initial_value, reducer):
        self._data_items[label] = initial_value
        self._reducers[label] = reducer

    @thread_safe
    def _offer(self, **kwargs):
        for label, new_value in kwargs.items():
            if label in self._data_items:
                current_value = self._data_items[label]

                self._data_items[label] = \
                    self._reducers[label](current_value, new_value)

    @thread_safe
    def _get_data_item(self, label):
        return self._data_items[label]

    @thread_safe
    def _get_data_items(self, labels):
        return tuple([self._get_data_item(label) for label in labels])

    def offer(self, **args):
        raise NotImplementedError

    def get(self):
        raise NotImplementedError

    def save(self, npz_save_path):
        npzfile = open(npz_save_path, 'w')
        np.savez(npzfile, **self._data_items)

    def load(self, npz_load_path):
        npzfile = np.load(npz_load_path)
        self._offer(**npzfile)

        return self.get()


class RunningAccuracy(RunningMetric):
    def __init__(self):
        super(RunningAccuracy, self).__init__()

        arr_reducer = lambda a, b: np.append(a, b)

        self._add_data_item('ids', np.zeros((0,), dtype=str), arr_reducer)
        self._add_data_item('y_true', np.zeros((0,)), arr_reducer)
        self._add_data_item('y_pred', np.zeros((0,)), arr_reducer)

    def offer(self, y_true, y_pred, ids=None):
        self._offer(
            ids=ids if ids is not None else np.zeros((0,), dtype=str),
            y_true=y_true,
            y_pred=y_pred)

    def get(self):
        y_true, y_pred = self._get_data_items(['y_true', 'y_pred'])
        correct, total = len(np.where(y_true == y_pred)[0]), y_true.shape[0]
        return correct / float(total)


class RunningAverageL2DistanceNormalized(RunningMetric):
    def __init__(self):
        super(RunningAverageL2DistanceNormalized, self).__init__()

        arr_reducer = lambda a, b: np.append(a, b)

        self._add_data_item(
            'original_L2', np.zeros((0,), dtype=np.float32), arr_reducer)
        self._add_data_item(
            'perturbation_L2', np.zeros((0,), dtype=np.float32), arr_reducer)

    def offer(self, original, perturbed):
        assert perturbed.shape == original.shape

        n = original.shape[0]
        original_, perturbed_ = original.reshape((n, -1,)),\
                                perturbed.reshape((n, -1,))

        self._offer(
            original_L2=np.linalg.norm(original_, ord=2, axis=1),
            perturbation_L2=np.linalg.norm(perturbed_ - original_, ord=2, axis=1))

    def get(self):
        original_L2, \
        perturbation_L2 = self._get_data_items(['original_L2', 'perturbation_L2'])

        return np.mean(perturbation_L2 / original_L2)


class RunningConfidenceScores(RunningMetric):
    def __init__(self):
        super(RunningConfidenceScores, self).__init__()

        arr_reducer = lambda a, b: np.append(a, b, axis=0)

        self._add_data_item('ids', np.zeros((0,), dtype=str), arr_reducer)
        self._add_data_item('y_true', np.zeros((0,)), arr_reducer)
        self._add_data_item('top_k_confidences', np.zeros((0, 5)), arr_reducer)
        self._add_data_item('top_k_preds', np.zeros((0, 5)), arr_reducer)

    def offer(self, y_true, top_k_confidences, top_k_preds, ids=None):
        self._offer(
            ids=ids if ids is not None else np.zeros((0,), dtype=str),
            y_true=y_true,
            top_k_confidences=top_k_confidences,
            top_k_preds=top_k_preds)

    def get(self):
        y_true, top_k_preds = self._get_data_items(['y_true', 'top_k_preds'])
        correct, total = len(np.where(y_true.reshape(-1, 1) == top_k_preds)[0]), y_true.shape[0]
        return correct / float(total)
