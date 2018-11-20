from threading import RLock as _Lock

import numpy as np


def _thread_safe(fn):
    """Decorator that expects a class method which has a _lock attribute.

    Args:
        fn: Method to be wrapped.

    Returns:
        method: The wrapper method
    """

    def wrapper(*args, **kwargs):
        self = args[0]

        with self._lock:
            return_value = fn(*args, **kwargs)

        return return_value

    return wrapper


class Meter(object):
    """Base class for creating thread-safe meters.

    A meter accepts parameter values iteratively and
    computes some metric from these values, e.g.,
    the AccuracyMeter accepts batches of true and predicted
    labels iteratively, and evaluates the accuracy
    over all batches.

    Parameter values can be iteratively offered
    to the meter, and the meter uses the corresponding
    private reducer methods to store the values
    in the original form, or in some processed form, e.g.,
    the reducer in AccuracyMeter takes the new batches of labels
    and appends them to the list of all labels.

    The meter is thread-safe, i.e., it can be used in
    parallel threads. The parameter values offered in
    one call are reduced by the reducer as an atomic unit.
    This functionality is handled by the base class,
    and the child classes need to only implement the
    abstract functions.
    """

    def __init__(self):
        self._lock = _Lock()

        self._parameters = {}
        self._reducers = {}

    def _add_parameter(self, name, initial_value, reducer):
        self._parameters[name] = initial_value
        self._reducers[name] = reducer

    @_thread_safe
    def _update_parameters(self, **kwargs):
        for name, new_value in kwargs.items():
            assert name in self._parameters

            current_value = self._parameters[name]
            reducer = self._reducers[name]

            self._parameters[name] = \
                reducer(current_value, new_value)

    @_thread_safe
    def _get_parameter(self, name):
        return self._parameters[name]

    @_thread_safe
    def _get_parameters(self, names):
        return tuple([self._get_parameter(name) for name in names])

    def offer(self, **kwargs):
        """Abstract function that accepts parameter values
        to be processed and saved by the meter."""

        raise NotImplementedError

    def evaluate(self):
        """Abstract function that processes the parameters
        and returns the current meter value."""

        raise NotImplementedError

    def save(self, npz_save_path):
        """Saves the meter parameters to a .npz file.

        Args:
            npz_save_path (str): Path to .npz file
                where parameters will be saved.
        """

        with open(npz_save_path, 'wb') as npzfile:
            np.savez(npzfile, **self._parameters)

    def load(self, npz_load_path):
        """Loads meter parameters from a .npz file.

        Args:
            npz_load_path (str): Path to .npz file
                where parameters will be loaded from.

        Returns:
            The meter value after loading the data.
        """

        npzfile = np.load(npz_load_path)
        self._update_parameters(**npzfile)

        return self.evaluate()


class AccuracyMeter(Meter):
    """A meter to track accuracy of the model."""

    def __init__(self):
        """Initializes the meter.

        This meter tracks the following parameters:
            - ids
            - y_pred
            - y_true
        """
        super(AccuracyMeter, self).__init__()

        arr_reducer = lambda a, b: np.append(a, b)

        self._add_parameter('ids', np.zeros((0,), dtype=str), arr_reducer)
        self._add_parameter('y_pred', np.zeros((0,)), arr_reducer)
        self._add_parameter('y_true', np.zeros((0,)), arr_reducer)

    def offer(self, y_pred, y_true, ids=None):
        """Tracks the corresponding parameter values.

        Args:
            y_pred (np.ndarray): A 1-D list of predicted class labels.
            y_true (np.ndarray): A 1-D list of true class labels.
            ids (np.ndarray, optional):  A 1-D list of instance ID's.
        """

        y_pred, y_true = np.array(y_pred), np.array(y_true)
        assert y_pred.shape[0] == y_true.shape[0]
        n = y_true.shape[0]

        ids = (ids if ids is not None
               else np.zeros((n,), dtype=str))

        self._update_parameters(
            ids=ids, y_true=y_true, y_pred=y_pred)

    def evaluate(self):
        """Calculates and returns the current accuracy tracked by the meter.

        Returns:
            float: Current accuracy.
        """

        y_pred, y_true = self._get_parameters(['y_pred', 'y_true'])
        correct, total = len(np.where(y_true == y_pred)[0]), y_true.shape[0]
        return correct / float(total)


class TopKAccuracyMeter(Meter):
    """A meter to track top-k accuracy of the model."""

    def __init__(self, k):
        """Initializes the meter.

        This meter tracks the following parameters:
            - ids
            - top_k_preds
            - y_true

        Args:
            k (int): Top k accuracies to be tracked.
        """

        super(TopKAccuracyMeter, self).__init__()

        self._k = k

        arr_reducer = lambda a, b: np.append(a, b, axis=0)

        self._add_parameter('ids', np.zeros((0,), dtype=str), arr_reducer)
        self._add_parameter('top_k_preds', np.zeros((0, k)), arr_reducer)
        self._add_parameter('y_true', np.zeros((0,)), arr_reducer)

    def offer(self, top_k_preds, y_true, ids=None):
        """Tracks the corresponding parameter values.

        Args:
            top_k_preds (np.ndarray): A matrix of shape (n, k)
                of predicted class labels.
            y_true (np.ndarray): A 1-D list of true class labels.
            ids (np.ndarray, optional):  A 1-D list of instance ID's.
        """

        y_true = np.array(y_true)
        assert top_k_preds.shape[0] == y_true.shape[0]
        n = y_true.shape[0]

        ids = (ids if ids is not None
               else np.zeros((n,), dtype=str))
        top_k_preds = top_k_preds[:, :self._k]

        self._update_parameters(
            ids=ids, top_k_preds=top_k_preds, y_true=y_true)

    def evaluate(self):
        """Calculates and returns the current top-k accuracy
        tracked by the meter.

        Returns:
            float: Current top-k accuracy.
        """

        top_k_preds, y_true = self._get_parameters(['top_k_preds', 'y_true'])
        correct, total = \
            len(np.where(y_true.reshape(-1, 1) == top_k_preds)[0]), \
            y_true.shape[0]
        return correct / float(total)


class AverageNormalizedL2DistanceMeter(Meter):
    """A meter to track the average normalized L2 distance
    between original and perturbed images."""

    def __init__(self):
        """Initializes the meter.

        This meter tracks the following parameters:
            - original_L2
            - perturbation_L2
        """

        super(AverageNormalizedL2DistanceMeter, self).__init__()

        arr_reducer = lambda a, b: np.append(a, b)

        self._add_parameter(
            'original_L2', np.zeros((0,), dtype=np.float32), arr_reducer)
        self._add_parameter(
            'perturbation_L2', np.zeros((0,), dtype=np.float32), arr_reducer)

    def offer(self, original, perturbed):
        """Tracks the corresponding parameter values.

        Args:
            original (np.ndarray): A matrix of shape (n, d1, d2, ..., dm)
                for the original batch of images.
            perturbed (np.ndarray): A matrix of shape (n, d1, d2, ..., dm)
                for the perturbed batch of images.
        """

        assert perturbed.shape == original.shape

        n = original.shape[0]
        original_, perturbed_ = original.reshape((n, -1,)), \
                                perturbed.reshape((n, -1,))

        self._update_parameters(
            original_L2=np.linalg.norm(
                original_,
                ord=2, axis=1),
            perturbation_L2=np.linalg.norm(
                perturbed_ - original_,
                ord=2, axis=1))

    def evaluate(self):
        """Calculates and returns the current average normalized L2 distance
        tracked by the meter.

        Returns:
            float: Current average normalized L2 distance.
        """

        original_L2, perturbation_L2 = \
            self._get_parameters(['original_L2', 'perturbation_L2'])

        return np.mean(perturbation_L2 / original_L2)
