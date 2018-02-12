import numpy as _np


def get_mode(labels=None):
    """
    Determine the mode of labels predicted by models of an ensemble

    :param labels: ndarray of shape (N, K), N is the number of samples, K is th number of votes
    :return: ndarray of shape (N, 1)
    """

    n = labels.shape[0]

    mode = []

    for i in range(n):
        v = _np.bincount(labels[i, :].astype(int))

        mode.append(v.argmax())

    return _np.array(mode, dtype=labels.dtype)


def get_mode_of_random_votes(labels=None, m=1):
    n, k = labels.shape

    return get_mode(
        labels=labels[:, _np.random.randint(k, size=m)])
