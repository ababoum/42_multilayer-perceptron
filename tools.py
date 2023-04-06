import numpy as np


def is_matrix_valid(x):
    if not isinstance(x, np.ndarray):
        return False
    if len(x.shape) == 1 and x.shape[0] < 1:
        return False
    if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] < 1):
        return False
    if x.size == 0:
        return False
    return True


def is_vector_valid(x):
    if not isinstance(x, np.ndarray):
        return False
    if len(x.shape) == 1 and x.shape[0] < 1:
        return False
    if len(x.shape) == 2 and (x.shape[0] < 1 or x.shape[1] != 1):
        return False
    if x.size == 0:
        return False
    return True


def data_splitter(x, y, proportion):
    """Shuffles and splits the dataset (given by x and y) into a training and a test set,
    while respecting the given proportion of examples to be kept in the training set.
    Args:
    x: has to be an numpy.array, a matrix of dimension m * n.
    y: has to be an numpy.array, a vector of dimension m * 1.
    proportion: has to be a float, the proportion of the dataset that will be assigned to the
    training set.
    Return:
    (x_train, x_test, y_train, y_test) as a tuple of numpy.array
    None if x or y is an empty numpy.array.
    None if x and y do not share compatible dimensions.
    None if x, y or proportion is not of expected type.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not is_matrix_valid(x) or not is_matrix_valid(y):
            return None
        if x.shape[0] != y.shape[0]:
            return None
        if not isinstance(proportion, float):
            return None

        # Shuffle the data
        data = np.concatenate([x, y], axis=1)
        np.random.shuffle(data)
        x = data[:, :-1]
        y = data[:, -1].reshape((-1, 1))

        # Split the data
        split = int(x.shape[0] * proportion)
        return (x[:split], x[split:], y[:split], y[split:])

    except:
        return None
