from dataset_utils import MDPDataset
import random
from sklearn.model_selection import StratifiedKFold
import numpy as np

RANDOM_SEED = 42
random.seed(RANDOM_SEED)


def train_evaluate_model_kfold(model_class, kwargs: dict, dataset: MDPDataset, k=10):
    """
    Performs K-fold training and evaluation
    :param model_class: model class
    :param kwargs: keyword arguments for the model
    :param dataset: dataset
    :param k: number of folds
    :return: y_pred, y_true
    """
    samples, labels = dataset.get_all_data()
    folds = generate_folds(samples, labels, k)
    samples, labels = np.array(samples), np.array(labels)


    y_true = []
    y_pred = []
    for train_indices, test_indices in folds:
        x_train, x_test = samples[train_indices], samples[test_indices]
        y_train, y_test = labels[train_indices], labels[test_indices]

        # save the true labels and predictions, we need to compute metrics on them later
        y_true.extend(list(y_test))
        y_pred.extend(
            train_evaluate_fold(model_class, kwargs, x_train, y_train, x_test)
        )

    return y_pred, y_true


def train_evaluate_fold(model_class, kwargs: dict, x_train, y_train, x_test) -> list:
    model = model_class(**kwargs)
    y_pred = model.fit(x_train, y_train).predict_proba(x_test)
    return y_pred


def generate_folds(samples, labels, k):
    # We use stratified K-fold to deal with class imbalance
    # The authors do not specify how they evaluated models other than the
    # best model
    # I would assume all models were evaluated with K-fold since
    # some of the datasets have few samples
    kfold = StratifiedKFold(n_splits=k, random_state=RANDOM_SEED)
    split_folds = kfold.split(X=samples, y=labels)
    return split_folds
