import math

import numpy as np
from scipy.stats import norm


class FlexibleNB:
    def __init__(self, var_smoothing):
        self.means = []
        self.stdevs = []
        self.class_probabilities = []
        self.fitted = False
        self.smoothing = var_smoothing

    def fit(self, x_train, y_train):
        """
        The y_train has to contain class indices, and the class indices
        have be in {0, ..., num_classes - 1} as they are used to index
        arrays.
        :param x_train:
        :param y_train:
        :return:
        """
        if self.fitted:
            raise Exception('This model has already been fitted')

        # separate samples by class
        num_of_classes = np.unique(y_train).shape[0]
        class_separated_x = [[] for _ in range(num_of_classes)]
        for x, y in zip(x_train, y_train):
            class_separated_x[y].append(x)

        self.__calculate_class_probabilities(class_separated_x, x_train.shape[0])
        self.__calculate_stdevs(class_separated_x)
        self.__calculate_means(class_separated_x)

        self.fitted = True
        return self

    def __calculate_class_probabilities(self, class_separated_x, num_samples):
        """
        Calculate the probability that a class will occur
        :param class_separated_x: list of lists of samples - first level is separation by class index
        :param num_samples: total number of samples
        :return:
        """
        for class_idx in range(len(class_separated_x)):
            self.class_probabilities.append(
                len(class_separated_x[class_idx]) / float(num_samples)
            )

    def __calculate_means(self, class_separated_x):
        """
        While classic gaussian NB calculates a mean for each class-feature pair,
        this can negatively impact the model when the features don't follow
        gaussian distribution. The flexible version tries to solve this by not
        limiting itself to a single mean - instead, it averages the probability
        estimates across all unique means present in the training data for the
        class-feature pair. This in turn means, that we have to (worst case) save
        the entirety of the training data (if no two values of a feature given a
        class are the same).
        This is slightly different approach from the original flexible NB, where
        the authors save all the samples by default (not eliminating duplicate values).
        This also prevents us from using ndarray to store the means
        :param class_separated_x:
        :return:
        """
        # We calculate different values for each class
        for clss in class_separated_x:
            samples = np.array(clss)
            class_unique_values = []

            # We need list of unique values for each feature
            for feature_idx in range(samples.shape[1]):
                feature_unique_values = []

                # So we go through the samples and collect all unique values of the feature
                for sample_idx in range(samples.shape[0]):
                    if samples[sample_idx][feature_idx] not in feature_unique_values:
                        feature_unique_values.append(samples[sample_idx][feature_idx])

                class_unique_values.append(feature_unique_values)

            self.means.append(class_unique_values)

    def __calculate_stdevs(self, class_separated_x):
        """
        Calculate standard deviations to be used in the gaussian PDF
        We use the same heuristic as the authors. While classic gaussian NB
        calculates the deviation from all the relevant samples (meaning for each
        class and feature), we use a universal deviation for each class
        which is inversely proportional to the number of samples in the class.
        This means that when we have more samples for the class, we make more
        strict decisions, while with few samples we are fairly lenient.
        :param class_separated_x:
        :return:
        """
        for clss in class_separated_x:
            self.stdevs.append(1 / math.sqrt(len(clss)))

    def predict_proba(self, x_test):
        """
        Returns likelihood estimates (np.exp of log probabilities)
        :param x_test:
        :return:
        """
        preds = []
        for sample_idx in range(x_test.shape[0]):
            preds.append(self.__classify_sample(x_test[sample_idx]))
        return np.array(preds)

    def __classify_sample(self, sample):
        """
        We need a prediction for each class
        """
        class_preds = []
        for class_idx in range(len(self.means)):
            class_preds.append(self.__class_prediction(sample, class_idx))

        return class_preds

    def __class_prediction(self, sample, class_idx):
        """
        This is the problematic stuff. We follow the standard NB formula, so we multiply
        the conditional probabilities for each feature and then multiply the result by the
        class probability. This often results in an underflow, so we move into the log space
        and add logarithms where we would multiply.

        We first calculate the probabilities for each feature and then log-sum them. Each
        feature probability is an average of multiple likelihood estimations given by a
        gaussian PDF - we make an estimation for each unique mean we have from training
        and then average them. This gives us the final estimate for the feature.
        :param sample:
        :param class_idx:
        :return:
        """
        # We would have multiplied by it at the end
        probability = np.log(self.class_probabilities[class_idx])

        # We need probabilities for all the features
        for feature_idx in range(sample.shape[0]):
            # We pass all the the means as an iterable, since scipy can deal with in and return
            # a vector of probabilities
            likelihoods = self.__calculate_gaussian(mean=self.means[class_idx][feature_idx],
                                                    std=self.stdevs[class_idx], x=sample[feature_idx])
            # We then average these probabilities
            feature_average_likelihood = np.mean(likelihoods)

            # Don't forget the log space
            probability += np.log(feature_average_likelihood + self.smoothing)

        return probability

    def __calculate_gaussian(self, mean, std, x):
        # We can use this trick, where we pass an array of means
        # and get an array of probabilities. The docs don't explicitly mention
        # it as a capability, but as the operations are performed with matrices,
        # we can do it. It has been checked for correctness.
        return norm.pdf(x=x, loc=mean, scale=std)
