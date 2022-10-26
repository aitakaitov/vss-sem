import math

import numpy as np
from scipy.stats import norm


class FlexibleNB:
    def __init__(self):
        # What do we need:

        #   - for each class and feature the different values the feature had in the training data (means_icl)
        #   - for each class and feature the number of different values the feature had in the training data (L_ic)
        self.means = []

        #   - for each class and feature the standard deviation the feature had in the training data (stdev_ic)
        #   - shape (num_classes, num_features)
        self.stdevs = []

        self.class_probabilities = []

        self.fitted = False

    def fit(self, x_train, y_train):
        if self.fitted:
            raise Exception('This model has already been fitted')

        num_of_classes = np.unique(y_train).shape[0]
        class_separated_x = [[] for _ in range(num_of_classes)]
        class_separated_y = [[] for _ in range(num_of_classes)]
        for x, y in zip(x_train, y_train):
            class_separated_y[y].append(y)
            class_separated_x[y].append(x)

        for class_idx in range(num_of_classes):
            self.class_probabilities.append(
                len(class_separated_x[class_idx]) / float(x_train.shape[0])
            )

        self.__calculate_stdevs(class_separated_x)
        self.__calculate_means(class_separated_x)

        self.fitted = True
        return self

    def __calculate_means(self, class_separated_x):
        for clss in class_separated_x:
            samples = np.array(clss)
            class_means = []
            for feature_idx in range(samples.shape[1]):
                feature_means = []
                for sample_idx in range(samples.shape[0]):
                    if samples[sample_idx][feature_idx] not in feature_means:
                        feature_means.append(samples[sample_idx][feature_idx])
                class_means.append(feature_means)
            self.means.append(class_means)

    def __calculate_stdevs(self, class_separated_x):
        for clss in class_separated_x:
            self.stdevs.append(1 / math.sqrt(len(clss)))


    def predict_proba(self, x_test):
        """
        Does not return probabilities but likelihood estimates
        despite the name. This is done to unify the interface with sklearn models.
        :param x_test:
        :return:
        """
        preds = []
        for sample_idx in range(x_test.shape[0]):
            preds.append(self.__classify_sample(x_test[sample_idx]))
        return np.array(preds)

    def __classify_sample(self, sample):
        # Go over all classes and calculate likelihood estimate for each of them
        class_preds = []
        for class_idx in range(len(self.means)):
            class_preds.append(self.__class_prediction(sample, class_idx))

        return class_preds

    def __class_prediction(self, sample, class_idx):
        # Estimate the likelihood of a single class
        # With FNB we do this using normal probability distribution as in GNB, but
        # we average the probabilities over all the unique means (samples) of the
        # feature from the training data. This way we allow for some compensation
        # if the variable does not follow normal distribution
        # That's why this looks so weird
        probability = 0.0
        for feature_idx in range(sample.shape[0]):
            sum_of_probs = 0.0
            for feature_mean in self.means[class_idx][feature_idx]:
                sum_of_probs += self.__calculate_gaussian(feature_mean, self.stdevs[class_idx],
                                                          sample[feature_idx])
            sum_of_probs /= len(self.means[class_idx][feature_idx])
            probability += sum_of_probs

        # log space to avoid underflows
        return np.log(self.class_probabilities[class_idx]) + np.log(probability)

    def __calculate_gaussian(self, mean, std, feature_value):
        return norm.pdf(x=feature_value, loc=mean, scale=std)
