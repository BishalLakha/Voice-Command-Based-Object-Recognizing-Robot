import os
import sys
import argparse

import cPickle as pickle
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing


# to train the classifier
class ClassifierTrainer(object):
    def __init__(self, X, label_words):
        # encoding the labels (words to numbers)
        self.le = preprocessing.LabelEncoder()

        # initialize one vs one classifier using linear kernel
        self.clf = OneVsOneClassifier(LinearSVC(random_state=0))

        y = self._encodeLabels(label_words)
        X = np.asarray(X)
        self.clf.fit(X, y)

    # predict the output class for the input datapoint
    def _fit(self, X):
        X = np.asarray(X)
        return self.clf.predict(X)

    # encode the labels (convert words to normal)
    def _encodeLabels(self, labels_words):
        self.le.fit(labels_words)
        return np.array(self.le.transform(labels_words), dtype=np.float32)

    # classify the input datapoint
    def classify(self, X):
        labels_nums = self._fit(X)
        labels_words = self.le.inverse_transform([int(x) for x in labels_nums])
        return labels_words

def start():
    feature_map_file = "feature_map.pkl"
    svm_file = "svm.pkl"

    # load the feature map
    with open(feature_map_file, 'rb') as f:
        feature_map = pickle.load(f)

    # extract feature vectors and the labels
    labels_words = [x['label'] for x in feature_map]

    # here, 0 refers to the first element in the
    # feature_map, and 1 refers to the second
    # element in the shape vector of that element
    # (which gives us the size)
    dim_size = feature_map[0]['feature_vector'].shape[1]

    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]

    # train the svm
    svm = ClassifierTrainer(X, labels_words)
    if svm_file:
        with open(svm_file, 'wb') as f:
            pickle.dump(svm, f)

if __name__ == '__main__':
    start()
