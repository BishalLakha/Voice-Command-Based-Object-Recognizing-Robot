import os
import sys
import cPickle as pickle

import cv2
import numpy as np

import create_features as cf
from training import ClassifierTrainer


# classifying an image
class ImageClassifier(object):
    def __init__(self, svm_file, codecook_file):
        # load SVM classifier
        with open(svm_file, 'r') as f:
            self.svm = pickle.load(f)

        # load the codebook
        with open(codecook_file, 'r') as f:
            self.kmeans, self.centroids = pickle.load(f)

    # method to get the output image tag
    def getImageTag(self, img):
        # resize the input image
        img = cf.resize_to_size(img)

        # extract the feature vector
        feature_vector = cf.FeatureExtractor().get_feature_vector(img, self.kmeans, self.centroids)

        # classify the feature vector and get the output tag
        image_tag = self.svm.classify(feature_vector)

        return image_tag


if __name__ == '__main__':
    svm_file = "svm.pkl"
    codebook_file = "codebook.pkl"

    input_image = cv2.imread("test.jpg")

    print "Output class: ", ImageClassifier(svm_file, codebook_file).getImageTag(input_image)
