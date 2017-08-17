import os
import sys
import argparse
import cPickle as pickle
import json
import time

import cv2
import numpy as np
from sklearn.cluster import KMeans


# load images from input folder
def load_input_map(label, input_folder):
    combined_data = []
    if not os.path.isdir(input_folder):
        raise IOError("The folder " + input_folder + " does not exist")

    # parse input folder and assign the labels
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})

    return combined_data


class FeatureExtractor(object):
    def extract_image_features(self, img):
        # SURF feature extractor
        kps, fvs = SURFExtractor().compute(img)
        return fvs

    # Extract the centroids from the feature points
    def get_centroids(self, input_map, num_samples_to_fit=10):
        kps_all = []

        count = 0
        cur_label = ''
        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_label != item['label']:
                    count = 0
                else:
                    continue

            count += 1

            if count == num_samples_to_fit:
                print "Built centroids for", item['label']

            cur_label = item['label']
            img = cv2.imread(item['image'])
            img = resize_to_size(img, 150)

            num_dims = 64
            fvs = self.extract_image_features(img)
            kps_all.extend(fvs)

        kmeans, centroids = Quantizer().quantize(kps_all)
        return kmeans, centroids

    def get_feature_vector(self, img, kmeans, centroids):
        return Quantizer().get_feature_vector(img, kmeans, centroids)


def extract_feature_map(input_map, kmeans, centroids):
    feature_map = []

    for item in input_map:
        temp_dict = {}
        temp_dict['label'] = item['label']

        print "Extracting features for", item['image']
        img = cv2.imread(item['image'])
        img = resize_to_size(img, 150)

        temp_dict['feature_vector'] = FeatureExtractor().get_feature_vector(img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map


# Vector quantization
class Quantizer(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 64
        self.extractor = SURFExtractor()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def quantize(self, datapoints):
        # Create KMeans object
        kmeans = KMeans(self.num_clusters, n_init=max(self.num_retries, 1), max_iter=10, tol=1.0)

        # run KMeans on the datapoints
        res = kmeans.fit(datapoints)

        # Extract the centroids of those clusters
        centroids = res.cluster_centers_

        return kmeans, centroids

    def normalize(self, input_data):
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    # extract feature vector from the image
    def get_feature_vector(self, img, kmeans, centroids):
        kps, fvs = self.extractor.compute(img)
        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)

        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalize(fv_image)


class SURFExtractor(object):
    def compute(self, image):
        if image is None:
            print "Not a valid image"
            raise TypeError

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = cv2.xfeatures2d.SURF_create(400).detectAndCompute(gray_image, None)
        return kps, des


# resize the shorter dimension to 'new_size'
# while maintaining the aspect ratio
def resize_to_size(input_image, new_size=150):
    h, w = input_image.shape[0], input_image.shape[1]
    ds_factor = new_size / float(h)

    if w < h:
        ds_factor = new_size / float(w)

    new_size = (int(w * ds_factor), int(h * ds_factor))
    return cv2.resize(input_image, new_size)

def start():
    input_map = []

    samples = [
        ["calculator", "images/calculator/"],      #1
        ["key", "images/key/"],                    #2
        ["multimeter", "images/multimeter/"],      #3
        ["pen", "images/pen/"],                    #4
        ["watch", "images/watch/"],                #5
        ["unrecognized", "images/unrecognized/"]
        ]

    codebook_file = "codebook.pkl"
    feature_map_file = "feature_map.pkl"

    for cls in samples:
        assert len(cls) >= 2, "Format for classes is '<label> file'"
        label = cls[0]
        input_map += load_input_map(label, cls[1])

    # building the codebook
    print "============= Building Codebook ============="
    kmeans, centroids = FeatureExtractor().get_centroids(input_map)
    if codebook_file:
        with open(codebook_file, 'wb') as f:
            pickle.dump((kmeans, centroids), f)

    # input data and labels
    print "============= Building feature map ============="
    feature_map = extract_feature_map(input_map, kmeans, centroids)
    if feature_map_file:
        with open(feature_map_file, 'wb') as f:
            pickle.dump(feature_map, f)

if __name__ == '__main__':
    start()
