# This module is the pre_processing_pipeline for the assignment.
# i.e it is designed to import the data from folders, compute histograms and generate the vocabulary
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from sklearn.cluster import KMeans


def read_images_to_array(path=""):
    """
    Tasks in path of the folders where data is stored and returns a list of images(concatinated)
    and list of corresponding labels
    :param path: Directory path to where the data/images are stored
    :return: image_list: Concatinated list of grayscale images.
             labels: List of corresponding labels.
    """
    label_list = os.listdir(path)
    image_list = []
    labels = []
    for folder in label_list:
        file_list = os.listdir(path + "\\" + folder)
        for file_ in file_list:
            try:
                image_list.append(cv2.imread(path + "\\" + folder + "\\" + file_, cv2.IMREAD_GRAYSCALE))
                labels.append(folder)
            except:
                print("Error reading file " + file_)
    return image_list, labels


def show_image(img):
    """
    Show image function meant to display image, incorporates the plt.show() for faster debugging
    :param img: Image to display
    :return: Nothing:0
    """
    plt.imshow(img)
    plt.show()
    return 0


# used during initial testing
# test_set, test_labels = read_images_to_array("Objects_Dataset_Images\\test")
# train_set, train_labels = read_images_to_array("Objects_Dataset_Images\\train")


def compute_descriptor_array(img_arr):
    """
    Computes the descriptors of the list of images pass
    :param img_arr: list of gray scale images
    :return: list of descriptors
    """
    descriptor_array = []
    sift = cv2.SIFT_create()
    for image in img_arr:
        descriptor_array.extend(
            sift.detectAndCompute(image[:, :], None)[1])  # creating one large list of descriptors to cluster later on
    return descriptor_array


def compute_cluster_centriods(descriptors, k=4):
    """
    Clusters the data in to specified K clusters using K means algorithm
    :param descriptors: Feature discriptors of the training set
    :param k: # of clusters
    :return: K means trained model
    """
    k_means_model = KMeans(n_clusters=k, random_state=42).fit(descriptors)
    return k_means_model


def compute_histogram(kmeans_model, descriptor, k):
    """
    Takes the trained K means model and the descriptors for an image and computes the histogram from said descriptors
    :param kmeans_model: Train/fitted cluster model
    :param descriptor: set of descriptors for the image
    :return: Histogram in the form of a numpy array of length = # of clusters
    """
    histogram = np.zeros((k))
    closest_cluster = kmeans_model.predict(np.array(descriptor, dtype=np.double))
    for index in closest_cluster:
        histogram[int(index)] += 1  # used int(index) since the closest_cluster values were in float representation form
    return histogram


def compute_vocabulary(image_arr, kmean, k):
    """
    Compute the Vocabulary using the trained k means model, This uses the point to cluster distance instead of the centeroid distance.
    This is to accmodate for the intra-class variability
    :param image_arr: Array of images whose vocabulary is to be generated.
    :param kmean: The trained k means model.
    :return: Vocabulary of array of images in form of a list
    """
    sift = cv2.SIFT_create()
    vocabulary_arr = []
    for image in image_arr:
        vocabulary_arr.append(compute_histogram(kmean, sift.detectAndCompute(image, None)[1], k))
    return vocabulary_arr
