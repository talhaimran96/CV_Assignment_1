# This part is the implementation of classification of the object dataset using SVM and different number of clusters
# Ranging from 2 to 128, resulting confusion matrices are present in the results folder

import Pre_processing_pipeline as preprocess
from sklearn.cluster import KMeans
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd

test_set, test_labels = preprocess.read_images_to_array("../Datasets/Objects_Dataset_Images/test")
train_set, train_labels = preprocess.read_images_to_array("../Datasets/Objects_Dataset_Images/train")

k = 2  # defining k as number of clusters

kmean = preprocess.compute_cluster_centriods(preprocess.compute_descriptor_array(train_set), k=k)
training_vocabulary = preprocess.compute_vocabulary(train_set, kmean=kmean, k=k)
test_vocabulary = preprocess.compute_vocabulary(test_set, kmean=kmean, k=k)

label_encoder = LabelEncoder()  # creating an instance of labelencoder, to convert categorical to numerical labels
label_encoder.fit(train_labels)  # fitting the label encoder

# encoding the test and training labels to numerical values
test_labs = label_encoder.transform(test_labels)  # accordian was named accordion in the training lable folder
# (I have renamed it to conform with the other one)
train_labs = label_encoder.transform(train_labels)

# SVM classifier
svm_classifier = svm.SVC()  # Creating an instance of SVM class, Only checked with linear SVM
svm_classifier.fit(training_vocabulary, train_labs)  # Training/fitting the SVM classifier
predicted_labs = (svm_classifier.predict(test_vocabulary))  # getting prediction on the test dataset

# Inverting the encodings for predicted and test labels
print(label_encoder.inverse_transform(test_labs))
print(label_encoder.inverse_transform(predicted_labs))

# Classification report
print(classification_report(test_labels, label_encoder.inverse_transform(predicted_labs),
                            target_names=label_encoder.classes_))
# Confusion Matrix
cm = confusion_matrix(test_labels, label_encoder.inverse_transform(predicted_labs),
                      labels=label_encoder.classes_)

display = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
display.plot()
plt.title(f"k = {k}")
plt.show()




def save_classification_report(report, path, k):
    """
    Save the classification report as csv
    :param report: Report to save
    :param path: Where to save the report
    :param k: hyper-parameter, k=# of clusters
    :return: Nothing
    """
    df = pd.DataFrame(report)
    df.to_csv(path + "\\" + "Classification_report_" +"k_"+ f"{k}.csv")
    return 0


def display_examples(images, actual_labs, predicted_labs):
    """
    To display examples of predictions on images
    :param images: Images to display
    :param actual_labs: Correct labels
    :param predicted_labs: Predicted labels from the classifier
    :return: Nothing
    """
    p = len(images)
    q = 2
    p = p // 2
    i = 1
    for img in images:
        plt.subplot(q, p, i)
        plt.imshow(img, cmap="gray")
        plt.title("Actual lab: " + actual_labs[i - 1] + ", "
                  + "Prediction: " + predicted_labs[i - 1])
        i += 1
    plt.show()
    return 0


## commented post data collection

# report = classification_report(test_labels, label_encoder.inverse_transform(predicted_labs),
#                                target_names=label_encoder.classes_, output_dict=True)
# save_classification_report(report=report, path="results\\Objects_SVM", k=k)
# display_examples(test_set, test_labels, label_encoder.inverse_transform(predicted_labs))
