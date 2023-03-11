# Computer Vision CS-867 (Assignment #1)

Image features Detection and Matching Image Classification Using Bag of Visual Words

**Muhammad Talha Imran
MSDS-2022**

## Abstract

Prior to the widespread adoption of deep learning, Bag of Visual words was the top of the line/state of the art model
for image classification tasks. The idea behind this technique is quite simple. Images of a particular class should have
discriminative keypoints/landmarks within images corresponding to same or similar classes, thus histograms for each
sample within the dataset can be generated, giving a feature map representation(with frequency of occurrence of each
feature with each of the images) of each image or a vocabulary. The
keypoints/features are
referred to as words while the histogram representations are referred to as vocabulary within the context of the
algorithm.
These histograms can then be used to train a classifier(SVM, Random forest etc.) to predict labels of new
images.

## Introduction

The dataset is first read into memory as grayscale images in the form of a list of 2D arrays along with the
corresponding labels. This array of images is then used to compute a list of descriptors(in our program we use SIFT) using
the opencv library. This list is then passed to the KMeans clustering algorithm which then computes K clusters from
these descriptors. Following this, for each of the samples the distance of each descriptor from the centeroid of each of
the clusters is measured, this in turn is used to create a histogram for each of the samples. These histograms(
vocabulary) are used to train a classifier(in our case SVM and Random forest) which is then used to generate inference
on the test samples.

**Following is an illustration of the above described steps:**

![Assignment_1_illustations_1.png](Figures%2FAssignment_1_illustations_1.png)

**The vocabulary generation process is illustrated below:**

![Assignment_1_illustations_2.png](Figures%2FAssignment_1_illustations_2.png)

## Issues and observations during training and implementation

**Mentioned below are some issues observed with the dataset during implementation and training**

- Accordian was mis-spelled- fixed caused error while processing


- Number of the examples within the dataset are irrelevent to the classification task at hand. I have removed them from
  the date. Some of the examples are mentioned below:

**These were removed from the rose sample folder, remainder examples are available in the
Figures\Dataset_mismatch_examples folder as reference
for the reader.**

![2863863372_605e29c03e_m.jpg](Figures%2FDataset_mismatch_examples%2F2863863372_605e29c03e_m.jpg)
![8590442797_07fa2141c0_n.jpg](Figures%2FDataset_mismatch_examples%2F8590442797_07fa2141c0_n.jpg)
![9406573080_60eab9278e_n.jpg](Figures%2FDataset_mismatch_examples%2F9406573080_60eab9278e_n.jpg)
![3713368809_eba7fa2fbf_m.jpg](Figures%2FDataset_mismatch_examples%2F3713368809_eba7fa2fbf_m.jpg)

## Instructions for setting up the environment for running the code

### Install dependencies

```bash
  pip install -r requirements.txt
``` 

**For manual installation**

- matplotlib==3.5.1
- numpy==1.21.5
- opencv_python==4.7.0.72
- pandas==1.4.2
- scikit-learn==1.0.2

## Instructions on running the train and test scripts on the train and test data

### link to dataset

```commandline
https://drive.google.com/drive/folders/1ebxtTnbVOZ5WWRH3g6hksg4Mb83j4R9-?usp=sharing
```

**Create** a **Datasets** folder in the project and place the dataset in the dataset folder(ignored while uploading to
github), place the contents in the dataset folder as
shown below.

![Dataset_folder.PNG](Figures%2FDataset_folder.PNG)

### Running the code

The files are ready to run and present in the SCR folder, after placing the dataset within the Datasets folder, specify
the value of **k** i.e. # of clusters and run the code.

![num_of_clusters.PNG](Figures%2Fnum_of_clusters.PNG)

#### .py file navigation

- **Assignment_1_1a.py** >> **Object dataset** using **SVM** with **k clusters**.
- **Assignment_1_1b.py** >> **Object dataset** using **Random forest** with **k clusters**.
- **Assignment_1_2a.py** >> **flower dataset** using **SVM** with **k clusters**.
- **Assignment_1_2b.py** >> **flower dataset** using **Random forest** with **k clusters**.
- **Pre_processing_pipeline.py** >> Contains the **complete preprocessing** and **vocabulary generation** pipeline.

![SCR_files.PNG](Figures%2FSCR_files.PNG)

## Quantitative results

Results in pictorial form, i.e confusion matrics, classification reports and pictorial results are present in the folder
mentioned below:

### Naming scheme and navigation:

#### Folders:

- Flower_RF >> **Flower(flower dataset)** & **RF(Random forest)**
- Flower_SVM >> **Flower(flower dataset)** & **SVM(Support Vector Machine)**
- Object_RF >> **Object(Object dataset)** & **RF(Random forest)**
- Object_SVM >> **Object(Object dataset)** & **SVM(Support Vector Machine)**

![Results_location.PNG](Figures%2FResults_location.PNG)

#### Files:

- Classification_report_k_x.csv >> **Classification report of the classifier with cluster k=x containing F1 Score,
  Precision, Recall etc.**

**Example:**
[Classification_report_k_128.csv](results%2FFlowers_RF%2FClassification_report_k_128.csv)

- flower_RF_k_x.png >> **Confusion matrix of the classifier on the dataset with clusters k=x**

**Example:**
![flowers_RF_k_128.png](results%2FFlowers_RF%2Fflowers_RF_k_128.png)

- Prediction_results_k_x.png >> **Samples images with prediciton results from the classifier**

**Example:**
![Predictions_results_k_128.png](results%2FFlowers_RF%2FPredictions_results_k_128.png)

### Detailed Results

**F1 Scores Object Dataset**

| Dataset | Classifier    | k   | Soccer_Ball | accordian | dollar_bill | motorbike |
|---------|---------------|-----|-------------|-----------|-------------|-----------|
| Objects | Random Forest | 2   | 0.80        | 0.50      | 0.50        | 0.67      |
|         |               | 4   | 0.80        | 0.80      | 0.67        | 0.67      |
|         |               | 10  | 0.67        | 0.67      | 0.67        | 0.50      |
|         |               | 50  | 1.00        | 0.67      | 0.80        | 1.00      |
|         |               | 128 | 1.00        | 1.00      | 0.80        | 0.67      |
| Objects | SVM           | 2   | 0.00        | 0.67      | 0.80        | 0.40      |
|         |               | 4   | 0.50        | 0.67      | 1.00        | 0.80      |
|         |               | 10  | 0.67        | 0.67      | 0.80        | 0.80      |
|         |               | 50  | 0.50        | 0.67      | 1.00        | 0.80      |
|         |               | 128 | 0.50        | 0.67      | 1.00        | 0.80      |

**F1 Scores Flowers Dataset**

| F1 Score |               |     |       |           |       |            |        |
|----------|---------------|-----|-------|-----------|-------|------------|--------|
| Dataset  | Classifier    | k   | daisy | dandelion | roses | sunflowers | tulips |
| Flowers  | SVM           | 2   | 0.00  | 0.42      | 0.00  | 0.35       | 0.13   |
|          |               | 4   | 0.26  | 0.49      | 0.23  | 0.40       | 0.16   |
|          |               | 5   | 0.30  | 0.50      | 0.29  | 0.40       | 0.16   |
|          |               | 10  | 0.36  | 0.54      | 0.24  | 0.51       | 0.41   |
|          |               | 50  | 0.47  | 0.66      | 0.36  | 0.63       | 0.47   |
|          |               | 128 | 0.54  | 0.70      | 0.49  | 0.65       | 0.56   |
|          |               | 200 | 0.56  | 0.72      | 0.52  | 0.64       | 0.54   |
| Flowers  | Random Forest | 2   | 0.23  | 0.27      | 0.13  | 0.34       | 0.25   |
|          |               | 4   | 0.28  | 0.48      | 0.31  | 0.37       | 0.30   |
|          |               | 5   | 0.33  | 0.45      | 0.33  | 0.42       | 0.27   |
|          |               | 10  | 0.42  | 0.54      | 0.36  | 0.50       | 0.39   |
|          |               | 50  | 0.45  | 0.63      | 0.41  | 0.52       | 0.44   |
|          |               | 128 | 0.49  | 0.67      | 0.41  | 0.57       | 0.50   |
|          |               | 200 | 0.50  | 0.68      | 0.49  | 0.59       | 0.50   |

**Accuracy Scores Object Dataset**

|         |     | Accuracy % | Accuracy %    |
|---------|-----|------------|---------------|
| Dataset | k   | SVM        | Random forest |
| Object  | 2   | 50         | 62.5          |
|         | 4   | 75         | 75            |
|         | 10  | 75         | 62.5          |
|         | 50  | 75         | 87.5          |
|         | 128 | 75         | 87.5          |

**Accuracy Scores Flowers Dataset**

|         |     | Accuracy % | Accuracy %    |
|---------|-----|------------|---------------|
| Dataset | k   | SVM        | Random forest |
| Flowers | 2   | 29.56      | 25.47         |
|         | 4   | 35.69      | 36.69         |
|         | 5   | 37.38      | 36.83         |
|         | 10  | 44.41      | 45.56         |
|         | 50  | 53.95      | 50.47         |
|         | 128 | 60.21      | 54.84         |
|         | 200 | 60.57      | 56.48         |