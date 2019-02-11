########################################################################################################################
# Author : Julien Maitre                                                                                               #
# Date : 03 - 08 - 2018                                                                                                #
# Version : 0.1                                                                                                        #
########################################################################################################################

""" This file defines the functions for a k-nearest-neighbors classifier.
    It allows to train, test (even cross validate), compute the performances and display the results. """

import time

import numpy as np
import itertools
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn import neighbors

from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, classification_report,\
    matthews_corrcoef, hamming_loss, jaccard_similarity_score, precision_score, recall_score, f1_score

########################################################################################################################
#                         Define the Classes to be Used for the K-Nearest-Neighbors Classifier                         #
########################################################################################################################


class KNearestNeighborsParameters(object):

    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                 metric_params=None, n_jobs=1):

        self.n_neighbors = n_neighbors
        self.weights = weights
        self.algorithm = algorithm
        self.leaf_size = leaf_size
        self.p = p
        self.metric = metric
        self.metric_params = metric_params
        self.n_jobs = n_jobs


########################################################################################################################
#               Define the Classes to be Used for the Performances of the K-Nearest-Neighbors Classifier               #
########################################################################################################################


class PerformancesKNearestNeighbors(object):

    def __init__(self, accuracy_fraction=None, accuracy_number=None, cohen_kappa_score=None,
                 confusion_matrix_without_normalization=None, confusion_matrix_with_normalization=None,
                 classification_report=None, hamming_loss=None, jaccard_similarity_score_with_normalization=None,
                 jaccard_similarity_score_without_normalization=None, micro_precision=None, macro_precision=None,
                 weighted_precision=None, none_precision=None, micro_recall=None, macro_recall=None,
                 weighted_recall=None, none_recall=None, micro_f1_score=None, macro_f1_score=None,
                 weighted_f1_score=None, none_f1_score=None, matthews_corrcoef=None):

        self.accuracy_score_fraction = accuracy_fraction
        self.accuracy_score_number = accuracy_number

        self.cohen_kappa_score = cohen_kappa_score

        self.confusion_matrix_without_normalization = confusion_matrix_without_normalization
        self.confusion_matrix_with_normalization = confusion_matrix_with_normalization

        self.classification_report = classification_report

        self.hamming_loss = hamming_loss

        self.jaccard_similarity_score_with_normalization = jaccard_similarity_score_with_normalization
        self.jaccard_similarity_score_without_normalization = jaccard_similarity_score_without_normalization

        self.micro_precision = micro_precision
        self.macro_precision = macro_precision
        self.weighted_precision = weighted_precision
        self.none_precision = none_precision

        self.micro_recall = micro_recall
        self.macro_recall = macro_recall
        self.weighted_recall = weighted_recall
        self.none_recall = none_recall

        self.micro_f1_score = micro_f1_score
        self.macro_f1_score = macro_f1_score
        self.weighted_f1_score = weighted_f1_score
        self.none_f1_score = none_f1_score

        self.matthews_corrcoef = matthews_corrcoef


########################################################################################################################
#                      Define the Functions for the Training of the K-Nearest-Neighbors Classifier                     #
########################################################################################################################


def train_k_nearest_neighbors_classifier(x_train, y_train, k_nearest_neighbors_parameters):

    # Print information in the console
    print("\nThe k-nearest-neighbors classifier will be created")

    # Create an instance of neighbours classifier
    k_nearest_neighbors_classifier = neighbors.KNeighborsClassifier(k_nearest_neighbors_parameters.n_neighbors,
                                                                    k_nearest_neighbors_parameters.weights,
                                                                    k_nearest_neighbors_parameters.algorithm,
                                                                    k_nearest_neighbors_parameters.leaf_size,
                                                                    k_nearest_neighbors_parameters.p,
                                                                    k_nearest_neighbors_parameters.metric,
                                                                    k_nearest_neighbors_parameters.metric_params,
                                                                    k_nearest_neighbors_parameters.n_jobs)

    # Print information in the console
    print("The k-nearest-neighbors classifier has been created")
    print("The k-nearest-neighbors classifier is training")

    # Get the start time of the training process
    start_time = time.time()

    # Train the model using the training sets
    k_nearest_neighbors_classifier.fit(x_train, y_train)

    # Get the end time of the training process
    end_time = time.time()

    # Compute the time that the training process took
    running_time = end_time - start_time

    # Print information in the console
    print("The k-nearest-neighbors classifier has done its training process")

    return k_nearest_neighbors_classifier, running_time


########################################################################################################################
#                      Define the Functions for the Testing of the K-Nearest-Neighbors Classifier                      #
########################################################################################################################


def test_k_nearest_neighbors_classifier(x_test, k_nearest_neighbors_classifier):

    # Print information in the console
    print("\nThe k-nearest-neighbors classifier is being tested with the testing set")

    # Get the start time of the testing process
    start_time = time.time()

    # Make predictions using the testing set
    y_test_predicted = k_nearest_neighbors_classifier.predict(x_test)

    # Get the end time of the testing process
    end_time = time.time()

    # Compute the time that the testing process took
    running_time = end_time - start_time

    # Print information in the console
    print("The k-nearest-neighbors classifier has done its testing process")

    return y_test_predicted, running_time


########################################################################################################################
#                 Define the Functions for Computing Performances of the K-Nearest-Neighbors Classifier                #
########################################################################################################################


def compute_performances_for_multiclass(y_test, y_test_predicted, class_names, performances):

    # Compute the accuracy classification score : return the fraction of correctly classified samples
    performances.accuracy_score_fraction = accuracy_score(y_test, y_test_predicted, normalize=True)
    # Compute the accuracy classification score : return return the number of correctly classified samples
    performances.accuracy_score_number = accuracy_score(y_test, y_test_predicted, normalize=False)

    # Print information in the console
    print("\nAccuracy classification score : ")
    print("         Fraction of correctly classified samples : %.2f" % performances.accuracy_score_fraction)
    print("         Number of correctly classified samples: %.2f" % performances.accuracy_score_number)

    # Compute the Cohen's kappa score
    performances.cohen_kappa_score = cohen_kappa_score(y_test, y_test_predicted)

    # Print information in the console
    print("\nCohen's kappa score : %.2f" % performances.cohen_kappa_score)

    # Compute the confusion matrix without normalization
    performances.confusion_matrix_without_normalization = confusion_matrix(y_test, y_test_predicted)
    # Compute the confusion matrix with normalization
    performances.confusion_matrix_with_normalization = \
        performances.confusion_matrix_without_normalization.astype('float') \
        / performances.confusion_matrix_without_normalization.sum(axis=1)[:, np.newaxis]

    # Print information in the console
    print("\nConfusion matrix : ")
    print("     Confusion matrix without normalization : ")
    square_matrix_size = len(performances.confusion_matrix_without_normalization)
    for i in range(square_matrix_size):
        if i == 0:
            print('                 [' + np.array2string(performances.confusion_matrix_without_normalization[i]))
        elif i == square_matrix_size -1:
            print('                  ' + np.array2string(performances.confusion_matrix_without_normalization[i]) + ']')
        else:
            print('                  ' + np.array2string(performances.confusion_matrix_without_normalization[i]))
    print("     Confusion matrix with normalization : ")
    square_matrix_size = len(performances.confusion_matrix_with_normalization)
    for i in range(square_matrix_size):
        if i == 0:
            print('                 [' + np.array2string(performances.confusion_matrix_with_normalization[i]))
        elif i == square_matrix_size - 1:
            print('                  ' + np.array2string(performances.confusion_matrix_with_normalization[i]) + ']')
        else:
            print('                  ' + np.array2string(performances.confusion_matrix_with_normalization[i]))

    # Compute the classification_report
    performances.classification_report = classification_report(y_test, y_test_predicted, target_names=class_names,
                                                               digits=4)

    # Print information in the console
    print("\nclassification_report : ")
    print(performances.classification_report)

    # Compute the average Hamming loss
    performances.hamming_loss = hamming_loss(y_test, y_test_predicted)

    # Print information in the console
    print("\nAverage Hamming loss : %.2f" % performances.hamming_loss)

    # Compute the Jaccard similarity coefficient score with normalization
    performances.jaccard_similarity_score_with_normalization = jaccard_similarity_score(y_test, y_test_predicted,
                                                                                        normalize=True)
    # Compute the Jaccard similarity coefficient score without normalization
    performances.jaccard_similarity_score_without_normalization = jaccard_similarity_score(y_test, y_test_predicted,
                                                                                           normalize=False)

    # Print information in the console
    print("\nJaccard similarity coefficient score : ")
    print("     Average of Jaccard similarity coefficient : %.2f"
          % performances.jaccard_similarity_score_with_normalization)
    print("     Sum of the Jaccard similarity coefficient over the sample set : %.2f"
          % performances.jaccard_similarity_score_without_normalization)

    # Compute the precision
    performances.micro_precision = precision_score(y_test, y_test_predicted, average='micro')
    performances.macro_precision = precision_score(y_test, y_test_predicted, average='macro')
    performances.weighted_precision = precision_score(y_test, y_test_predicted, average='weighted')
    performances.none_precision = precision_score(y_test, y_test_predicted, average=None)

    # Print information in the console
    print("\nPrecision score : ")
    print("     micro : %.2f" % performances.micro_precision)
    print("     macro : %.2f" % performances.macro_precision)
    print("     weighted : %.2f" % performances.weighted_precision)
    print("     None : " + np.array2string(performances.none_precision))
    print("     Classes : " + np.array2string(class_names))

    # Compute the recall
    performances.micro_recall = recall_score(y_test, y_test_predicted, average='micro')
    performances.macro_recall = recall_score(y_test, y_test_predicted, average='macro')
    performances.weighted_recall = recall_score(y_test, y_test_predicted, average='weighted')
    performances.none_recall = recall_score(y_test, y_test_predicted, average=None)

    # Print information in the console
    print("\nRecall score : ")
    print("     micro : %.2f" % performances.micro_recall)
    print("     macro : %.2f" % performances.macro_recall)
    print("     weighted : %.2f" % performances.weighted_recall)
    print("     None : " + np.array2string(performances.none_recall))
    print("     Classes : " + np.array2string(class_names))

    # Compute the F1 score
    performances.micro_f1_score = f1_score(y_test, y_test_predicted, average='micro')
    performances.macro_f1_score = f1_score(y_test, y_test_predicted, average='macro')
    performances.weighted_f1_score = f1_score(y_test, y_test_predicted, average='weighted')
    performances.none_f1_score = f1_score(y_test, y_test_predicted, average=None)

    # Print information in the console
    print("\nF1-score : ")
    print("     micro : %.2f" % performances.micro_f1_score)
    print("     macro : %.2f" % performances.macro_f1_score)
    print("     weighted : %.2f" % performances.weighted_f1_score)
    print("     None : " + np.array2string(performances.none_f1_score))
    print("     Classes : " + np.array2string(class_names))

    # Compute the Matthews correlation coefficient
    performances.matthews_corrcoef = matthews_corrcoef(y_test, y_test_predicted)

    # Print information in the console
    print("\nMatthews correlation coefficient : %.2f" % performances.matthews_corrcoef)

    return performances


########################################################################################################################
#              Define the Functions to Display the Results on Graph for the K-Nearest-Neighbors Classifier             #
########################################################################################################################


def display_confusion_matrix(performances, class_names, title='Confusion matrix', cmap=plt.cm.Blues):

    cm = performances.confusion_matrix_without_normalization

    if len(class_names) <= 10:

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)

        fmt = 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

    else:

        print("The confusion matrix cannot be displayed because the number of classes is superior to 10")

    plt.show()


def display_features_and_classification_for_knn_classifier(x, y, class_names, k_nearest_neighbors_classifier,
                                                           k_nearest_neighbors_parameters):

    number_of_classes = len(class_names)
    number_of_features = len(x[0])

    all_combination = list(itertools.combinations(range(number_of_features), 2))
    number_of_combination = len(all_combination)

    # blue, red, green, yellow, brown, magenta, black, DarSlateBlue, DimGray, DarkOrange
    list_cmap_bold = ['#0000FF', '#FF0000', '#008000', '#FFFF00', '#A52A2A', '#FF00FF', '#000000',
                      '#483D8B', '#696969', '#FF8C00']

    # LightBlue, LightSalmon, LightGreen, yellow, brown, violet, DarkGray, SlateBlue, LightSlateGray, Orange
    list_cmap_light = ['#ADD8E6', '#FFA07A', '#90EE90', '#FFFF00', '#A52A2A', '#EE82EE', '#A9A9A9',
                      '#6A5ACD', '#778899', '#FFA500']

    if (len(class_names) <= 10) and (number_of_features <= 5):

        # Step size in the mesh
        h = .01

        # Create color maps
        cmap_light = ListedColormap(list_cmap_light[0:number_of_classes])
        cmap_bold = ListedColormap(list_cmap_bold[0:number_of_classes])

        indice_plot_1 = number_of_combination/2

        for i in range(number_of_combination):

            indice_1 = all_combination[i][0]
            indice_2 = all_combination[i][1]

            if ((number_of_combination % 2) == 1) and (number_of_combination <= 1):

                # Plot the decision boundary.
                # For that, we will assign a color to each point in the mesh [x_min, x_max]x[y_min, y_max].
                x_min, x_max = x[:, indice_1].min() - 1, x[:, indice_1].max() + 1
                y_min, y_max = x[:, indice_2].min() - 1, x[:, indice_2].max() + 1

                xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
                Z = k_nearest_neighbors_classifier.predict(np.c_[xx.ravel(), yy.ravel()])

                # Put the result into a color plot
                Z = Z.reshape(xx.shape)
                plt.figure()
                plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

                # Plot also the training points
                plt.scatter(x[:, indice_1], x[:, indice_2], c=y, cmap=cmap_bold, edgecolor='k', s=20)

                plt.xlim(xx.min(), xx.max())
                plt.ylim(yy.min(), yy.max())

            elif (number_of_combination % 2) == 0:

                ppp = str(int(indice_plot_1)) + str(2) + str(i+1)

                plt.subplot(int(ppp))
                plt.scatter(x[:, indice_1], x[:, indice_2], c=y, cmap=cmap_bold, edgecolor='k', s=20)
                plt.title("%i-Class classification of feature %i and feature %i"
                          % (number_of_classes, indice_1, indice_2))
                plt.xlabel('Feature ' + str(indice_1))
                plt.ylabel('Feature ' + str(indice_2))

        n_neighbors = k_nearest_neighbors_parameters.n_neighbors
        weights = k_nearest_neighbors_parameters.weights
        plt.tight_layout()
        plt.suptitle("%i-Class classification (k = %i, weights = '%s')" % (number_of_classes, n_neighbors, weights))

        plt.show()

    else:

        print("The confusion matrix cannot be displayed because the number of classes is superior to 10 or the number "
              "of features is superior to 5")
