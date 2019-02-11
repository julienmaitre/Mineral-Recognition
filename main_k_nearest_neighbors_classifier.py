import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets

import nearest_neighbors_classification

from sklearn.model_selection import train_test_split

import pickle

with open('dataset.pickle', 'rb') as f:
    X_train, y_train,  X_test, y_test, IDPeopleForTraining, IDPeopleForTesting = pickle.load(f)

print(len(X_train))
print(len(y_train))
print(len(X_test))
print(len(y_test))

class_names = np.array(['Assiette', 'Tasse', 'Cuillere', 'Couteau', 'Fourchette', 'Cafetiere', 'Poele', 'MainAVide'])

print('People ID for the training dataset :')
print('         %s' % IDPeopleForTraining)
print('People ID for the testing dataset :')
print('         %s' % IDPeopleForTesting)

########################################################################################################################
#                                 User Settings for the K-Nearest-Neighbors Classifier                                 #
########################################################################################################################

# Create a class object that define the parameters of the k-nearest-neighbors classifier
k_nearest_neighbors_parameters = nearest_neighbors_classification.KNearestNeighborsParameters()

""" Number of neighbors to use for 'kneighbors' queries. 
    It should be an integer (and by defaut=5). """
k_nearest_neighbors_parameters.n_neighbors = 5

""" Weight function used in prediction 
    The choices are :
                    ‘uniform’: uniform weights. All points in each neighborhood are weighted equally.
                    ‘distance’: weight points by the inverse of their distance. In this case, closer neighbors of a 
                                query point will have a greater influence than neighbors which are further away.
                    [callable]: a user-defined function which accepts an array of distances, and returns an array 
                                of the same shape containing the weights. """
k_nearest_neighbors_parameters.weights = 'distance'

""" Algorithm used to compute the nearest neighbors. 
    The choices are :
                    ‘ball_tree’: will use BallTree
                    ‘kd_tree’: will use KDTree
                    ‘brute’: will use a brute-force search.
                    ‘auto’: will attempt to decide the most appropriate algorithm based on the values passed
                            to fit method. 
                             
                    Note: fitting on sparse input will override the setting of this parameter, using brute force. """
k_nearest_neighbors_parameters.algorithm = 'auto'

""" Leaf size passed to BallTree or KDTree. 

    This can affect the speed of the construction and query, as well as the memory required to store the tree. 
    The optimal value depends on the nature of the problem.
    
    It should be an integer (and by defaut=30). 
    
    Note: This parameter is unnecessary if the algorithm is not BallTree or KDTree. """
k_nearest_neighbors_parameters.leaf_size = 30

""" Power parameter for the Minkowski metric. 

    When p = 1, this is equivalent to using manhattan_distance (l1), and euclidean_distance (l2) for p = 2. 
    For arbitrary p, minkowski_distance (l_p) is used.
    
    It should be an integer (and by defaut=2, so the euclidean_distance is the metric used by default). """
k_nearest_neighbors_parameters.p = 2

""" The distance metric to use for the tree. 

    The default metric is 'minkowski', and with p=2 is equivalent to the standard Euclidean metric. 
    The choices are :
                    For real-valued vector spaces
                      'euclidean': Euclidean Distance, no arguments, sqrt(sum((x - y)^2))
                      'manhattan': Manhattan Distance, no arguments, sum(|x - y|)
                      'chebyshev': Chebyshev Distance, no arguments, max(|x - y|)
                      'minkowski': Minkowski Distance, p, sum(|x - y|^p)^(1/p)
                      'wminkowski': WMinkowski Distance, p,w, sum(w * |x - y|^p)^(1/p)
                      'seuclidean': SEuclidean Distance, V, sqrt(sum((x - y)^2 / V))
                      'mahalanobis': Mahalanobis Distance, V, sqrt((x - y)' V^-1 (x - y)) """
k_nearest_neighbors_parameters.metric = 'euclidean'

""" Additional keyword arguments for the metric function. 
    The value by default is default=None. """
k_nearest_neighbors_parameters.metric_params = None

""" The number of parallel jobs to run for neighbors search. 
    If -1, then the number of jobs is set to the number of CPU cores. It doesn’t affect fit method.
    The value by default is default=1. """
k_nearest_neighbors_parameters.n_jobs = 1


# Create a class object that define the performances container of the k-nearest-neighbors classifier
performances = nearest_neighbors_classification.PerformancesKNearestNeighbors()


########################################################################################################################
#                               Execute the K-Nearest-Neighbors Classifier on the Dataset                              #
########################################################################################################################

# Create and train the k-nearest-neighbors classifier
k_nearest_neighbors_classifier, training_running_time = \
    nearest_neighbors_classification.train_k_nearest_neighbors_classifier(X_train, y_train, k_nearest_neighbors_parameters)

# Print information in the console
print("The training process of k-nearest-neighbors classifier took : %.8f second" % training_running_time)

# Test the k-nearest-neighbors classifier
y_test_predicted, testing_running_time = \
    nearest_neighbors_classification.test_k_nearest_neighbors_classifier(X_test, k_nearest_neighbors_classifier)

# Print information in the console
print("The testing process of k-nearest-neighbors classifier took : %.8f second" % testing_running_time)

# Compute the performances of the k-nearest-neighbors classifier
cm = nearest_neighbors_classification.compute_performances_for_multiclass(y_test, y_test_predicted, class_names,
                                                                          performances)

# Display the results
nearest_neighbors_classification.display_confusion_matrix(performances, class_names)
nearest_neighbors_classification.display_features_and_classification_for_knn_classifier(X_test, y_test, class_names,
                                                                                        k_nearest_neighbors_classifier,
                                                                                        k_nearest_neighbors_parameters)

plt.show()