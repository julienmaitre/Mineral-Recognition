import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split

import pickle


""" This function counts the number of instances per class in the dataset """


def instance_number_per_class(classes, name_of_dataset):

    print("We are counting the number of instances per class for the " + name_of_dataset + ". Please wait ...")

    # Get all names of the different classes
    class_names = np.unique(classes)

    # Initialize variables
    matches_per_class = {}

    # Do for each class
    for i in range(len(class_names)):

        # Initialize a variable
        match = []

        # Compute the number of instances in the dataset corresponding to the class i
        for j in range(len(classes)):

            # If the instance j corresponds to the class i put 1 in match, else 0
            match.append(classes[j] == class_names[i])

        # Compute the total number of instances corresponding to the class i
        number_of_instances_for_1_class = sum(match)

        # If the number of instances for the class i is 0 put 0, else put the total number of instances corresponding
        # to the class i
        if number_of_instances_for_1_class == 0:

            matches_per_class.update({class_names[i]: 0})

        else:

            matches_per_class.update({class_names[i]: number_of_instances_for_1_class})

        del match

    matches_per_class_sorted = sorted(matches_per_class.items())

    print(" \nThe number of instances per class of the " + name_of_dataset)
    print(matches_per_class_sorted)
    print("\n")

    return matches_per_class_sorted


""" This function re-sample the dataset according to a minimum number of instances per class. In fact, we keep only
    classes with this minimum number and correspond also to a maximum number of instances """


def resample_dataset(x, y, maximum_number_of_instances):

    # Get all names of the different classes
    class_names = np.unique(y)

    # Initialize variables
    x_red = []
    y_red = []

    for i in range(len(class_names)):

        if class_names[i] != "Inconnu" and class_names[i] != "Quartz":

            # Get all observation indices that correspond to the class "i"
            indexes = [index for index in range(len(y)) if y[index] == class_names[i]]

            # Permute the indices (in a random manner) in order to re-sample the dataset corresponding to the class "i"
            permutation_of_indices = np.random.permutation(indexes)

            # Re-sample the dataset for the class "i"
            if len(indexes) >= maximum_number_of_instances:

                list_1 = list(x[i] for i in permutation_of_indices)
                list_2 = list(y[i] for i in permutation_of_indices)

                # Keep only the first "maximum_number_of_instances" instances
                x_red = x_red + list_1[0:maximum_number_of_instances]
                y_red = y_red + list_2[0:maximum_number_of_instances]

                del list_1, list_2

            del indexes, permutation_of_indices

    return x_red, y_red


""" This function allows to group different classes together in the dataset. It corresponds to a part of data cleaning,
    or data exclusion. """


def group_classes(x, y):

    # Initialize variables
    x_new = []
    y_new = []

    ####################################################################################################################
    # Group together "Albite, None" and "Microcline, None" classes

    # Get indices of instances corresponding to the "Albite, None" and "Microcline, None" classes
    indexes_albite = [index for index in range(len(y)) if y[index] == "Albite, None"]
    indexes_microcline = [index for index in range(len(y)) if y[index] == "Microcline, None"]

    # Get instances of feature vectors in the list corresponding to the "Albite, None" and "Microcline, None" classes
    list_albite = list(x[i] for i in indexes_albite)
    list_microcline = list(x[i] for i in indexes_microcline)

    # Get all the instances in x_new
    x_new = x_new + list_albite + list_microcline

    del list_albite, list_microcline

    # With the same indexes, get the class names of those indexes
    list_albite = list(y[i] for i in indexes_albite)
    list_microcline = list(y[i] for i in indexes_microcline)

    # Put the two lists together
    yy = list_albite + list_microcline

    # Compute the number of instances
    number_of_instances = len(yy)

    # Give new class name to those instances
    yy = ["Albite, Microcline"] * number_of_instances

    # Store the class name of instances
    y_new = y_new + yy

    del yy

    ####################################################################################################################
    # Group together "Augite, None" and "Tschermakite, None" classes

    # Do the same for instances of the classes "Augite, None" class
    indexes_augite = [index for index in range(len(y)) if y[index] == "Augite, None"]
    indexes_tschermakite = [index for index in range(len(y)) if y[index] == "Tschermakite, None"]

    list_augite = list(x[i] for i in indexes_augite)
    list_tschermakite = list(x[i] for i in indexes_tschermakite)

    x_new = x_new + list_augite + list_tschermakite

    del list_augite, list_tschermakite

    list_augite = list(y[i] for i in indexes_augite)
    list_tschermakite = list(y[i] for i in indexes_tschermakite)

    yy = list_augite + list_tschermakite
    number_of_instances = len(yy)

    yy = ["Augite, Tschermakite"] * number_of_instances

    y_new = y_new + yy

    del yy

    ####################################################################################################################
    # Group together "Background" and "Background, None" classes

    # Do the same for instances of the classes "Background"
    indexes_background = [index for index in range(len(y)) if y[index] == "Background"]
    indexes_background_none = [index for index in range(len(y)) if y[index] == "Background, None"]

    list_background = list(x[i] for i in indexes_background)
    list_background_none = list(x[i] for i in indexes_background_none)

    x_new = x_new + list_background + list_background_none

    del list_background, list_background_none

    list_background = list(y[i] for i in indexes_background)
    list_background_none = list(y[i] for i in indexes_background_none)

    yy = list_background + list_background_none

    number_of_instances = len(yy)

    yy = ["Background"]*number_of_instances

    y_new = y_new + yy

    del yy

    # Do the same for instances of the classes "Hypersthene"
    indexes_hypersthene = [index for index in range(len(y)) if y[index] == "Hypersthene"]

    list_hypersthene = list(x[i] for i in indexes_hypersthene)

    x_new = x_new + list_hypersthene

    del list_hypersthene

    list_hypersthene = list(y[i] for i in indexes_hypersthene)

    yy = list_hypersthene
    number_of_instances = len(yy)
    yy = ["Hypersthene, None"] * number_of_instances

    y_new = y_new + yy

    del yy

    ####################################################################################################################
    # Group together "Ilmenite, None" and "Magnetite, None" classes

    # Get instances of the "Ilmenite, None" and "Magnetite, None" classes
    indexes_ilmenite = [index for index in range(len(y)) if y[index] == "Ilmenite, None"]
    indexes_magnetite = [index for index in range(len(y)) if y[index] == "Magnetite, None"]

    list_ilmenite = list(x[i] for i in indexes_ilmenite)
    list_magnetite = list(x[i] for i in indexes_magnetite)

    x_new = x_new + list_ilmenite + list_magnetite

    del list_ilmenite, list_magnetite

    list_ilmenite = list(y[i] for i in indexes_ilmenite)
    list_magnetite = list(y[i] for i in indexes_magnetite)

    yy = list_ilmenite + list_magnetite
    number_of_instances = len(yy)
    yy = ["Ilmenite, Magnetite"] * number_of_instances

    y_new = y_new + yy

    del yy

    # Get instances of the "Pumpellyite" class
    indexes_pumpellyite = [index for index in range(len(y)) if y[index] == "Pumpellyite, None"]

    list_pumpellyite = list(x[i] for i in indexes_pumpellyite)
    x_new = x_new + list_pumpellyite

    del list_pumpellyite

    list_pumpellyite = list(y[i] for i in indexes_pumpellyite)
    yy = list_pumpellyite
    number_of_instances = len(yy)
    yy = ["Pumpellyite"] * number_of_instances

    y_new = y_new + yy

    del yy

    # Get instances of the "Titanite" class
    indexes_titanite = [index for index in range(len(y)) if y[index] == "Titanite, None"]

    list_titanite = list(x[i] for i in indexes_titanite)
    x_new = x_new + list_titanite

    del list_titanite

    list_titanite = list(y[i] for i in indexes_titanite)
    yy = list_titanite
    number_of_instances = len(yy)
    yy = ["Titanite"] * number_of_instances

    y_new = y_new + yy

    return x_new, y_new


""" This function allows to exclude outlier data according to the isolation forest algorithm """


def exclude_outlier_instances(x, y):

    # Get all different class names
    class_names = np.unique(y)

    x_new = []
    y_new = []

    for i in range(len(class_names)):

        indexes = [index for index in range(len(y)) if y[index] == class_names[i]]

        list_1 = list(x[i] for i in indexes)
        list_2 = list(y[i] for i in indexes)

        classifier = IsolationForest(max_samples=1000, random_state=np.random.RandomState(42), max_features=20,
                                     contamination=0.8)
        classifier.fit(list_1)
        y_predicted = classifier.predict(list_1)

        del indexes

        indexes = [index for index in range(len(y_predicted)) if y_predicted[index] == 1]

        list_3 = list(list_1[i] for i in indexes)
        list_4 = list(list_2[i] for i in indexes)

        x_new = x_new + list_3
        y_new = y_new + list_4

        del list_1, list_2, list_3, list_4, indexes

    return x_new, y_new


""" This function allows to split the dataset into a training and testing datasets """


def classic_splitting_of_dataset(x, y, test_size):

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=0)

    return x_train, x_test, y_train, y_test


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the path of the dataset to process
pathName = 'C:/Users/Julien/Desktop/Echantillon 1/InformationFile5.pickle'

# Define the minimum number of instances per class that you want (it correspond also to the maximum number of
# observations after the re-sample)
minimumInstanceNumber = 2000

# Define the ratio for the testing dataset
testingDatasetRatio = 0.3

########################################################################################################################
#                                                   Load the Dataset                                                   #
########################################################################################################################

# Get back the dataset (Instances - Attributes + Classes)
with open(pathName, 'rb') as f:
    dataset, classes = pickle.load(f)

########################################################################################################################
#                                                Preprocess the Dataset                                                #
########################################################################################################################

# Compute the number of instances per class in the original dataset
matchesPerClass = instance_number_per_class(classes, "original dataset")

# Re-sample the dataset according to a minimum number of instances per class
newDataset, newClasses = resample_dataset(dataset, classes, minimumInstanceNumber)

# Recompute the number of instances per class
matchesPerClass = instance_number_per_class(newClasses, "resample dataset")

# Group certain classes together
newDataset, newClasses = group_classes(newDataset, newClasses)

# Recompute the number of instances per class
matchesPerClass = instance_number_per_class(newClasses, "resample dataset grouped")

# Exclude outlier instances
newDataset, newClasses = exclude_outlier_instances(newDataset, newClasses)

# Recompute the number of instances per class
matchesPerClass = instance_number_per_class(newClasses, "new dataset to exploit for classification")

# Prepare the training dataset and the testing dataset with a ratio xx% (training) - yy% (testing)
xTrain, xTest, yTrain, yTest = classic_splitting_of_dataset(newDataset, newClasses, testingDatasetRatio)

########################################################################################################################
#                                          Save the Dataset for Classification                                         #
########################################################################################################################

newClasses = np.unique(newClasses)

with open('C:/Users/Julien/Desktop/Echantillon 1/Classification_Dataset_V1.pickle', 'wb') as f:
    pickle.dump([xTrain, xTest, yTrain, yTest, newClasses], f)

