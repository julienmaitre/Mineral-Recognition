import os

import warnings

import time
import progressbar as pb

import numpy as np

import cv2
from skimage.segmentation import slic

import pickle

# Ignore the warning messages
warnings.filterwarnings("ignore")

# Define progress timer class
class progress_timer:

    def __init__(self, n_iter, description="Something"):
        self.n_iter = n_iter
        self.iter = 0
        self.description = description + ': '
        self.timer = None
        self.initialize()

    def initialize(self):
        # Initialize timer
        widgets = [self.description, pb.Percentage(), ' ', pb.Bar(), ' ', pb.ETA()]
        self.timer = pb.ProgressBar(widgets=widgets, maxval=self.n_iter).start()

    def update(self, q=1):
        # Update timer
        self.timer.update(self.iter)
        self.iter += q

    def finish(self):
        # End timer
        self.timer.finish()


""" This function allows to equalize an histogram in order to improve the contrast of our image """


def histogram_equalize(image):

    # Get each color channel of the image
    blue, green, red = cv2.split(image)

    # Equalize the histogram of each color channel of the image
    blue = cv2.equalizeHist(blue)
    green = cv2.equalizeHist(green)
    red = cv2.equalizeHist(red)

    # Merge the color channels together to reconstitute the image
    final_image = cv2.merge((blue, green, red))

    return final_image


""" This function allows to equalize an histogram in order to improve the contrast of our image. 
    However, here, the image is first converted into a gray scale image, then the histogram is equalized """


def enhancing_contrast(image):

    # Convert the original image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Equalize the histogram of the gray scale image
    final_image = cv2.equalizeHist(gray_image)

    return final_image


""" This function allows to increase the saturation of the image """


def increase_saturation(image, increase_value_saturation):

    # Convert the original image into an HSV image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get each channel of the HSV image
    hue, saturation, value = cv2.split(hsv_image)

    # Increase the saturation channel of the HSV image with the value of "increase_value_saturation"
    saturation = saturation + increase_value_saturation

    # Clip each pixel of the saturation channel (that are inferior to 0 or superior to 255) to the values 0 and 255
    saturation = np.clip(saturation, 0, 255)

    # Merge the HSV channels together to reconstitute the image
    final_hsv_image = cv2.merge((hue, saturation, value))

    # Convert the HSV image into a BGR image
    final_image = cv2.cvtColor(final_hsv_image, cv2.COLOR_HSV2BGR)

    return final_image


""" This function allows to increase the brightness of the image """


def increase_brightness(image, increase_value_brightness):

    # Convert the original image into an HSV image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Get each channel of the HSV image
    hue, saturation, value = cv2.split(hsv_image)

    # Compute the limit value of the brightness pixel that can be increase without to reach the value 255
    limit_value = 255 - increase_value_brightness

    # Increase the brightness channel of the HSV image with the value of "increase_value_brightness"
    value[value > limit_value] = 255
    value[value <= limit_value] += increase_value_brightness

    # Merge the HSV channels together to reconstitute the image
    final_hsv_image = cv2.merge((hue, saturation, value))

    # Convert the HSV image into a BGR image
    final_image = cv2.cvtColor(final_hsv_image, cv2.COLOR_HSV2BGR)

    return final_image


""" This function allows processing the image to use it for segmentation """


def image_processing_for_segmentation(image):

    # Convert the BGR (blue, green, red) image into gray scale image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get a mask of the image background (black color for this sample) to increase the differentiation between the
    # grains and the background
    ret, mask = cv2.threshold(gray_image, 40, 255, cv2.THRESH_BINARY)

    # Increase the contrast of the image
    image_with_better_contrast = histogram_equalize(image)

    # Give black color [0, 0, 0] to the background
    image_with_black_background = cv2.bitwise_and(image_with_better_contrast, image_with_better_contrast, mask=mask)

    # Segmentation of the image with the SLIC algorithm
    segmentations = slic(image_with_black_background, n_segments=250, compactness=20, sigma=1, max_iter=10)

    return segmentations, image_with_black_background


def show_label_super_pixels(image, image_with_black_background, ground_truth_image, segmentations, count, dictionary):

    # Initialize variables (lists)
    coordinate_of_super_pixels = []
    raw_data_of_super_pixels = []
    class_of_super_pixels = []
    index_of_super_pixels = []  # Index number of the file name of the image to process
    number_of_pixels = []

    # Loop over the unique segment values
    for (j, segment_values) in enumerate(np.unique(segmentations)):

        # Get the image number of the super pixel "j"
        index_of_super_pixels.append(count)

        # Get the True and False element in the image (matrix dimension - pixels) corresponding to the super pixel "j"
        elements = (segmentations == segment_values)

        # Get the coordinate (x,y - rows,columns) of the image corresponding to the super pixel "j"
        x, y = np.where(elements == True)

        # Store the coordinate (x,y) of the image corresponding to the super pixel "j"
        coordinate_of_super_pixels.append(np.array([x, y]))

        # Store the raw data (value of the BGR channels) of the image corresponding to the super pixel "j"
        raw_data_of_super_pixels.append(image[x, y])

        # Compute the total number of pixels of the super pixel "j"
        total_number_of_pixel = len(image_with_black_background[x, y])

        # Compute the number of black pixels in the super pixel "j"
        b = np.sum(image_with_black_background[x, y] == [0, 0, 0], axis=1)
        number_of_black_pixels = np.sum(b == np.array([3]))

        # Compute the number of the other color pixels
        number_of_other_pixels = total_number_of_pixel - number_of_black_pixels

        if number_of_black_pixels > 1.5 * number_of_other_pixels:

            class_of_super_pixels.append("Background")

        else:

            total_number_of_classes = len(dictionary)

            for k in range(total_number_of_classes):

                b = np.sum(ground_truth_image[x, y] == dictionary[k][1][::-1], axis=1)
                number_of_pixels = np.append(number_of_pixels, np.sum(b == np.array([3])))

                del b

            classes_of_the_super_pixel = ""

            # print(number_of_pixels)
            #
            # # construct a mask for the segment
            #mask = np.zeros(image.shape[:2], dtype="uint8")
            #
            #mask[segmentations == segment_values] = 255
            #
            # # show the masked region
            #cv2.imshow("Applied", cv2.bitwise_and(image, image, mask=mask))
            #cv2.imshow('1', cv2.bitwise_and(ground_truth_image, ground_truth_image, mask=mask))
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

            for k in range(1, 3, 1):

                indices_max = np.argmax(number_of_pixels)

                if number_of_pixels[indices_max] < 20:

                    if k == 1:

                        classes_of_the_super_pixel = classes_of_the_super_pixel + "Background"

                    else:

                        classes_of_the_super_pixel = classes_of_the_super_pixel + ", None"

                else:

                    if k == 1:

                        classes_of_the_super_pixel = classes_of_the_super_pixel + dictionary[indices_max][0]

                    else:

                        classes_of_the_super_pixel = classes_of_the_super_pixel + ", " + dictionary[indices_max][0]

                number_of_pixels[indices_max] = 0

            # print(classes_of_the_super_pixel)
            class_of_super_pixels.append(classes_of_the_super_pixel)

            del number_of_pixels

        number_of_pixels = []

    return coordinate_of_super_pixels, raw_data_of_super_pixels, class_of_super_pixels, index_of_super_pixels


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the file name of the data (image segmentation and super pixel labelling) to use for features extraction
dataFileName = 'Raw_Dataset_V1'

########################################################################################################################
#                                           Get Information and Prepare Data                                           #
########################################################################################################################

print("\nWe are loading all request information. Please wait ...")

########################################################################################################################
informationFileName1 = "C:/Users/Julien/Desktop/Echantillon 1/InformationFile1.pickle"

# Get the folder of the original sub-images
with open(informationFileName1, 'rb') as f:
    pathToSave, pathFilename, smallerImageNumber, rootFilename, windowSizeRow, windowSizeColumn = pickle.load(f)

del pathFilename, smallerImageNumber, rootFilename, windowSizeRow, windowSizeColumn

# Get the pathname where all information about the sample is saved
pathNameOriginalImages = os.path.join(pathToSave, 'Original Image')

# List the files present in the folder [i] of the ground truth images
onlyOriginalImages = [f for f in os.listdir(pathNameOriginalImages)
                      if os.path.join(os.path.join(pathNameOriginalImages, f))]

########################################################################################################################
informationFileName2 = 'C:/Users/Julien/Desktop/Echantillon 1/InformationFile2.pickle'

# Get the legend of the ground truth colors representing minerals
with open(informationFileName2, 'rb') as f:
    dictionary, pathToSave, newPathNameFile = pickle.load(f)

del pathToSave, newPathNameFile

########################################################################################################################
informationFileName3 = 'C:/Users/Julien/Desktop/Echantillon 1/InformationFile3.pickle'

# Get the folder of the ground truth sub-images
with open(informationFileName3, 'rb') as f:
    pathToSave, pathFilename, rootFilename = pickle.load(f)

del pathFilename, rootFilename

# Get the pathname where all information about the sample is saved
pathNameGroundTruthImages = os.path.join(pathToSave, 'Original Ground Truth Image')

# List the files present in the folder [i] of the ground truth images
onlyGroundTruthImages = [f for f in os.listdir(pathNameGroundTruthImages)
                         if os.path.join(os.path.join(pathNameGroundTruthImages, f))]

########################################################################################################################

print("\nThe request information are loaded.")

# Initialize variables
countForSaving = 0
indexNumberForSaving = 0

coordinateSuperPixels = []
rawDataSuperPixels = []
classSuperPixels = []
imageIndex = []

print("\nWe are creating the folder 'Raw Data'. Please wait ...")

newPathName = os.path.join(pathToSave, 'Raw Data')

if os.path.exists(newPathName) == False:

    os.mkdir(newPathName)

print("The folder 'Raw Data' is created.\n")

########################################################################################################################
#                                           Image Segmentation and Labelling                                           #
########################################################################################################################

# Compute the number of smaller images
smallerImagesNumber = len(onlyOriginalImages) - 1

# Initialize the progressbar
sentence = 'Segmentation in superpixels and labelling them of ' + str(smallerImagesNumber) + ' smaller images'
pt = progress_timer(description=sentence, n_iter=smallerImagesNumber)

# Get the current clock to measure the time consuming of the division
start = time.clock()

# for count in range(0, len(onlyOriginalImages), 1):
for count in range(0, smallerImagesNumber, 1):

    # Increment the count of sub-images processed to save data
    countForSaving = countForSaving + 1

    # Load the original and the ground truth sub-images
    i = count + 1  # Increment the count to correspond to the image index
    #print(i)
    originalImage = cv2.imread(os.path.join(pathNameOriginalImages, onlyOriginalImages[i]))
    groundTruthImage = cv2.imread(os.path.join(pathNameGroundTruthImages, onlyGroundTruthImages[i]))

    # Pre-process (image processing + segmentation) the loaded original sub-image
    segmentations, imageWithBlackBackground = image_processing_for_segmentation(originalImage)

    # Give a class name to each super pixel of the original sub-image according to the corresponding ground truth images
    coordinates, rawData, classes, index = show_label_super_pixels(originalImage, imageWithBlackBackground,
                                                                   groundTruthImage, segmentations, i, dictionary)

    # Get the data of the segmentation and labelling
    coordinateSuperPixels = coordinateSuperPixels + coordinates
    rawDataSuperPixels = rawDataSuperPixels + rawData
    classSuperPixels = classSuperPixels + classes
    imageIndex = imageIndex + index

    # Save the data in the new folder
    if countForSaving == 50 or i == len(onlyOriginalImages)-1:

        # Re-initialize the count for saving
        countForSaving = 0

        # Increment the index
        indexNumberForSaving = indexNumberForSaving + 1

        currentDataFilename = dataFileName + "_" + str(indexNumberForSaving) + ".pkl"

        with open(os.path.join(newPathName, currentDataFilename), 'wb') as f:
            pickle.dump([coordinateSuperPixels, rawDataSuperPixels, classSuperPixels, imageIndex], f)

        # Re-initialize variables
        coordinateSuperPixels = []
        rawDataSuperPixels = []
        classSuperPixels = []
        imageIndex = []

    # Update the progressbar
    pt.update()

# Finish the progressbar
pt.finish()

# Print the time consuming of the image division
totalTime = time.clock() - start
print("\nThe total time of this execution is : %f second(s)" % totalTime)

# Save Information
informationFileName = 'InformationFile4.pickle'

with open(os.path.join(pathToSave, informationFileName), 'wb') as f:
    pickle.dump([pathToSave, dataFileName], f)
