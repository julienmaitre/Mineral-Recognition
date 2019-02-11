import os
import time
import progressbar as pb

import numpy as np
from scipy.stats import skew, kurtosis

# from matplotlib import pyplot as plt

from skimage.feature import greycomatrix, greycoprops

import cv2
import warnings
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


""" This function allows processing the image to use it for feature extraction """


def image_processing(image):

    # Convert the BGR (blue, green, red) image to gray color
    image_converted_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Get a mask of the image background (black color for this sample) to increase the differentiation between the
    # grains and the background
    ret, mask = cv2.threshold(image_converted_gray, 50, 255, cv2.THRESH_BINARY)

    # Increase the brightness of the image
    image_brightness = increase_brightness(image, 50)

    # Give black color [0, 0, 0] to the background
    image_with_black_background = cv2.bitwise_and(image_brightness, image_brightness, mask=mask)

    # Increase the saturation of the image
    saturated_image = increase_saturation(image_with_black_background, 50)

    # Increase the contrast of the image
    image_with_better_contrast = histogram_equalize(saturated_image)

    # Close small holes inside the foreground objects, or small black points on the object.
    kernel = np.ones((7, 7), np.uint8)
    closing_image = cv2.morphologyEx(image_with_better_contrast, cv2.MORPH_CLOSE, kernel)

    return closing_image


""" This function allows transforming the image as an HSV image and a Lab image """


def image_transformation(image):

    # Convert the original image into an HSV image
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Convert the original image into an Lab image
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    return hsv_image, lab_image


""" This function allows splitting the image in order to get each channel of the image """


def split_color_image(image, lab_image):

    blue, green, red = cv2.split(image)
    luminance, a, b = cv2.split(lab_image)

    return blue, green, red, luminance, a, b


""" This function applies each image processes defined above to get all images for feature extraction """


def images_for_features(image):

    closing_image = image_processing(image)
    hsv_image, lab_image = image_transformation(image)
    blue, green, red, luminance, a, b = split_color_image(image, lab_image)

    return closing_image, hsv_image, lab_image, blue, green, red, luminance, a, b


""" This function extracts the mean, std, skewness and the kurtosis features for each super pixel """


def mean_std_skewness_kurt(super_pixel):

    mean = np.mean(super_pixel, axis=0)
    std = np.std(super_pixel, axis=0)
    skewness = skew(super_pixel, axis=0)
    kurt = kurtosis(super_pixel, axis=0)

    if np.isnan(mean).all() == True:

        mean = np.array([0, 0, 0])
        std = np.array([0, 0, 0])
        skewness = np.array([0, 0, 0])
        kurt = np.array([0, 0, 0])

    return mean, std, skewness, kurt


def maximum_values_histogram(image, raw_data_of_super_pixels, coordinate_super_pixels, count):

    number = len(raw_data_of_super_pixels)
    max_values_of_his = np.array([])

    # Crop the image
    max_x = max(coordinate_super_pixels[count][0])
    min_x = min(coordinate_super_pixels[count][0])

    max_y = max(coordinate_super_pixels[count][1])
    min_y = min(coordinate_super_pixels[count][1])

    diff_x = 100 - (max_x - min_x)

    if min_x - round(diff_x / 2) < 0:

        x_1 = min_x
        x_2 = max_x + diff_x

    elif max_x + (diff_x - round(diff_x / 2)) > 600:

        x_1 = min_x - diff_x
        x_2 = max_x

    else:

        x_1 = min_x - round(diff_x / 2)
        x_2 = max_x + (diff_x - round(diff_x / 2))

    diff_y = 100 - (max_y - min_y)

    if min_y - round(diff_y / 2) < 0:

        y_1 = min_y
        y_2 = max_y + diff_y

    elif max_y + (diff_y - round(diff_y / 2)) > 600:

        y_1 = min_y - diff_y
        y_2 = max_y

    else:

        y_1 = min_y - round(diff_y / 2)
        y_2 = max_y + (diff_y - round(diff_y / 2))

    image_to_return = image[int(x_1):int(x_2), int(y_1):int(y_2)]

    #glcm = greycomatrix(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), [20], [0], 256, symmetric=True, normed=True)
    #print(glcm)
    # print(greycoprops(glcm, 'dissimilarity')[0, 0])
    # print(greycoprops(glcm, 'correlation')[0, 0])
    # print(greycoprops(glcm, 'contrast')[0, 0])
    # print(greycoprops(glcm, 'homogeneity')[0, 0])
    # print(greycoprops(glcm, 'ASM')[0, 0])
    # print(greycoprops(glcm, 'energy')[0, 0])
    # print("\n")

    glcm = greycomatrix(cv2.cvtColor(image_to_return, cv2.COLOR_BGR2GRAY), [5], [0], 256, symmetric=True, normed=True)
    #print(glcm)
    # print(greycoprops(glcm, 'dissimilarity')[0, 0])
    # print(greycoprops(glcm, 'correlation')[0, 0])
    # print(greycoprops(glcm, 'contrast')[0, 0])
    # print(greycoprops(glcm, 'homogeneity')[0, 0])
    # print(greycoprops(glcm, 'ASM')[0, 0])
    # print(greycoprops(glcm, 'energy')[0, 0])

    # cv2.imshow("Image with", image)
    # cv2.imshow("Super Pixel", image_to_return)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    hist = cv2.calcHist([image], [0], None, [256], [0, 256])

    #plt.plot(hist[1::], color='b')

    idx_1 = np.argmax(hist[1::])
    max_value_blue_1 = hist[idx_1 + 1] / number

    hist[idx_1+1] = 0

    idx_2 = np.argmax(hist[1::])
    max_value_blue_2 = hist[idx_2 + 1] / number

    max_values_of_his = np.append(max_values_of_his, max_value_blue_1)
    max_values_of_his = np.append(max_values_of_his, idx_1)
    max_values_of_his = np.append(max_values_of_his, max_value_blue_2)
    max_values_of_his = np.append(max_values_of_his, idx_2)

    hist = cv2.calcHist([image], [1], None, [256], [0, 256])

    #plt.plot(hist[1::], color='g')

    idx_1 = np.argmax(hist[1::])
    max_value_green_1 = hist[idx_1 + 1] / number

    hist[idx_1+1] = 0

    idx_2 = np.argmax(hist[1::])
    max_value_green_2 = hist[idx_2 + 1] / number

    max_values_of_his = np.append(max_values_of_his, max_value_green_1)
    max_values_of_his = np.append(max_values_of_his, idx_1)
    max_values_of_his = np.append(max_values_of_his, max_value_green_2)
    max_values_of_his = np.append(max_values_of_his, idx_2)

    hist = cv2.calcHist([image], [2], None, [256], [0, 256])

    #plt.plot(hist[1::], color='r')
    #plt.show()

    idx_1 = np.argmax(hist[1::])
    max_value_red_1 = hist[idx_1 + 1] / number

    hist[idx_1+1] = 0

    idx_2 = np.argmax(hist[1::])
    max_value_red_2 = hist[idx_2 + 1] / number

    max_values_of_his = np.append(max_values_of_his, max_value_red_1)
    max_values_of_his = np.append(max_values_of_his, idx_1)
    max_values_of_his = np.append(max_values_of_his, max_value_red_2)
    max_values_of_his = np.append(max_values_of_his, idx_2)

    #max_values_of_his = np.append(max_values_of_his, idx_2)

    return max_values_of_his


def degree_of_luminance(blue, green, red):

    luminance_factor = np.mean(0.299*red + 0.587*green + 0.114*blue, axis=0)

    return luminance_factor


""" This function extracts the super pixel from images """


def get_super_pixel(count, raw_data_super_pixels, coordinate_super_pixels, improving_colors_image, hsv_image,
                    lab_image, blue_channel, green_channel, red_channel, luminance_image):

    original_super_pixel = raw_data_super_pixels[count]

    hsv_super_pixel = hsv_image[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]]

    lab_super_pixel = lab_image[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]]

    improving_colors_super_pixel = improving_colors_image[coordinate_super_pixels[count][0],
                                                          coordinate_super_pixels[count][1]]

    luminance_super_pixel = luminance_image[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]]

    blue_super_pixel = blue_channel[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]]
    green_super_pixel = green_channel[coordinateSuperPixels[count][0], coordinate_super_pixels[count][1]]
    red_super_pixel = red_channel[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]]

    return original_super_pixel, hsv_super_pixel, lab_super_pixel, improving_colors_super_pixel, \
           luminance_super_pixel, blue_super_pixel, green_super_pixel, red_super_pixel


""" This function extracts features from super pixel """


def feature_vector_extraction(dataset_of_attributes, raw_data_super_pixels, coordinate_super_pixels, count,
                              original_image, original_superpixel, hsv_super_pixel, lab_super_pixel,
                              improving_colors_image, luminance_super_pixel, blue_super_pixel, green_super_pixel,
                              red_super_pixel):

    # Compute the mean, the standard deviation, the skewness and the kurtosis of the original super pixels (raw data)
    # "count"
    mean, std, skewness, kurt = mean_std_skewness_kurt(original_superpixel)
    dataset_of_attributes.append(np.array([mean, std, skewness, kurt]))

    # Compute the mean, the standard deviation, the skewness and the kurtosis of the HSV super pixels "count"
    mean, std, skewness, kurt = mean_std_skewness_kurt(hsv_super_pixel)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], np.array([mean, std, skewness, kurt]))

    # Compute the mean, the standard deviation, the skewness and the kurtosis of the Lab super pixels "count"
    # Only the L channel of the Lab image is used as features.
    mean, std, skewness, kurt = mean_std_skewness_kurt(lab_super_pixel)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], np.array([mean[0], std[0], skewness[0], kurt[0]]))

    # Compute the mean, the standard deviation, the skewness and the kurtosis of the improving colors of super pixels
    # "count"
    mean, std, skewness, kurt = mean_std_skewness_kurt(improving_colors_image)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], np.array([mean, std, skewness, kurt]))

    # Compute the mean, the standard deviation, the skewness and the kurtosis of the luminance super pixels "count"
    mean, std, skewness, kurt = mean_std_skewness_kurt(luminance_super_pixel)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], np.array([mean, std, skewness, kurt]))

    # Compute the degree of luminance of the improving colors of super pixels "count"
    luminance_factor = degree_of_luminance(blue_super_pixel, green_super_pixel, red_super_pixel)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], np.array(luminance_factor))

    # Create a mask to isolate the super pixel on the original image
    mask = np.zeros(original_image.shape[:2], dtype="uint8")
    mask[coordinate_super_pixels[count][0], coordinate_super_pixels[count][1]] = 255
    img = cv2.bitwise_and(original_image, original_image, mask=mask)

    # Extract the maximum peak
    list_of_hist_values = maximum_values_histogram(img, raw_data_super_pixels[count], coordinate_super_pixels, count)
    dataset_of_attributes[count] = np.append(dataset_of_attributes[count], list_of_hist_values)

    return dataset_of_attributes


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the file name of the dataset to use for classification
dataFileName = 'Dataset_V1'

########################################################################################################################
#                                           Get Information and Prepare Data                                           #
########################################################################################################################

print("\nWe are loading all request information. Please wait ...")

########################################################################################################################
informationFileName1 = "C:/Users/Julien/Desktop/Echantillon 1/InformationFile1.pickle"

# Get the folder of the original sub-images
with open(informationFileName1, 'rb') as f:
    pathToSave, pathFilename, smallerImageNumber, rootFilename, windowSizeRow, windowSizeColumn = pickle.load(f)

del pathFilename, smallerImageNumber, windowSizeRow, windowSizeColumn

# Get the pathname where all information about the sample is saved
pathNameOriginalImages = os.path.join(pathToSave, 'Original Image')

########################################################################################################################
informationFileName2 = "C:/Users/Julien/Desktop/Echantillon 1/InformationFile4.pickle"

# Get the folder of the original sub-images
with open(informationFileName2, 'rb') as f:
    pathToSave, dataFileName = pickle.load(f)

# Get the pathname where all information about the sample is saved
pathNameRawDataset = os.path.join(pathToSave, 'Raw Data')

# List the files present in the folder [i] of the raw dataset
onlyRawDatasetFile = [f for f in os.listdir(pathNameRawDataset)
                      if os.path.join(os.path.join(pathNameRawDataset, f))]

########################################################################################################################

print("The request information are loaded.\n")

# Initialize variables
datasetAttributes = []
dataset = []  # global dataset of feature extracted from each super pixel of each sub-image
classes = []  # classes corresponding to each observation of the global dataset

########################################################################################################################
#                                           Image Segmentation and Labelling                                           #
########################################################################################################################

# Compute the number of raw dataset files
rawDataFileNumber = len(onlyRawDatasetFile)

# Initialize the progressbar
sentence = 'Feature extraction of ' + str(rawDataFileNumber) + ' raw datasets (composed of 50 images)'
pt = progress_timer(description=sentence, n_iter=rawDataFileNumber)

# Get the current clock to measure the time consuming of the division
start = time.clock()

for count in range(9, rawDataFileNumber, 1):

    # Get the super pixel data of sub-images
    with open(os.path.join(pathNameRawDataset, onlyRawDatasetFile[count]), 'rb') as f:
        coordinateSuperPixels, rawDataSuperPixels, classSuperPixels, imageIndex = pickle.load(f)

    # Define the index of the first sub-image
    i = imageIndex[0]

    # Update the progressbar
    pt.update()

    # Load the first sub-image in the corresponding folder
    originalImage = cv2.imread(os.path.join(pathNameOriginalImages, rootFilename + "_" + str(i) + ".png"))

    # Process the sub-image to get sub-images for the feature extraction
    improvingColorsImage, hsvImage, labImage, blue, green, red, luminanceImage, a, b = \
        images_for_features(originalImage)

    for counter in range(len(coordinateSuperPixels)):

        # Get the index of the sub-image for the super-pixel "counter"
        index = imageIndex[counter]

        # Condition to load the next sub-image
        if i < index:

            # Increment i by +1
            i = i+1

            # Load the sub-image "i"
            originalImage = cv2.imread(os.path.join(pathNameOriginalImages, rootFilename + "_" + str(i) + ".png"))

            # Process the sub-image to get sub-images for the feature extraction
            improvingColorsImage, hsvImage, labImage, blue, green, red, luminanceImage, a, b = \
                images_for_features(originalImage)

        # Get the super pixel "counter" from the sub-images "i"
        originalSuperpixel, hsvSuperpixel, labSuperpixel, improvingColorsSuperpixel, luminanceSuperpixel, \
        blueSuperpixel, greenSuperpixel, redSuperpixel = get_super_pixel(counter, rawDataSuperPixels,
                                                                         coordinateSuperPixels, improvingColorsImage,
                                                                         hsvImage, labImage, blue, green, red,
                                                                         luminanceImage)

        # Extract features on the super pixel "counter"
        datasetAttributes = feature_vector_extraction(datasetAttributes, rawDataSuperPixels, coordinateSuperPixels,
                                                      counter, originalImage, originalSuperpixel, hsvSuperpixel,
                                                      labSuperpixel, improvingColorsSuperpixel, luminanceSuperpixel,
                                                      blueSuperpixel, greenSuperpixel, redSuperpixel)

    # Join the data (features + classes) to the rest of the generated data until now
    classes = classes + classSuperPixels
    dataset = dataset + datasetAttributes

    # Re-initialize variable
    datasetAttributes = []

# Finish the progressbar
pt.finish()

# Print the time consuming of the image division
totalTime = time.clock() - start
print("\nThe total time of this execution is : %f second(s)" % totalTime)

with open('C:/Users/Julien/Desktop/Echantillon 1/InformationFile5.pickle', 'wb') as f:
    pickle.dump([dataset, classes], f)
