import os

import warnings

import time
import progressbar as pb

import numpy as np
import math
import cv2
from PIL import Image, ImageStat

import pickle

# Remove the limit of the image decompression allowed by python
Image.MAX_IMAGE_PIXELS = None

# Ignore the warning messages
warnings.filterwarnings("ignore")


# Define progress timer class
class progressTimer:

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


""" This function allows to get the color codes (RGB) in an image """


def getColorCodes(newPath):

    # Initialize variable
    colorCodes = []

    # Read the image related to the path "newPath"
    img = Image.open(newPath)

    # Get all color codes in RGB and the related number of pixels per color code
    codes = img.convert('RGB').getcolors()

    # Get all color codes except the black code (0, 0, 0) and white code (255, 255, 255)
    for i in range(len(codes)):

        if codes[i][1] != (0, 0, 0) and codes[i][1] != (255, 255, 255):

            colorCodes.append(codes[i])

    sortedColorCode = sorted(colorCodes, key=lambda s: s[0], reverse=True)

    return sortedColorCode


# Define the dictionary of the mineral and color code (in RGB) related to this one
dictionary = [("Actinolite", [0, 58, 7]), ("Albite", [188, 205, 210]), ("Andradite", [95, 217, 7]), ("Anorthoclase", [128, 174, 188]),
              ("Apatite", [232, 71, 131]), ("Augite", [118, 60, 23]), ("Corindon", [171, 87, 34]),
              ("Enstatite", [196, 131, 89]), ("Gahnite", [6, 38, 174]), ("Hypersthene", [68, 31, 7]),
              ("Ilmenite", [6, 163, 174]), ("Inconnu1", [226, 226, 226]), ("Inconnu2", [174, 174, 174]),
              ("Inconnu3", [128, 128, 128]), ("Inconnu4", [88, 88, 88]), ("Inconnu5", [66, 66, 66]), ("Kyanite", [58, 158, 62]),
              ("Magnetite", [30, 51, 120]), ("Microcline", [170, 219, 184]), ("Monazite", [11, 20, 96]),
              ("Pumpellyite", [8, 237, 44]), ("Quartz", [119, 167, 167]), ("Rutile", [20, 183, 182]),
              ("Spinel-Cr", [20, 118, 183]), ("Spinel", [44, 107, 149]), ("Titanite", [44, 84, 44]),
              ("Tschermakite", [15, 102, 16]), ("Zircon", [231, 115, 15])]


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# The path names of the right and the left ground truth images provided by the MEM.
# Always, first the right, then the left.
pathName = ["C:/Users/Julien/Desktop/Echantillon 1/Ground Truth/Droite",
            "C:/Users/Julien/Desktop/Echantillon 1/Ground Truth/Gauche"]

########################################################################################################################
#                  Create the Images of the Ground Truth According to the Color Code and the Minerals                  #
########################################################################################################################

print("\nThe creation of the first two ground truth images is in progress. Please Wait ...\n")

# The new file names of the ground truth images modified
newFileNames = ["RightGroundTruth.png", "LeftGroundTruth.png"]

# Initialize variable
newPathNames = []

# Initialize the progressbar
sentence = 'For ground truth parts of ' + str(len(newFileNames)) + ' parts'
pt1 = progressTimer(description=sentence, n_iter=len(newFileNames))

# Get the current clock to measure the time consuming
start = time.clock()

for i in range(0, len(pathName), 1):

    # List the files present in the folder [i] of the ground truth images.
    onlyFiles = [f for f in os.listdir(pathName[i]) if os.path.join(os.path.join(pathName[i], f))]

    # Get only the name of the minerals of the dictionnary.
    resList = [x[0] for x in dictionary]

    # Initialize variable.
    j = 0

    # Update the progressbar
    pt1.update()

    # Initialize the progressbar
    sentence = 'For ground truth files of ' + str(len(onlyFiles)) + ' files'
    pt2 = progressTimer(description=sentence, n_iter=len(onlyFiles))

    # For each ground truth images of the folder [i].
    for f in onlyFiles:

        # Load the ground truth image "f".
        image = cv2.imread(os.path.join(pathName[i], f))

        # Get all color code of the image (except the black and white color).
        colorCodes = getColorCodes(os.path.join(pathName[i], f))

        # Get the first color code of the list corresponding to the majority of the pixels in the image.
        colorCode = list(colorCodes[0][1])

        # Split the name and the extension of the files.
        mineralName = os.path.splitext(f)

        # Get the position of the mineral of the file in the dictionnary.
        index = resList.index(mineralName[0])

        # Change the color of the mineral in the original ground truth image to the new color code defined in the
        # dictionnary.
        image[np.where((image == colorCode[::-1]).all(axis=2))] = list(dictionary[index][1][::-1])

        # Superimpose images
        if j == 0:

            newImage = image

        else:

            newImage = cv2.addWeighted(newImage, 1, image, 1, 0)

        j = j + 1

        # Update the progressbar
        pt2.update()

    # Finish the progressbar
    pt2.finish()

    # Save the new image of the ground truth
    cv2.imwrite(os.path.join(pathName[i], newFileNames[i]), newImage)
    newPathNames.append(os.path.join(pathName[i], newFileNames[i]))

# Finish the progressbar
pt1.finish()

print("\nThe creation of the first two ground truth images is done.")

# Print the time consuming of the image division
totalTime = time.clock() - start
print("The total time of this execution is : %f second(s)" % totalTime)

########################################################################################################################
#                                      Find the Location of the Overlapping Start                                      #
########################################################################################################################

print("\nWe are tring to find the location of the overlapping beginning of the two ground truth images.\n")

# Define the ratio of the overlapping between this two ground truth images.
# This value cannot be set by the user in this version.
overlappingRatio = 0.1

# Load the ground truth of the right part.
rightGTImage = cv2.imread(newPathNames[0])

# Compute the overlapping in pixels.
height, width = rightGTImage.shape[:2]
overlappingPixels = int(width*overlappingRatio)

# Get a template sample of the rightGTImage.
rightTemplateImage = rightGTImage[:, 0:overlappingPixels]

# Load the ground truth of the left part.
leftGTImage = cv2.imread(newPathNames[1])

# Initialize variables.
diff = math.inf
index = -1

# Compute the width of the LeftGTImage.
height, width = leftGTImage.shape[:2]

# Initialize the progressbar.
sentence = 'For the sliding windows of ' + str(width-overlappingPixels) + ' windows'
pt3 = progressTimer(description=sentence, n_iter=width-overlappingPixels)

# Get the current clock to measure the time consuming.
start = time.clock()

# Search the left template image matching with the right template image.
for i in range(0, width-overlappingPixels, 1):

    # Get a template sample of the leftGTImage.
    # This template is sliding from the right to the left by a step of 1 pixel.
    leftTemplateImage = leftGTImage[:, width-overlappingPixels-1-i:width-1-i]

    # Perform the difference between the two images.
    # If the pixel is black, so the difference is perfect, and the pixel of the right template matches with the pixel to
    # the right image.
    newDiffImage = (leftTemplateImage - rightTemplateImage)

    # Compute the sum of all the pixel values of the new image.
    # More the sum is closed to 0, more the two images match.
    newDiffImage = Image.fromarray(newDiffImage)
    stat = ImageStat.Stat(newDiffImage)

    # Update the progressbar.
    pt3.update()

    # Keep in memory the minimum over the iteration.
    if sum(stat.sum) < diff:

        diff = sum(stat.sum)
        index = i

# Finish the progressbar.
pt3.finish()

print("\nWe found the overlapping beginning.")

# Print the time consuming of the image division.
totalTime = time.clock() - start
print("The total time of this execution is : %f second(s)" % totalTime)

########################################################################################################################
#                                             Put Together the Two Images                                              #
########################################################################################################################

with open('C:/Users/Julien/Desktop/Echantillon 1/InformationFile1.pickle', 'rb') as f:
    pathToSave, pathFilename = pickle.load(f)

newImage = np.hstack((leftGTImage[:, 0:width-overlappingPixels-1-index], rightGTImage))

newPathNameFile = os.path.join(pathToSave, "newImage.png")
cv2.imwrite(newPathNameFile, newImage)

# Save Information
informationFileName = 'InformationFile2.pickle'

with open(os.path.join(pathToSave, informationFileName), 'wb') as f:
    pickle.dump([dictionary, pathToSave, newPathNameFile], f)

