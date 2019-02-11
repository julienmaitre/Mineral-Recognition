import os

import warnings

import time
import progressbar as pb

import numpy as np
import math
from skimage import io
import cv2
from PIL import Image

import pickle

# Remove the limit of the image decompression allowed by python
Image.MAX_IMAGE_PIXELS = None

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


########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

# Define the pathname+filename of the original image to divide into blocks
# In other terms, select the image to divide into smaller images
pathFilename = "C:/Users/Julien/Desktop/Echantillon 1/Echantillion1.png"

# Define the window size (the size of the smaller image in pixels)
windowSizeRow = 600
windowSizeColumn = 600

########################################################################################################################
#                                         Divide the Image into Smaller Images                                         #
########################################################################################################################

print("\nWe are creating the folder 'Original Image'. Please wait ...")

# Get the pathname where all information about the sample is saved
pathToSave = os.path.split(pathFilename)[0]
newPathName = os.path.join(pathToSave, 'Original Image')

if os.path.exists(newPathName) == False:

    os.mkdir(newPathName)

print("The folder 'Original Image' is created.")

print("\nThe image is loading. Please Wait ...")

# Load the original image to divide into smaller images
# We use the scikit image library because the open cv library cannot open a large image
image = io.imread(pathFilename)

print("The image is loaded.")

# Compute the size of the image to divide into smaller images
height, width, channels = np.shape(image)

print("\nThe image has a size of : ")
print("         width : %d pixels" % width)
print("         height : %d pixels" % height)
print("         channels : %d\n" % channels)

# Split the pathname of the original file image
pathList = os.path.split(pathFilename)

# Transform the folder name of the sample to a root of the smaller image filenames
rootFilename = os.path.splitext(pathList[-1])[0]

# Define a counting index for the saving of the smaller images
counter = 0

# Compute the number of smaller images
smallerImagesNumber = math.floor(width/windowSizeColumn)*math.floor(height/windowSizeRow)

# Initialize the progressbar
sentence = 'For image saving of ' + str(smallerImagesNumber) + ' smaller images'
pt = progress_timer(description=sentence, n_iter=smallerImagesNumber)

# Get the current clock to measure the time consuming of the division
start = time.clock()

# Divide the original image into smaller images
for row in range(0, image.shape[0] - windowSizeRow, windowSizeRow):

    for column in range(0, image.shape[1] - windowSizeColumn, windowSizeColumn):

        # Increment the counting index
        counter = counter + 1

        # Get the smaller image
        window = image[row:row+windowSizeRow, column:column+windowSizeColumn]

        # Construct the filename of the smaller image
        filename = rootFilename + "_" + str(counter) + ".png"
        path = newPathName + "/" + filename

        # Save the smaller image
        cv2.imwrite(path, cv2.cvtColor(window, cv2.COLOR_RGB2BGR))

        # Update the progressbar
        pt.update()

# Finish the progressbar
pt.finish()

# Print the time consuming of the image division
totalTime = time.clock() - start
print("\nThe total time of this execution is : %f second(s)" % totalTime)

# Save Information
informationFileName = 'InformationFile1.pickle'

# Get the number of smaller images that have been generated
smallerImageNumber = counter

with open(os.path.join(pathToSave, informationFileName), 'wb') as f:
    pickle.dump([pathToSave, pathFilename, smallerImageNumber, rootFilename, windowSizeRow, windowSizeColumn], f)
