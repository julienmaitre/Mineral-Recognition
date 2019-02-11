import os
import progressbar as pb
import numpy as np
import math
from skimage import io
import cv2
from PIL import Image, ImageStat
import warnings
import time
import pickle

import matplotlib.pyplot as plt

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

########################################################################################################################
#                                                   User Parameters                                                    #
########################################################################################################################

with open('C:/Users/Julien/Desktop/Echantillon 1/InformationFile1.pickle', 'rb') as f:
    pathToSave, pathFilename = pickle.load(f)

with open('C:/Users/Julien/Desktop/Echantillon 1/InformationFile2.pickle', 'rb') as f:
    dictionary, pathToSave, newPathNameFile = pickle.load(f)

# Load the original image to divide into smaller images
# We use the scikit image library because the open cv library cannot open a large image
#originalImage = io.imread(pathFilename)

# Load the original image to divide into smaller images
# We use the scikit image library because the open cv library cannot open a large image
image = io.imread(newPathNameFile)

#plt.figure(1)
#plt.imshow(originalImage)
#
#plt.figure(2)
#plt.imshow(image)
#
#plt.show()

srcTri = np.array([(655, 4994), (3157, 101), (5964, 5601)], np.float32)
dstTri = np.array([(3881, 24781), (15910, 2270), (28685, 28387)], np.float32)

########################################################################################################################
#                             Resize the Image to Match with the Sample Image of Minerals                              #
########################################################################################################################

warpMat = cv2.getAffineTransform(srcTri, dstTri)
dst = cv2.warpAffine(image, warpMat, (33653, 34747))

cv2.imwrite(os.path.join(pathToSave, "GroundTruthAlign.png"), cv2.cvtColor(dst, cv2.COLOR_RGB2BGR))