# import the necessary packages
import numpy as np
import cv2
import pywt
from skimage import feature

class DescribeTexture:

        def describe_texture(self, img):
                # converting the given image to grayscale and normalizing it
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                imArray =  np.float32(gray_img)   
                imArray /= 255;

                # initializing arrays to store mean, variance and final features
                imMean, imVar, feats = [], [], []
                
                # calculating the dimensions of the regions, for which mean and variance are to be calculated
                # we split the image into 8X8 separate regions
                (h, w) = gray_img.shape[:2]
                (cX, cY) = (int(w * 0.125), int(h * 0.125))

                # iterating over all of the 64 separate regions
                for r in range(8):
                    imRMean, imRVar = [],[]
                    for c in range(8):  
                        tMean, tVar = [],[]

                        # perforimg wavelet decomposition of the region using db1 mode at level 1
                        coeffs=pywt.wavedec2(imArray[r*cY:r*cY+cY,c*cX:c*cX+cX], 'db1', 1)

                        # finding the mean and variance of 4 arrays that resulted from decomposition
                        # coeffs[0] will be a low res copy of image region
                        # coeffs[1][0..2] will be 3 band passed filter results in horizontal, 
                        # vertical and diagonal directions respectively 
                        for i in range(4):

                            # appending the mean and variance values of a region into two vectors
                            if i == 0:   
                                tMean.append(np.mean(coeffs[i]))
                                tVar.append(np.var(coeffs[i]))
                            else:
                                tMean.append(np.mean(coeffs[1][i-1]))
                                tVar.append(np.var(coeffs[1][i-1]))

                        # appending the mean and variance vectors of all regions along the row 
                        imRMean.append(tMean)
                        imRVar.append(tVar)

                    # appending the mean and variance vectors of all rows
                    imMean.append(imRMean)
                    imVar.append(imRVar)
                
                # appending mean and variance vectors into one features vector
                feats.append(imMean)
                feats.append(imVar)

                # flattening the features vector
                feats = np.asarray(feats)
                feats =  cv2.normalize(feats).flatten()

                # returning the features vector / histogram
                return feats

class ColorTree:

        def __init__(self, bins):
                # saving the bin count for our 3D histogram
                self.bins = bins

        def color_tree(self, img):
                # converting the given image to grayscale and initializing
                # an array to save features used to represent the image
                img = img/42
                img = img*42
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                feats = []
 
                # calculating the center of the image
                (h, w) = img.shape[:2]
                (cX, cY) = (int(w * 0.5), int(h * 0.5))

                # partitioning the image into four rectangles/segments (top-left,
                # top-right, bottom-right, bottom-left)
                segs = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                        (0, cX, cY, h)]
 
                # building a mask in ellipse shape to represent the image centre
                (Xaxis, Yaxis) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
                ellipMask = np.zeros(img.shape[:2], dtype = "uint8")
                cv2.ellipse(ellipMask, (cX, cY), (Xaxis, Yaxis), 0, 0, 360, 255, -1)
 
                # iterating over the segments
                for (startX, endX, startY, endY) in segs:
                        # building a mask for each image corner, subtracting the elliptical center
                        cornerMask = np.zeros(img.shape, dtype = "uint8")
                        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
                        cornerMask = cv2.subtract(cornerMask, ellipMask)
 
                        # extracting an histogram of colors from the corner regions, then updating the
                        # features array
                        hist = self.histogram(img, cornerMask)
                        feats.extend(hist)
 
                # extracting an histogram from the elliptical centre, then updating the
                # features array
                hist = self.histogram(img, ellipMask)
                hist.sort
                feats.extend(hist)
 
                # returning the array with features
                return feats

        def histogram(self, img, mask):
                # getting a 3D histogram from the masked part of the image
                # using the given bin count per channel, followed by histogram normalization
                hstgm = cv2.calcHist([img], [0], mask, self.bins,[0, 253])
                hstgm =  cv2.normalize(hstgm).flatten()
 
                # returning the histogram
                return hstgm
		print hstgm

class ColorDescriptor:
        def __init__(self, bins):
                # store the number of bins for the 3D histogram
                self.bins = bins

        def describe(self, image):
                # convert the image to the HSV color space and initialize
                # the features used to quantify the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
                features = []

                # grab the dimensions and compute the center of the image
                (h, w) = image.shape[:2]
                (cX, cY) = (int(w * 0.5), int(h * 0.5))

                # divide the image into four rectangles/segments (top-left,
                # top-right, bottom-right, bottom-left)
                segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                        (0, cX, cY, h)]

                # construct an elliptical mask representing the center of the
                # image
                (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
                ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

                # loop over the segments
                for (startX, endX, startY, endY) in segments:
                        # construct a mask for each corner of the image, subtracting
                        # the elliptical center from it
                        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
                        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
                        cornerMask = cv2.subtract(cornerMask, ellipMask)

                        # extract a color histogram from the image, then update the
                        # feature vector
                        hist = self.histogram(image, cornerMask)
                        features.extend(hist)

                # extract a color histogram from the elliptical region and
                # update the feature vector
                hist = self.histogram(image, ellipMask)
                features.extend(hist)

                # return the feature vector
                return features

        def histogram(self, image, mask):
                # extract a 3D color histogram from the masked region of the
                # image, using the supplied number of bins per channel; then
                # normalize the histogram
                hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                        [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist).flatten()

                # return the histogram
                return hist


class LocalBinaryPatterns:
	def __init__(self, numPoints, radius):
		# store the number of points and radius
		self.numPoints = numPoints
		self.radius = radius
 
	def describe(self, image, eps=1e-7):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                feats = []
		# compute the Local Binary Pattern representation
		# of the image, and then use the LBP representation
		# to build the histogram of patterns
		lbp = feature.local_binary_pattern(image, self.numPoints,
			self.radius, method="uniform")
		(hist, _) = np.histogram(lbp.ravel(),
			bins=np.arange(0, self.numPoints + 3),
			range=(0, self.numPoints + 2))
 
		# normalize the histogram
		hist = hist.astype("float")
		hist /= (hist.sum() + eps)
		feats.append(hist)
 
		# return the histogram of Local Binary Patterns
		return hist
	
class DescribeLAB:
	def __init__(self, bins):
                # store the number of bins for the 3D histogram
                self.bins = bins

        def describe(self, image):
                # convert the image to the HSV color space and initialize
                # the features used to quantify the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                features = []

                # grab the dimensions and compute the center of the image
                (h, w) = image.shape[:2]
                (cX, cY) = (int(w * 0.5), int(h * 0.5))

                # divide the image into four rectangles/segments (top-left,
                # top-right, bottom-right, bottom-left)
                segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                        (0, cX, cY, h)]

                # construct an elliptical mask representing the center of the
                # image
                (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
                ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

                # loop over the segments
                for (startX, endX, startY, endY) in segments:
                        # construct a mask for each corner of the image, subtracting
                        # the elliptical center from it
                        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
                        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
                        cornerMask = cv2.subtract(cornerMask, ellipMask)

                        # extract a color histogram from the image, then update the
                        # feature vector
                        hist = self.histogram(image, cornerMask)
                        features.extend(hist)

                # extract a color histogram from the elliptical region and
                # update the feature vector
                hist = self.histogram(image, ellipMask)
                features.extend(hist)

                # return the feature vector
                return features

        def histogram(self, image, mask):
                # extract a 3D color histogram from the masked region of the
                # image, using the supplied number of bins per channel; then
                # normalize the histogram
                hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                        [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist).flatten()

                # return the histogram
                return hist


		
class DescribeYCrCb:
	def __init__(self, bins):
                # store the number of bins for the 3D histogram
                self.bins = bins

        def describe(self, image):
                # convert the image to the HSV color space and initialize
                # the features used to quantify the image
                image = cv2.cvtColor(image, cv2.COLOR_BGR2YCR_CB)
                features = []

                # grab the dimensions and compute the center of the image
                (h, w) = image.shape[:2]
                (cX, cY) = (int(w * 0.5), int(h * 0.5))

                # divide the image into four rectangles/segments (top-left,
                # top-right, bottom-right, bottom-left)
                segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h),
                        (0, cX, cY, h)]

                # construct an elliptical mask representing the center of the
                # image
                (axesX, axesY) = (int(w * 0.75) / 2, int(h * 0.75) / 2)
                ellipMask = np.zeros(image.shape[:2], dtype = "uint8")
                cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)

                # loop over the segments
                for (startX, endX, startY, endY) in segments:
                        # construct a mask for each corner of the image, subtracting
                        # the elliptical center from it
                        cornerMask = np.zeros(image.shape[:2], dtype = "uint8")
                        cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
                        cornerMask = cv2.subtract(cornerMask, ellipMask)

                        # extract a color histogram from the image, then update the
                        # feature vector
                        hist = self.histogram(image, cornerMask)
                        features.extend(hist)

                # extract a color histogram from the elliptical region and
                # update the feature vector
                hist = self.histogram(image, ellipMask)
                features.extend(hist)

                # return the feature vector
                return features

        def histogram(self, image, mask):
                # extract a 3D color histogram from the masked region of the
                # image, using the supplied number of bins per channel; then
                # normalize the histogram
                hist = cv2.calcHist([image], [0, 1, 2], mask, self.bins,
                        [0, 180, 0, 256, 0, 256])
                hist = cv2.normalize(hist).flatten()

                # return the histogram
                return hist
		
'''class EdgeDetector:
    def describe_edge(self, image):
		feats= []
		
		edges = cv.Canny(image,100,200)
		
			# converting the given image to grayscale
	#	gray= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
		
	# initializing arrays to store final features
		feats = []
	
	# compute the SIFT representation
	# of the image, and then use the SIFT representation
	# to build the histogram of patterns
		edges = cv2.Canny(image,100,200)
		
		feats.append(edges)
	# flattening the features vector
		feats = np.asarray(feats)
		feats = cv2.normalize(feats).flatten()

    # returning the features vector / histogram
		return feats
		
		'''
