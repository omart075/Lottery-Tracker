from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
from PIL import Image


def resizeImage(image, width, height):
	'''
	Resizes an image and returns the resized image name
	'''
	try:
	    resizingImg = Image.open(image)

	    exif = resizingImg.info['exif']

	    newImg = resizingImg.resize((width, height), Image.ANTIALIAS)
	    newImg.save(image, exif=exif)
	except:
	    resizingImg = Image.open(image)

	    newImg = resizingImg.resize((width, height), Image.ANTIALIAS)
	    newImg.save(image)

	return image


def analyzeImage(image, kernelVals, aspectRatio, contourWidth, contourHeightRange):
	'''
	Analyzes image to find rectangular area that contains lottery numbers
	'''
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelVals)
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))


	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	# apply a tophat (whitehat) morphological operator to find light
	# regions against a dark background (i.e., the credit card numbers)
	tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
	cv2.imshow("tophat", tophat)
	cv2.waitKey(0)

	# compute the Scharr gradient of the tophat image, then scale
	# the rest back into the range [0, 255]
	gradX = cv2.Sobel(tophat, ddepth=cv2.CV_32F, dx=1, dy=0,
		ksize=-1)
	gradX = np.absolute(gradX)
	(minVal, maxVal) = (np.min(gradX), np.max(gradX))
	gradX = (255 * ((gradX - minVal) / (maxVal - minVal)))
	gradX = gradX.astype("uint8")

	# apply a closing operation using the rectangular kernel to help
	# cloes gaps in between credit card number digits, then apply
	# Otsu's thresholding method to binarize the image
	gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
	thresh = cv2.threshold(gradX, 0, 255,
		cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

	# apply a second closing operation to the binary image, again
	# to help close gaps between credit card number regions
	thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)

	cv2.imshow("thresh", thresh)
	cv2.waitKey(0)
	# find contours in the thresholded image, then initialize the
	# list of digit locations
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[0] if imutils.is_cv2() else cnts[1]
	locs = []
	#print(cnts)
	# loop over the contours
	for (i, c) in enumerate(cnts):
		# compute the bounding box of the contour, then use the
		# bounding box coordinates to derive the aspect ratio
		(x, y, w, h) = cv2.boundingRect(c)
		ar = w / float(h)

		# since credit cards used a fixed size fonts with 4 groups
		# of 4 digits, we can prune potential contours based on the
		# aspect ratio
		if ar > aspectRatio:
			# contours can further be pruned on minimum/maximum width
			# and height
			if (w > contourWidth) and (h > contourHeightRange[0] and h < contourHeightRange[1]):
				# append the bounding box region of the digits group
				# to our locations list
				locs.append((x, y, w, h))

	# sort the digit locations from left-to-right, then initialize the
	# list of classified digits
	locs = sorted(locs, key=lambda x:x[0])
	output = []
	#print (locs[0])

	#analyzeDigits()

	#cv2.imshow("Image", image)
	#cv2.waitKey(0)
	return locs[0]


def analyzeDigits():
	'''
	Performs font matching
	'''
	# loop over the 4 groupings of 4 digits
	for (i, (gX, gY, gW, gH)) in enumerate(locs):
		# initialize the list of group digits
		groupOutput = []

		# extract the group ROI of 4 digits from the grayscale image,
		# then apply thresholding to segment the digits from the
		# background of the credit card
		group = gray[gY - 5:gY + gH + 5, gX - 5:gX + gW + 5]
		group = cv2.threshold(group, 0, 255,
			cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

		# detect the contours of each individual digit in the group,
		# then sort the digit contours from left to right
		digitCnts = cv2.findContours(group.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		digitCnts = digitCnts[0] if imutils.is_cv2() else digitCnts[1]
		digitCnts = contours.sort_contours(digitCnts,
			method="left-to-right")[0]

		#print(digitCnts)
		# loop over the digit contours
		for c in digitCnts:
			# compute the bounding box of the individual digit, extract
			# the digit, and resize it to have the same fixed size as
			# the reference OCR-A images
			(x, y, w, h) = cv2.boundingRect(c)
			roi = group[y:y + h, x:x + w]
			roi = cv2.resize(roi, (57, 88))

			# initialize a list of template matching scores
			scores = []

			# loop over the reference digit name and digit ROI
			for (digit, digitROI) in digits.items():
				# apply correlation-based template matching, take the
				# score, and update the scores list
				result = cv2.matchTemplate(roi, digitROI,
					cv2.TM_CCOEFF)
				(_, score, _, _) = cv2.minMaxLoc(result)
				scores.append(score)

			# the classification for the digit ROI will be the reference
			# digit name with the *largest* template matching score
			groupOutput.append(str(np.argmax(scores)))

		# draw the digit classifications around the group
		cv2.rectangle(image, (gX - 5, gY - 5),
			(gX + gW + 5, gY + gH + 5), (0, 0, 255), 2)
		cv2.putText(image, "".join(groupOutput), (gX, gY - 15),
			cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

		# update the output digits list
		output.extend(groupOutput)


def cleanData(numbersFound):
	'''
	Cleans data returned by pytesseract to extract only the lottery numbers as
	an array
	'''
	lottoNumbers = []

	# Get rid of non-digit characters
	for char in numbersFound:
		if not char.isdigit():
			numbersFound = numbersFound.replace(char, '')

	#print numbersFound

	x = 0
	while x < len(numbersFound):
		lottoNumbers.append(int(numbersFound[x:x+2]))
		x += 2
	print lottoNumbers
