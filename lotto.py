# USAGE
# python ocr_template_match.py --image images/credit_card_01.png --reference ocr_a_reference.png

# import the necessary packages
import pytesseract
from imutils import contours
import numpy as np
import argparse
import imutils
import cv2
import PIL
from PIL import Image


def resizeImage(image, width, height):
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
	# initialize a rectangular (wider than it is tall) and square
	# structuring kernel
	rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernelVals)
	sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

	# load the input image, resize it, and convert it to grayscale
	# image = Image.open(args["image"])
	# if image.size[0] > 1000:
	# 	image = resizeImage(args["image"], 804, 504)
	# else:
	# 	image = args["image"]


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


###############################################################################
###############################################################################
###############################################################################


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
ap.add_argument("-r", "--reference", required=True,
	help="path to reference OCR-A image")
args = vars(ap.parse_args())


# load the reference OCR-A image from disk, convert it to grayscale,
# and threshold it, such that the digits appear as *white* on a
# *black* background
# and invert it, such that the digits appear as *white* on a *black*
ref = cv2.imread(args["reference"])
ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]

# find contours in the OCR-A image (i.e,. the outlines of the digits)
# sort them from left to right, and initialize a dictionary to map
# digit name to the ROI
refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
digits = {}

# loop over the OCR-A reference contours
for (i, c) in enumerate(refCnts):
	# compute the bounding box for the digit, extract it, and resize
	# it to a fixed size
	(x, y, w, h) = cv2.boundingRect(c)
	roi = ref[y:y + h, x:x + w]
	roi = cv2.resize(roi, (57, 88))

	# update the digits dictionary, mapping the digit name to the ROI
	digits[i] = roi


# Analyze the full ticket to find the numbers
image = cv2.imread(args["image"])
image = imutils.resize(image, width=300)
locs = analyzeImage(image, (20, 3), 1, 150, (10, 23))


# Analyze the cropped image with only the numbers in it
img = cv2.imread(args["image"])
img = imutils.resize(img, width=300)
crop = img[locs[1] - 5:locs[1] + locs[3] + 5, locs[0] - 5:locs[0] + locs[2] + 5]
cv2.imwrite('newCropped.png', crop)

image = cv2.imread('newCropped.png')
locs = analyzeImage(image, (10, 1), 0, 0, (0, 23))


#cv2.rectangle(image, (x[0] - 2, x[1] - 2 ), (x[0]+x[2] + 2, x[1]+x[3] + 2), (255,0,0), 1)
cv2.imshow("Image", image)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#crop = image[x[1] - 2:x[1]+x[3] + 2, x[0] - 2:x[0]+x[2] + 2]
#crop = cv2.morphologyEx(crop, cv2.MORPH_BLACKHAT, rectKernel)
crop = cv2.GaussianBlur(gray, (3,3), 0)
crop = cv2.addWeighted(crop, 1.5, crop, 0.2, 0)
cv2.imwrite('croppedNum.png', crop)

numbersFound = pytesseract.image_to_string(Image.open('croppedNum.png'))
print numbersFound
cv2.waitKey(0)

cleanData(numbersFound)
