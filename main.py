import pytesseract
from imutils import contours
import argparse
import imutils
import cv2
from PIL import Image
import lotto


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
# ref = cv2.imread(args["reference"])
# ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
# ref = cv2.threshold(ref, 10, 255, cv2.THRESH_BINARY_INV)[1]
#
# # find contours in the OCR-A image (i.e,. the outlines of the digits)
# # sort them from left to right, and initialize a dictionary to map
# # digit name to the ROI
# refCnts = cv2.findContours(ref.copy(), cv2.RETR_EXTERNAL,
# 	cv2.CHAIN_APPROX_SIMPLE)
# refCnts = refCnts[0] if imutils.is_cv2() else refCnts[1]
# refCnts = contours.sort_contours(refCnts, method="left-to-right")[0]
# digits = {}
#
# # loop over the OCR-A reference contours
# for (i, c) in enumerate(refCnts):
# 	# compute the bounding box for the digit, extract it, and resize
# 	# it to a fixed size
# 	(x, y, w, h) = cv2.boundingRect(c)
# 	roi = ref[y:y + h, x:x + w]
# 	roi = cv2.resize(roi, (57, 88))
#
# 	# update the digits dictionary, mapping the digit name to the ROI
# 	digits[i] = roi



# Check if the image is rotated from camera, rotate it back and expand to fill entire img
image = Image.open("sample_imgs/" + args["image"])
image.save("sample_imgs/test.jpg")

testImage = Image.open("sample_imgs/test.jpg")
if testImage.size[0] > testImage.size[1]:
	rotated = image.rotate(-90, expand=1)
	rotated.save("sample_imgs/" + args["image"])

#If image is too large, resize it
image = Image.open("sample_imgs/" + args["image"])
if image.size[0] > 600:
	image = resizeImage("sample_imgs/" + args["image"], 504, 804)
else:
	image = "sample_imgs/" + args["image"]


# Analyze the full ticket to find the numbers
image = cv2.imread("sample_imgs/" + args["image"])
image = imutils.resize(image, width=300)
locs = lotto.analyzeImage(image, (20, 3), 1, 150, (10, 21))


# Analyze the cropped image with only the numbers in it
img = cv2.imread("sample_imgs/" + args["image"])
img = imutils.resize(img, width=300)
crop = img[locs[1] - 5:locs[1] + locs[3] + 5, locs[0] - 5:locs[0] + locs[2] + 5]
cv2.imwrite('sample_imgs/newCropped.png', crop)


#Commented code keeps trying different values for the upper height bound since
#some pics use [10, 21] and others use [10, 23]

# foundVals = None
# startVal = 21
# while foundVals is None:
# 	try:
# 		# Analyze the full ticket to find the numbers
# 		image = cv2.imread(args["image"])
# 		image = imutils.resize(image, width=300)
# 		locs = analyzeImage(image, (20, 3), 1, 150, (10, startVal))
#
#
# 		# Analyze the cropped image with only the numbers in it
# 		img = cv2.imread(args["image"])
# 		img = imutils.resize(img, width=300)
# 		crop = img[locs[1] - 5:locs[1] + locs[3] + 5, locs[0] - 5:locs[0] + locs[2] + 5]
# 		cv2.imwrite('newCropped.png', crop)
#
# 		image = cv2.imread('newCropped.png')
# 		locs = analyzeImage(image, (10, 1), 0, 0, (0, startVal))
# 		foundVals = True
# 		#print startVal
# 	except:
# 		#print startVal
# 		startVal += 1



image = cv2.imread('sample_imgs/newCropped.png')
cv2.imshow("Image", image)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.threshold(gray, 0, 255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

blurred = cv2.GaussianBlur(gray, (3,3), 0)
sharp = cv2.addWeighted(blurred, 1.5, blurred, -0.2, 0)
cv2.imwrite('sample_imgs/croppedNum.png', sharp)

numbersFound = pytesseract.image_to_string(Image.open('sample_imgs/croppedNum.png'))
print numbersFound
cv2.waitKey(0)

lotto.cleanData(numbersFound)
