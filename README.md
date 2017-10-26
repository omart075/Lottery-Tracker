# Lottery-Tracker
This Python script uses OpenCV to read the lotto numbers off of a lottery ticket.

# Examples
Image:
![Alt text](/sample_imgs/output1.png?raw=true "Script in Action")

Morphology:
![Alt text](/sample_imgs/output2.png?raw=true "Script in Action")

Isolated Numbers:
![Alt text](/sample_imgs/output3.png?raw=true "Script in Action")

Output:
![Alt text](/sample_imgs/output4.png?raw=true "Script in Action")

The array shown holds the numbers read from the ticket.


# Process:
  1. Perform a morphology using a rectangular kernel:
      * Since lines of text can be seen as rectangles, this kernel works best.
      
  2. Apply Sobel gradient:
      * This performs a Gaussian smoothing and differentiation
      
  3. Find contours and filter by size and aspect ratio
      * Only the contours for the numbers should be left
  
  4. Crop image and feed into Tesseract
      * Using dimensions from the contour and binarizing the cropped image makes it easy for Tesseract to read the numbers

# Dependencies:
  1. Install [OpenCV](http://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/)
      
  2. Install numpy:
  
  ```
    pip install numpy
  ```  
  3. Install Pillow
  
  ```
    pip install Pillow
  ```
  4. Install imutils:
  
  ```
    pip install imutils
  ```
  
# Note:
A resizing of the image is done prior to calculations. If the image is too large, the contours are thrown off leading to  inaccurate data. Glare also skews results since OpenCV's algorithms don't perform well with glare. Further testing is still in progress to see how the code performs in different scenarios.
