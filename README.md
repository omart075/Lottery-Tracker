# Lottery-Tracker
This Python script uses OpenCV to read the lotto numbers off of a lottery ticket.

# Examples
Input:
![Alt text](/sample_imgs/output1.png?raw=true "Script in Action")

Output:
![Alt text](/sample_imgs/output2.png?raw=true "Script in Action")

Input:
![Alt text](/sample_imgs/output3.png?raw=true "Script in Action")

Output:
![Alt text](/sample_imgs/output4.png?raw=true "Script in Action")

The array shown holds the numbers read from the ticket.


# Process:
  1. Perform a morphology using a rectangular kernel:
      * Since lines of text can be seen as rectangles, this kernel works best.
      
  2. Apply Sobel gradient:
      * This performs a Gaussian smoothing and differentiation
      
  3. Determine which coins were found:
      * Create a dict with each coin (for now: q, d, n, p), its actual size, and value
      * Iterate through the coins found
      * Match each size with sizes in dict
      * For every match, add to a total sum

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
  4. Install scipy
  
  ```
    pip install scipy
  ```  
  5. Install imutils:
  
  ```
    pip install imutils
  ```
  
# Note:
A resizing of the image is done prior to calculations. If the image is too large, the contours are thrown off leading to  inaccurate data. Also, images may be rotated due to how the camera was positioned when picture was taken. This is handled as well, prior to calculations. Further testing is still in progress to see how the code performs in different scenarios.
