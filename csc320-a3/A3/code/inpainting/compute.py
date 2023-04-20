# CSC320 Fall 2022
# Assignment 3
# (c) Kyros Kutulakos
#
#
# UPLOADING THIS CODE TO GITHUB OR OTHER CODE-SHARING SITES IS
# STRICTLY FORBIDDEN.
#
# DISTRIBUTION OF THIS CODE ANY FORM (ELECTRONIC OR OTHERWISE,
# AS-IS, MODIFIED OR IN PART), WITHOUT PRIOR WRITTEN AUTHORIZATION
# BY KYROS KUTULAKOS IS STRICTLY FORBIDDEN. VIOLATION OF THIS
# POLICY WILL BE CONSIDERED AN ACT OF ACADEMIC DISHONESTY.
#
# THE ABOVE STATEMENTS MUST ACCOMPANY ALL VERSIONS OF THIS CODE,
# WHETHER ORIGINAL OR MODIFIED.

#
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

import numpy as np
import cv2 as cv

# File psi.py define the psi class. You will need to 
# take a close look at the methods provided in this class
# as they will be needed for your implementation
from . import psi        

# File copyutils.py contains a set of utility functions
# for copying into an array the image pixels contained in
# a patch. These utilities may make your code a lot simpler
# to write, without having to loop over individual image pixels, etc.
from . import copyutils

#########################################
## PLACE YOUR CODE BETWEEN THESE LINES ##
#########################################

# If you need to import any additional packages
# place them here. Note that the reference 
# implementation does not use any such packages

#########################################


#########################################
#
# Computing the Patch Confidence C(p)
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    confidenceImage:
#         An OpenCV image of type uint8 that contains a confidence 
#         value for every pixel in image I whose color is already known.
#         Instead of storing confidences as floats in the range [0,1], 
#         you should assume confidences are represented as variables of type 
#         uint8, taking values between 0 and 255.
#
# Return value:
#         A scalar containing the confidence computed for the patch center
#

def computeC(psiHatP=None, filledImage=None, confidenceImage=None):
    assert confidenceImage is not None  # shape: H * W grayscale
    assert filledImage is not None  # shape: H * W grayscale
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    # Zhonghan Chen's Implementation
    #########################################
    
    # Replace this dummy value with your own code
    C = 1 

    patchRadius = psiHatP.radius()
    patchCoordR = psiHatP.row()
    patchCoordC = psiHatP.col()

    # Get the confidence window --> to see how many reliable pixels around
    confidenceWindow, validConf = copyutils.getWindow(confidenceImage, (patchCoordR, patchCoordC), patchRadius)

    # Filled Image Window
    filledWindow, validFilled = copyutils.getWindow(filledImage, (patchCoordR, patchCoordC), patchRadius)

    # Compute the area of the patch
    # Assume the area of one pixel is 1
    ValidPatch_Area = np.sum(validFilled)

    # The step is to get the intersection
    validConfwindow = np.where(filledWindow > 0, confidenceWindow, -1)

    C = np.sum(validConfwindow) / ValidPatch_Area

    if C == 1:
        raise Exception('Confidence Compute Failure Error')
    #########################################
    
    return C

#########################################
#
# Computing the max Gradient of a patch on the fill front
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    inpaintedImage:
#         A color OpenCV image of type uint8 that contains the 
#         image I, ie. the image being inpainted
#
# Return values:
#         Dy: The component of the gradient that lies along the 
#             y axis (ie. the vertical axis).
#         Dx: The component of the gradient that lies along the 
#             x axis (ie. the horizontal axis).
#
    
def computeGradient(psiHatP=None, inpaintedImage=None, filledImage=None):
    assert inpaintedImage is not None
    assert filledImage is not None
    assert psiHatP is not None
    
    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    ## Zhonghan Chen's Implementation
    #########################################
    Dy, Dx = None, None
    
    patchRadius = psiHatP.radius()
    patchCoordR = psiHatP.row()
    patchCoordC = psiHatP.col()

    # Obtain the image patches
    imagePatch, validIndicator = copyutils.getWindow(inpaintedImage, (patchCoordR, patchCoordC), patchRadius)
    # Transform the image patch to Grayscale
    imagePatch_gray = cv.cvtColor(imagePatch, cv.COLOR_BGR2GRAY)

    # Setup parameter for sobel operator
    kernel_size = 3
    ddepth = cv.CV_16S
    # Compute the gradient of the patch using cv2
    dx = cv.Sobel(imagePatch_gray, ddepth, 1, 0, ksize=kernel_size)
    dy = cv.Sobel(imagePatch_gray, ddepth, 0, 1, ksize=kernel_size)

    # Obtain the gradient magnitude
    grad_magnitude = np.sqrt((dx * dx) + (dy * dy))

    # Filter out the valid gradient
    if grad_magnitude.shape == validIndicator.shape:
        valid_grad = grad_magnitude * validIndicator
    else:
        raise ValueError('Shape Not Match')

    # Filter out those pixels that are filled
    # Fq refers to the F(q) in the paper
    Fq, _ = copyutils.getWindow(filledImage, (patchCoordR, patchCoordC), patchRadius)

    if valid_grad.shape == Fq.shape:
        grad = valid_grad * (Fq > 0)
    else:
        raise ValueError('Shape Not Match')

    # find index of argmax of those gradients
    i, j = np.unravel_index(grad.argmax(), grad.shape)

    Dx, Dy = dx[i][j], dy[i][j]
    #########################################    
    return Dy, Dx

#########################################
#
# Computing the normal to the fill front at the patch center
#
# Input arguments: 
#    psiHatP: 
#         A member of the PSI class that defines the
#         patch. See file inpainting/psi.py for details
#         on the various methods this class contains.
#         In particular, the class provides a method for
#         accessing the coordinates of the patch center, etc
#    filledImage:
#         An OpenCV image of type uint8 that contains a value of 255
#         for every pixel in image I whose color is known (ie. either
#         a pixel that was not masked initially or a pixel that has
#         already been inpainted), and 0 for all other pixels
#    fillFront:
#         An OpenCV image of type uint8 that whose intensity is 255
#         for all pixels that are currently on the fill front and 0 
#         at all other pixels
#
# Return values:
#         Ny: The component of the normal that lies along the 
#             y axis (ie. the vertical axis).
#         Nx: The component of the normal that lies along the 
#             x axis (ie. the horizontal axis).
#
# Note: if the fill front consists of exactly one pixel (ie. the
#       pixel at the patch center), the fill front is degenerate
#       and has no well-defined normal. In that case, you should
#       set Nx=None and Ny=None
#

def computeNormal(psiHatP=None, filledImage=None, fillFront=None):
    assert filledImage is not None
    assert fillFront is not None
    assert psiHatP is not None

    #########################################
    ## PLACE YOUR CODE BETWEEN THESE LINES ##
    ## Zhonghan Chen's Implementation
    #########################################
    
    # Replace these dummy values with your own code
    Ny = 0
    Nx = 1

    # the first case
    if np.count_nonzero(fillFront == 255) == 1:
         Ny, Nx = None, None
         print('computeNormal: Degenerated Fill Front Appears')
         return Ny, Nx
    

    patchRadius = psiHatP.radius()
    patchCoordR = psiHatP.row()
    patchCoordC = psiHatP.col()
    patchCoord = (patchCoordR, patchCoordC)

    # Get the patch of the current center
    patchFillfront, _ = copyutils.getWindow(fillFront, patchCoord, patchRadius)

    # Setup parameter for sobel operator
    kernel_size = 3
    ddepth = cv.CV_16S
    # Compute the derivative of the entire patch
    dxs = cv.Sobel(patchFillfront, ddepth, 1, 0, ksize=kernel_size)
    dys = cv.Sobel(patchFillfront, ddepth, 0, 1, ksize=kernel_size)

    # Obtain the derivative at the center, w.r.t x and y directions
    raw_dx = dxs[patchRadius][patchRadius]
    raw_dy = dys[patchRadius][patchRadius]

    vec_magnitue = np.sqrt((raw_dx ** 2) + (raw_dy ** 2))

    if vec_magnitue != 0:
        Ny, Nx = raw_dx / vec_magnitue, raw_dy / vec_magnitue
    else:
        Ny, Nx = None, None
        print('computeNormal: Degenerated Fill Front Appears')

    #########################################

    return Ny, Nx
