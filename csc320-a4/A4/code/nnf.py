# CSC320 Spring 2023
# Assignment 4
# (c) Olga (Ge Ya) Xu, Kyros Kutulakos
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
# DO NOT MODIFY THIS FILE 
#

"""
The most important data structure in the PatchMatch algorithm is the one
representing the nearest-neighbour field (NNF). As explained in Section 3 of
the paper, this is a vector field f over the image, ie. a mapping from pixel
coordinates (x,y) to 2D displacements f(x,y). In this module we will represent
a displacement field as a numpy matrix. Given a NxM openCV source image,
the nearest-neighbour field is a matrix of size NxMx2.
"""

# import basic packages
from typing import List, Tuple, Union
import cv2 as cv
import numpy as np
from algorithm import make_coordinates_matrix


def init_NNF(source_image: np.ndarray) -> np.ndarray:
    """
    Generate an NNF that is a just a random displacement field.

    Each pixel (y,x) in the input image is assigned a nearest neighbor that is
    a random pixel in the image. This is done by first generating a pair of
    random numbers for each image pixel and then subtracting those numbers from
    the pixel's coordinates. In this way, the generated random numbers are
    turned into 2D displacements.

    Args:
        source_image:

    Returns:
        A numpy matrix of dimensions im_shape[0] x im_shape[1] x 2
          representing a random displacement field.
    """

    # get the shape of the source image
    im_shape = source_image.shape

    #
    # Generate a matrix f of size (im_shape[0] x im_shape[1] x 2) that
    # assigns random X and Y displacements to each pixel
    #

    # We first generate a matrix of random x coordinates
    x = np.random.randint(low=0, high=im_shape[1],
                          size=(im_shape[0], im_shape[1]))
    # Then we generate a matrix of random y coordinates
    y = np.random.randint(low=0, high=im_shape[0],
                          size=(im_shape[0], im_shape[1]))
    # To create matrix f, we stack those two matrices
    f = np.dstack((y, x))

    #
    # Now we generate a matrix g of size (im_shape[0] x im_shape[1] x 2)
    # such that g(y,x) = [y,x]
    #
    g = make_coordinates_matrix(im_shape)

    # define the NNF to be the difference of these two matrices
    f = f - g

    return f


def create_NNF_image(f: np.ndarray) -> np.ndarray:
    """
    Generate a color openCV image to visualize an NNF f.

    Since f is a vector field, we visualize it with a color image whose
    saturation indicates the magnitude of each vector and whose hue indicates
    the vector's orientation.

    Args:
        f:

    Returns:
        An RGB openCV image that represents the nearest-neighbour field f.

    """

    # square the individual coordinates
    magnitude = np.square(f)
    # sum the coordinates to compute the magnitude
    magnitude = np.sqrt(np.sum(magnitude, axis=2))
    # compute the orientation of each vector
    orientation = np.arccos(f[:, :, 1] / magnitude) / np.pi * 180
    # rescale the orientation to create a hue channel
    hue = np.array(orientation, np.uint8)
    # rescale the magnitude to create a saturation channel
    magnitude = magnitude / np.max(magnitude) * 255
    saturation = np.array(magnitude, np.uint8)
    # create a constant brightness channel
    brightness = np.zeros(magnitude.shape, np.uint8) + 200
    # create the HSV image
    hsv = np.dstack((hue, saturation, brightness))
    # return an RGB image with the specified HSV values
    rgb_image = cv.cvtColor(hsv, cv.COLOR_HSV2RGB)

    return rgb_image

# Generate a color openCV image to visualize an NNF f as a set of
# 2D correspondences: the new image contains the source and target
# image side by side, with 2D vectors drawn between them indicating
# the correspondences for a sparse set of image pixels
#
# We do the rendering using matplotlib to write a temporary image to a file,
# then re-read it as an openCV image


def create_NNF_vectors_image(source: np.ndarray,
                             target: np.ndarray,
                             f: np.ndarray,
                             patch_size: int,
                             server: bool = True,
                             subsampling: int = 50,
                             line_width: float = 0.5,
                             line_color: str = 'k',
                             tmpdir: str = './') -> str:
    """
    Visualize an NNF f as a sparse vector field of 2D correspondences.

    The function outputs a color openCV image that contains the source and
    target images side by side, with 2D vectors drawn between them indicating
    the correspondences for a sparse set of image pixels. The function does
    the rendering using matplotlib. It writes a temporary image to a file,
    then re-reads it as an openCV image.

    Args:
        source:
            Source image.
        target:
            Destination image.
        f:
            A python matrix representing the displacement field.
        patch_size:
            Size of the rectangle to be drawn at the origin of each vector.
        server:
            The backend to use for matplotlib.
        subsampling:
            Density of vectors to be drawn on the images, expressed as pixels
              per vector.
        line_width:
            Line width for drawing the individual vectors.
        line_color:
            Line color for drawing the individual vectors.
        tmpdir:
            Path to store the temporary image being created.

    Returns:
        Path of the image file that holds the vector field visualization.

    """

    import matplotlib.pyplot as plt

    # get the shape of the source image
    im_shape = source.shape

    # if you are using matplotlib on a server
    if server:
        plt.switch_backend('agg')
    import matplotlib.patches as patches

    fig = plt.figure(frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)

    source = cv.cvtColor(source, cv.COLOR_BGR2RGB)
    target = cv.cvtColor(target, cv.COLOR_BGR2RGB)

    # create an image that contains the source and target side by side
    plot_im = np.concatenate((source, target), axis=1)
    ax.imshow(plot_im)

    vector_coords = make_coordinates_matrix(im_shape, step=subsampling)
    vshape = vector_coords.shape
    vector_coords = np.reshape(vector_coords, (vshape[0] * vshape[1], 2))

    for coord in vector_coords:
        rect = patches.Rectangle((coord[1] - patch_size / 2.0, coord[0] - patch_size / 2.0),
                                 patch_size,
                                 patch_size,
                                 linewidth=line_width,
                                 edgecolor=line_color,
                                 facecolor='none')
        ax.add_patch(rect)

        arrow = patches.Arrow(coord[1],
                              coord[0],
                              f[coord[0], coord[1], 1] + im_shape[1],
                              f[coord[0], coord[1], 0],
                              lw=line_width,
                              edgecolor=line_color)
        ax.add_patch(arrow)

    dpi = fig.dpi
    fig.set_size_inches(im_shape[1] * 2 / dpi, im_shape[0] / dpi)
    tmp_image = tmpdir+'/tmpvecs.png' \

    fig.savefig(tmp_image)
    plt.close(fig)
    return tmp_image


def save_NNF(f: np.ndarray, filename: str) -> Tuple[bool, str]:
    """
    Save the nearest-neighbour field (NNF) matrix in numpy file.

    Args:
        f: The NNF represented as a numpy array.
        filename: Filename to save the NNF.

    Returns:
        A tuple whose first element is a boolean indicating success and
         whose second element is the error string (if success=False).
    """
    try:
        np.save('{}'.format(filename), f)
    except IOError as e:
        return False, e
    else:
        return True, None


def load_NNF(filename: str, shape: Union[Tuple,None]):
    """
    Load the nearest-neighbour field from a numpy file.

    Args:
        filename:
            Path to the file containing the NNF numpy matrix.
        shape:
            A tuple (rows, cols) specifying the rows and columns of the NNF.
    Returns:
        A tuple containing (1) True if read was successful and False otherwise,
          (2) the NNF represented as a numpy matrix and (3) the error string if
          the read was not successful.
    """
    try:
        f = np.load(filename)
    except IOError as e:
        return False, None, e
    else:
        if shape is not None:
            if (f.shape[0] != shape[0] or f.shape[1] != shape[1]):
                return False, None, 'NNF has incorrect dimensions'
        return True, f, None
