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
# DO NOT MODIFY THIS FILE ANYWHERE EXCEPT WHERE INDICATED
#

# Import basic packages.
from typing import List, Union, Tuple, Dict
import numpy as np
#
# Basic numpy configuration
#

# Set random seed.
np.random.seed(seed=131)
# Ignore division-by-zero warning.
np.seterr(divide='ignore', invalid='ignore')


def propagation_and_random_search(
        source_patches: np.ndarray,
        target_patches: np.ndarray,
        f: np.ndarray,
        alpha: float,
        w: int,
        propagation_enabled: bool,
        random_enabled: bool,
        odd_iteration: bool,
        best_D: Union[np.ndarray, None] = None,
        global_vars: Union[Dict, None] = None) -> \
            Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Basic PatchMatch loop.

    This function implements the basic loop of the PatchMatch algorithm, as
    explained in Section 3.2 of the paper. The function takes an NNF f as
    input, performs propagation and random search, and returns an updated NNF.

    Args:
        source_patches:
            A numpy matrix holding the patches of the color source image,
            as computed by the make_patch_matrix() function in this module.
            
            For an NxM source image and patches of width P, the matrix has
            dimensions NxMxCx(P^2) where C is the number of color channels
            and P^2 is the total number of pixels in the patch.  

            Shape: N * M * C * (P^2)
            
            For your purposes, you may assume that source_patches[i,j,c,:]
            gives you the list of intensities for color channel c of
            all pixels in the patch centered at pixel [i,j]. Note that patches
            that go beyond the image border will contain NaN values for
            all patch pixels that fall outside the source image.
        
        target_patches:
            The matrix holding the patches of the target image, represented
              exactly like the source_patches argument.

        f:
            The current nearest-neighbour field.
            shape: N x M x 2

        alpha:
            Algorithm parameter, as explained in Section 3 and Eq.(1).
        w:
            Algorithm parameter, as explained in Section 3 and Eq.(1).
        propagation_enabled:
            If true, propagation should be performed. Use this flag for
              debugging purposes, to see how your
              algorithm performs with (or without) this step.
        random_enabled:
            If true, random search should be performed. Use this flag for
              debugging purposes, to see how your
              algorithm performs with (or without) this step.
        odd_iteration:
            True if and only if this is an odd-numbered iteration.
              As explained in Section 3.2 of the paper, the algorithm
              behaves differently in odd and even iterations and this
              parameter controls this behavior.
        best_D:
            And NxM matrix whose element [i,j] is the similarity score between
              patch [i,j] in the source and its best-matching patch in the
              target. Use this matrix to check if you have found a better
              match to [i,j] in the current PatchMatch iteration.
        global_vars:
            (optional) if you want your function to use any global variables,
              return them in this argument and they will be stored in the
              PatchMatch data structure.

    Returns:
        A tuple containing (1) the updated NNF, (2) the updated similarity
          scores for the best-matching patches in the target, and (3)
          optionally, if you want your function to use any global variables,
          return them in this argument and they will be stored in the
          PatchMatch data structure.
    """
    new_f = f.copy()
    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    # Zhonghan Chen's Implementation
    #############################################    
    shapeInfo = source_patches.shape
    h, w = shapeInfo[0], shapeInfo[1] 
    best_D = np.zeros((h, w))

    # ui =v_0+w * a^{i}R_{i}
    # compute a^{i}R_{i} part of the formula for further use
    ep, temp_coef = 0, []
    while w * (alpha ** ep) >= 1:
        temp_coef.append(w * (alpha ** ep))
        ep += 1
    r = np.random.uniform(low=-1.0, high=1.0, size=(ep, 2))
    temp_coef = np.array(temp_coef).reshape((ep, 1))
    addon = temp_coef * r

    if odd_iteration:

        for i in range(h):

            for j in range(w):
                
                if not propagation_enabled:

                    sample_pool = dict()

                    # get the target coordinate of itself and its neighnour
                    t0_coord = np.array([i, j]) + f[i, j]

                    curr_patch = source_patches[i, j]  # compute the four channel patch

                    if t0_coord[0] < h and t0_coord[1] < w:  #and t0_coord[0] >=0 and t0_coord[1] >= 0:
                        target_patch = target_patches[t0_coord[0], t0_coord[1]]  # the target patch
                        # compute the error:
                        diff0 = (target_patch - curr_patch)
                        diff0 = diff0[np.where(~np.isnan(diff0))]
                        dv0 = np.sum(diff0 * diff0)
                        sample_pool[tuple(f[i, j])] = dv0
                    

                    #if t1_coord is not None:
                    if i - 1 >= 0:
                        t1_coord = np.array([i - 1, j]) + f[i - 1, j]
                        if t1_coord[0] < h and t1_coord[1] < w:
                            sample1_patch = target_patches[t1_coord[0], t1_coord[1]]
                            diff1 = (sample1_patch - curr_patch)
                            diff1 = diff1[np.where(~np.isnan(diff1))]
                            dv1 = np.sum(diff1 * diff1)
                            sample_pool[tuple(f[i - 1, j])] = dv1


                    #if t2_coord is not None:
                    if j - 1 >= 0:
                        t2_coord = np.array([i, j - 1]) + f[i, j - 1]
                        if t2_coord[0] < h and t2_coord[1] < w:
                            sample2_patch = target_patches[t2_coord[0], t2_coord[1]]
                            diff2 = (sample2_patch - curr_patch)
                            diff2 = diff2[np.where(~np.isnan(diff2))]
                            dv2 = np.sum(diff2 * diff2)
                            sample_pool[tuple(f[i, j - 1])] = dv2

                    if len(sample_pool) != 0:
                        raw_offset = min(sample_pool, key=sample_pool.get)
                        best_D[i, j] = sample_pool[raw_offset]
                        new_f[i, j] = np.array([raw_offset[0], raw_offset[1]])

                if not random_enabled:

                    offset_vcandidate = f[i, j] + addon

                    tar_coord = np.round(np.array([i, j]) + offset_vcandidate).astype(int)
                    tar_coord[:, 0] = np.clip(tar_coord[:, 0], a_min=0, a_max=h-1)
                    tar_coord[:, 1] = np.clip(tar_coord[:, 1], a_min=0, a_max=w-1)

                    tar_patch = target_patches[tar_coord[:, 0], tar_coord[:, 1]]

                    vdiff = source_patches[i, j] - tar_patch

                    # if i use: vdiff[np.where(~np.isnan(vdiff))], some pixel would be pretty weird
                    #vdiff[np.where(~np.isnan(vdiff))]
                    vdiff[np.isnan(vdiff)] = 255

                    #all_channel_sum = np.sum(vdiff * vdiff, axis=2)  # (9 x 4)
                    finalDiff = np.sum(np.sum(vdiff * vdiff, axis=2), axis=1)

                    minimumD = finalDiff[np.argmin(finalDiff)]

                    if best_D[i, j] > minimumD:
                        best_D[i, j] = minimumD
                        new_f[i, j] = offset_vcandidate[np.argmin(finalDiff)]
    
    # refer to the even iterations
    else:

        for i in range(h - 1, 0, -1):

            for j in range(w - 1, 0, -1):

                if not propagation_enabled:

                    sample_pool = dict()

                    # get the target coordinate of itself and its neighnour
                    t0_coord = np.array([i, j]) + f[i, j]

                    curr_patch = source_patches[i, j]  # compute the four channel patch

                    if t0_coord[0] < h and t0_coord[1] < w:
                        target_patch = target_patches[t0_coord[0], t0_coord[1]]  # the target patch
                        # compute the error:
                        diff0 = (target_patch - curr_patch)
                        diff0 = diff0[np.where(~np.isnan(diff0))]
                        dv0 = np.sum(diff0 * diff0)
                        sample_pool[tuple(f[i, j])] = dv0
                    
                    
                    if i + 1 < h:
                        t1_coord = np.array([i + 1, j]) + f[i + 1, j]
                        if t1_coord[0] < h and t1_coord[1] < w:
                            t1_coord = np.array([i + 1, j]) + f[i + 1, j]
                            sample1_patch = target_patches[t1_coord[0], t1_coord[1]]
                            diff1 = (sample1_patch - curr_patch)
                            diff1 = diff1[np.where(~np.isnan(diff1))]
                            dv1 = np.sum(diff1 * diff1)
                            sample_pool[tuple(f[i + 1, j])] = dv1


                    if j + 1 < w:
                        t2_coord = np.array([i, j + 1]) + f[i, j + 1]
                        if t2_coord[0] < h and t2_coord[1] < w:
                            sample2_patch = target_patches[t2_coord[0], t2_coord[1]]
                            diff2 = (sample2_patch - curr_patch)
                            diff2 = diff2[np.where(~np.isnan(diff2))]
                            dv2 = np.sum(diff2 * diff2)
                            sample_pool[tuple(f[i, j + 1])] = dv2
                    
                    if len(sample_pool) != 0:
                        raw_offset = min(sample_pool, key=sample_pool.get)
                        best_D[i, j] = sample_pool[raw_offset]
                        new_f[i, j] = np.array([raw_offset[0], raw_offset[1]])
                
                if not random_enabled:

                    offset_vcandidate = f[i, j] + addon

                    tar_coord = np.round(np.array([i, j]) + offset_vcandidate).astype(int)

                    tar_coord[:, 0] = np.clip(tar_coord[:, 0], a_min=0, a_max=h-1)
                    tar_coord[:, 1] = np.clip(tar_coord[:, 1], a_min=0, a_max=w-1)

                    tar_patch = target_patches[tar_coord[:, 0], tar_coord[:, 1]]

                    vdiff = source_patches[i, j] - tar_patch

                    # if i use: vdiff[np.where(~np.isnan(vdiff))], some pixel would be pretty weird
                    vdiff[np.isnan(vdiff)] = 255

                    #all_channel_sum = np.sum(vdiff * vdiff, axis=2)
                    finalDiff = np.sum(np.sum(vdiff * vdiff, axis=2), axis=1)

                    minimumD = finalDiff[np.argmin(finalDiff)]

                    if best_D[i, j] > minimumD:
                        best_D[i, j] = minimumD
                        new_f[i, j] = offset_vcandidate[np.argmin(finalDiff)]

    return new_f, best_D, global_vars


def reconstruct_source_from_target(target: np.ndarray,
                                   f: np.ndarray) -> np.ndarray:
    """
    Reconstruct a source image using pixels from a target image.

    This function uses a computed NNF f(x,y) to reconstruct the source image
    using pixels from the target image.  To reconstruct the source, the
    function copies to pixel (x,y) of the source the color of
    pixel (x,y)+f(x,y) of the target.

    The goal of this routine is to demonstrate the quality of the
    computed NNF f. Specifically, if patch (x,y)+f(x,y) in the target image
    is indeed very similar to patch (x,y) in the source, then copying the
    color of target pixel (x,y)+f(x,y) to the source pixel (x,y) should not
    change the source image appreciably. If the NNF is not very high
    quality, however, the reconstruction of source image
    will not be very good.

    You should use matrix/vector operations to avoid looping over pixels,
    as this would be very inefficient.

    Args:
        target:
            The target image that was used as input to PatchMatch.
        f:
            A nearest-neighbor field the algorithm computed.
    Returns:
        An openCV image that has the same shape as the source image.
    """
    rec_source = None

    #############################################
    ###  PLACE YOUR CODE BETWEEN THESE LINES  ###
    # Zhonghan Chen's Implementation
    #############################################
    h, w = target.shape[0], target.shape[1]

    # construct the initial matrix
    image_idx = make_coordinates_matrix(target.shape)
    target_idx = image_idx + f  # (x,y)+f(x,y)

    # make all part legal
    target_idx_h = np.clip(target_idx[:, :, 0], 0, h - 1)
    target_idx_w = np.clip(target_idx[:, :, 1], 0, w - 1)

    # extract from the target
    rec_source = target[target_idx_h, target_idx_w]
    #############################################

    return rec_source


def make_patch_matrix(im: np.ndarray, patch_size: int) -> np.ndarray:
    """
    PatchMatch helper function.

    This function is called by the initialized_algorithm() method of the
    PatchMatch class. It takes an NxM image with C color channels and a patch
    size P and returns a matrix of size NxMxCxP^2 that contains, for each
    pixel [i,j] in the image, the pixels in the patch centered at [i,j].

    You should study this function very carefully to understand precisely
    how pixel data are organized, and how patches that extend beyond
    the image border are handled.

    Args:
        im:
            A image of size NxM.
        patch_size:
            The patch size.

    Returns:
        A numpy matrix that holds all patches in the image in vectorized form.
    """
    phalf = patch_size // 2
    # create an image that is padded with patch_size/2 pixels on all sides
    # whose values are NaN outside the original image
    padded_shape = im.shape[0] + patch_size - 1, \
                   im.shape[1] + patch_size - 1, \
                   im.shape[2]
    padded_im = np.zeros(padded_shape) * np.NaN
    padded_im[phalf:(im.shape[0] + phalf), phalf:(im.shape[1] + phalf), :] = im

    # Now create the matrix that will hold the vectorized patch of each pixel.
    # If the original image had NxM pixels, this matrix will have
    # NxMx(patch_size*patch_size) pixels
    patch_matrix_shape = im.shape[0], im.shape[1], im.shape[2], patch_size ** 2
    patch_matrix = np.zeros(patch_matrix_shape) * np.NaN
    for i in range(patch_size):
        for j in range(patch_size):
            patch_matrix[:, :, :, i * patch_size + j] = \
                padded_im[i:(i + im.shape[0]), j:(j + im.shape[1]), :]

    return patch_matrix


def make_coordinates_matrix(im_shape: Tuple, step: int = 1) -> np.ndarray:
    """
    PatchMatch helper function.

    This function returns a matrix g of size (im_shape[0] x im_shape[1] x 2)
    such that g(y,x) = [y,x].

    Pay attention to this function as it shows how to perform these types
    of operations in a vectorized manner, without resorting to loops.

    Args:
        im_shape:
            A tuple that specifies the size of the input images.
        step:
            (optional) If specified, the function returns a matrix that is
              step times smaller than the full image in each dimension.
    Returns:
        A numpy matrix holding the function g.
    """
    range_x = np.arange(0, im_shape[1], step)
    range_y = np.arange(0, im_shape[0], step)
    axis_x = np.repeat(range_x[np.newaxis, ...], len(range_y), axis=0)
    axis_y = np.repeat(range_y[..., np.newaxis], len(range_x), axis=1)

    return np.dstack((axis_y, axis_x))
