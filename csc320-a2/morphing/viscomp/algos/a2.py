# CSC320 Spring 2023
# Assignment 2
# (c) Kyros Kutulakos, Towaki Takikawa, Esther Lin
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

import sys
import numpy as np
import cv2
import viscomp.ops.image as img_ops

def run_a2_algo(source_image, destination_image, source_morph_lines, destination_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, interpolate, param_t, vectorize):
    """Run the entire A2 algorithm.

    In the A2 backward mapping algorithm, there is no linear transformation computed beforehand
    like with A1's homographies. Instead, you directly compute the source_coords based on the 
    Beier-Neely algorithm's formulation.

    For more information about this algorithm (especially with respect to what some of the params
    actually are), consult the Beier-Neely paper provided in class (probably in the course Dropbox).

    Args:
        source_image (np.ndarray): The source image of shape [H, W, 4]
        destination_image (np.ndarray): The destination image of shape [H, W, 4]
        source_morph_lines (np.ndarray): [N, 2] tensor of coordinates for lines in 
                                         normalized [-1, 1] space.
        destination_morph_lines (np.ndarray): [N, 2] tensor of coordinates for lines in
                                              normalized [-1, 1] space.
        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.
        supersampling (int): The patch size for supersampling.
        bilinear (bool): If True, will use bilinear interpolation on the pixels.
        interpolate (bool): If True, will interpolate between the two images.
        param_t (float): The interpolation parameter between [0, 1]. If interpolate is False, 
                         will only interpolate the arrows (and not the images).
        vectorize (bool): If True, will use the vectorized version of the code (optional).

    Returns:
        (np.ndarray): Written out image of shape [H, W, 4]
    """
    interpolated_morph_lines = (source_morph_lines * (1.0 - param_t)) + (destination_morph_lines * param_t)
    if interpolate:
        output_image_0 = backward_mapping(source_image, destination_image, 
                source_morph_lines, interpolated_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, vectorize)
        output_image_1 = backward_mapping(destination_image, source_image, 
                destination_morph_lines, interpolated_morph_lines,
                param_a, param_b, param_p, supersampling, bilinear, vectorize)
        output_buffer = (output_image_0 * (1.0 - param_t)) + (output_image_1 * param_t)
    else:
        output_buffer = backward_mapping(source_image, destination_image, 
                                         source_morph_lines, interpolated_morph_lines,
                                         param_a, param_b, param_p, supersampling, bilinear, vectorize)
    return output_buffer

# student_implementation (Zhonghan Chen, 1005770541)

def perp(vec):
    """Find the perpendicular vector to the input vector.

    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): Perpendicular vectors of size [N, 2].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################

    """
    My implementation is based on the simple idea:
    vec = [a, b]
    perp_vec = [-b, a]
    """

    N, T = vec.shape[0], vec.shape[1]

    output = np.zeros((T, N))

    x_coord, y_coord = vec.T[0], vec.T[1]

    output[0], output[1] = y_coord, -x_coord

    return output.T
    #################################
    ######### DO NOT MODIFY #########
    #################################


def norm(vec):
    """Find the norm (length) of the input vectors.

    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): An array of vector norms of shape [N].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    output = np.linalg.norm(vec, axis=1)  # Note: axis=1 is important here
    return output
    #################################
    ######### DO NOT MODIFY #########
    #################################


def normalize(vec):
    """Normalize vectors to unit vectors of length 1.

    Hint: Use the norm function you implemented!
    
    Args:
        vec (np.array): Vectors of size [N, 2].

    Returns:
        (np.array): Normalized vectors of size [N, 2].
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    normVals = norm(vec)
    output = vec / normVals[:, np.newaxis]
    return output
    #################################
    ######### DO NOT MODIFY #########
    #################################


def calculate_uv(p, q, x):
    """Find the u and v coefficients for morphing based on the destination line and destination coordinate.

    This implements Equations 1 and 2 from the Beier-Neely paper.

    This function returns a tuple, which you can expand as follows:

    u, v = calculate_uv(p, q, x)

    Hint #1:
        The functions you implemented above like `norm` and `perp` take in as input a collection
        of vectors, as in size [num_coords, 2] and the like. Often times, you'll find that the arrays 
        you have (like `origin` and `destination`) are size [2]. You can _still_ use these with those
        vectorized functions, by just doing something like `origin[None]` which reshapes the
        array into size [1, 2].

    Args:
        p (np.array): Origin (the P point) of the destination line of shape [2].
        q (np.array): Destination (the Q point) of the destination line of shape [2].
        x (np.array): The destination coords to calculate the uv for (the X point) of shape [2].

    Returns:
        (float, float):
            - The u coefficients for morphing for each coordinate.
            - The v coefficients for morphing for each coordinate.
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################

    if p.shape != (1, 2) and q.shape != (1, 2):
        p, q = p[None], q[None]

    # Calculate U first:
    u_numerator = (x - p) @ (q - p).T

    u_denominator = norm(q - p) ** 2

    u = u_numerator / u_denominator

    # Calculate V:
    v_numerator = (x - p) @ (perp(q - p).T)

    v_denominator = norm(q - p)

    v = v_numerator / v_denominator

    return u, v
    #################################
    ######### DO NOT MODIFY #########
    #################################


def calculate_x_prime(p_prime, q_prime, u, v):
    """Find the source coordinates (X') from the source line (P', Q') and the u, v coefficients.

    This function should implement Equation 3 on page 36 of the Beier-Neely algorithm. 

    Args:
        p_prime (np.array): Origin (the P' point) of the source line of shape [2].
        q_prime (np.array): Destination (the Q' point) of the destination line of shape [2].
        u (float): The u coefficients for morphing.
        v (float): The v coefficients for morphing.

    Returns:
        (np.array): The source coordinates (X') of shape [2]. 
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    if p_prime.shape != (1, 2) and q_prime.shape != (1, 2):
        p_prime, q_prime = p_prime[None], q_prime[None]
    seg1 = p_prime

    seg2 = u @ (q_prime - p_prime)

    seg3 = v @ perp(q_prime - p_prime) / norm(q_prime - p_prime)

    return seg1 + seg2 + seg3
    #################################
    ######### DO NOT MODIFY #########
    #################################


def single_line_pair_algorithm(x, p, q, p_prime, q_prime):
    """Transform the destination coordinates (X) to the source (X') using the single line pair algorithm.

    This should implement the first pseudo-code from the top left of page 37 of the Beier-Neely paper.

    Args:
        x (np.array): The destination coordinates (X) of shape [2].
        p (np.array): Origin (the P point) of the destination line of shape [2].
        q (np.array): Destination (the Q point) of the destination line of shape [2].
        p_prime (np.array): Origin (the P' point) of the source line of shape [2].
        q_prime (np.array): Destination (the Q' point) of the source line of shape [2].
    
    Returns:
        (np.array): The source coordinates (X') of shape [2]
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    u, v = calculate_uv(p=p, q=q, x=x)
    x_prime = calculate_x_prime(p_prime, q_prime, u, v)
    return x_prime
    #################################
    ######### DO NOT MODIFY #########
    #################################


def _min_dist_calculator(x, pi, qi):
    """
    Helper function for multiple_line_pair_algorithm function:
    
    Return the shortest distance from point x to the line connected by p_i q_i
    """
    delta = qi - pi
    dist = x - pi
    oldnew = np.dot(dist, delta)
    lineseg_len = np.dot(delta, delta)
    t = np.clip(oldnew / lineseg_len, 0, 1)
    closestPoint = pi + t * delta
    output = np.linalg.norm(x - closestPoint)
    return output
    

def multiple_line_pair_algorithm(x, ps, qs, ps_prime, qs_prime, param_a, param_b, param_p):
    """Transform the destination coordinates (X) to the source (X') using the multiple line pair algorithm.

    This function should implement the pseudo code on the bottom right of page 37 of the Beier-Neely paper.

    Args: 
        x (np.array): The destination coordinates (X) of shape [2].

        ps (np.array): Origin (the P point) of the destination line of shape [num_lines, 2].

        qs (np.array): Destination (the Q point) of the destination line of shape [num_lines, 2].
        
        ps_prime (np.array): Origin (the P' point) of the source line of shape [num_lines, 2].
        qs_prime (np.array): Destination (the Q' point) of the source line of shape [num_lines, 2].
        
        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.
    
    Returns:
        (np.array): The source coordinates (X') of shape [2]
    """
    ################################
    ####### PUT YOUR CODE HERE #####
    ################################
    DSUM = np.array([0, 0])
    weightSum = 0

    lineNum = ps_prime.shape[0]

    length_info = norm(ps - qs)

    for i in range(lineNum):

        x_prime = single_line_pair_algorithm(x, ps[i], qs[i], ps_prime[i], qs_prime[i])

        di = x_prime - x

        dist = _min_dist_calculator(x, ps[i], qs [i])

        length = length_info[i]

        weight = (length ** param_p / (param_a + dist)) ** param_b

        DSUM = np.add(di[0] * weight, DSUM)

        weightSum += weight

    final_x_prime = x + DSUM / weightSum

    return final_x_prime
    #################################
    ######### DO NOT MODIFY #########
    #################################


def interpolate_at_x(source_image, x, bilinear=False):
    """Interpolates the source_image at some location x.
    
    Args:
        source_image (np.array): The source image of shape [H, W, 4]
        x (np.array): The source coordinates (X) of shape [2] in [-1, 1] coordinates.
        bilinear (bool): If true, will turn on bilinear sampling.
    
    Returns:
        (np.array): The source pixel of shape [4].
    """
    h, w = source_image.shape[:2]
    
    # [0, w] and [0, h] in floats
    pixel_float = img_ops.unnormalize_coordinates(x, h, w)

    if bilinear:

        raw_x, raw_y = pixel_float[0], pixel_float[1]

        # Get the floor / ceil information
        fl_x, ceil_x = int(np.floor(raw_x)), int(np.ceil(raw_x))
        fl_y, ceil_y = int(np.floor(raw_y)), int(np.ceil(raw_y))

        # Define Four Corner Points First for convience of accessing
        Q11, Q21 = (fl_x, fl_y), (ceil_x, fl_y)
        Q12, Q22 = (fl_x, ceil_y), (ceil_x, ceil_y)

        pix11, pix12, pix21, pix22 = np.zeros(4), np.zeros(4), np.zeros(4), np.zeros(4)

        BLACK = np.array([0, 0, 0, 1])

        # Obtain Original Color
        if fl_x >= 0 and fl_y >= 0 and fl_x < w and fl_y < h:
            pix11 = source_image[fl_y, fl_x]
        else:
            pix11 = BLACK

        if ceil_x >= 0 and fl_y >= 0 and ceil_x < w and fl_y < h:
            pix21 = source_image[fl_y, ceil_x]
        else:
            pix21 = BLACK

        if fl_x >= 0 and ceil_y >= 0 and fl_x < w and ceil_y < h:
            pix12 = source_image[ceil_y, fl_x]
        else:
            pix12 = BLACK

        if ceil_x >= 0 and ceil_y >= 0 and ceil_x < w and ceil_y < h:
            pix22 = source_image[ceil_y, ceil_x]
        else:
            pix22 = BLACK
        
        # Interpolate on X-Axis

        if fl_x - ceil_x != 0:  # To make sure no invalid division would happen
            f_x_y1 = (ceil_x - raw_x) / (ceil_x - fl_x) * pix11 + (raw_x - fl_x) / (ceil_x - fl_x) * pix21
            f_x_y2 = (ceil_x - raw_x) / (ceil_x - fl_x) * pix12 + (raw_x - fl_x) / (ceil_x - fl_x) * pix22

        else:
            f_x_y1 = pix11
            f_x_y2 = pix12

        # Interpolate on Y-Axis
        if fl_y - ceil_y != 0:  # To make sure no invalid division would happen
            f_x_y = (ceil_y - raw_y) / (ceil_y - fl_y) * f_x_y1 + (raw_y - fl_y) / (ceil_y - fl_y) * f_x_y2
        else:
            f_x_y = f_x_y1

        return f_x_y


    else:
        # Nearest neighbour interpolation
        # [0, w] and [0, h] in integers
        # We round, because the X.0 boundaries are the pixel centers. We select the nearest pixel centers.
        # When you implement bilinear interpolation, make sure you handle this correctly... samples can on
        # either sides of the pixel center!
        pixel_int = np.round(pixel_float).astype(np.int32)
        c, r = list(pixel_int)
        if c >= 0 and r >= 0 and c < w and r < h:
            return source_image[r, c]
        else:
            return np.zeros([4])



def backward_mapping(source_image, destination_image, source_morph_lines, destination_morph_lines,
                     param_a, param_b, param_p, supersampling, bilinear, vectorize):
    """Perform backward mapping onto the destination image.
    
    Args:
        source_image (np.ndarray): The source image of shape [H, W, 4]
        destination_image (np.ndarray): The destination image of shape [H, W, 4]

        source_morph_lines (np.ndarray): [N, 2, 2] tensor of coordinates for lines. The format is:
                                         [num_lines, (origin, destination), (x, y)]

        destination_morph_lines (np.ndarray): [N, 2, 2] tensor of coordinates for lines. The format is:
                                              [num_lines, (origin, destination), (x, y)]

        param_a (float): The `a` parameter from the Beier-Neely paper controlling morph strength.
        param_b (float): The `b` parameter from the Beier-Neely paper controlling relative
                         line morph strength by distance.
        param_p (float): The `p` parameter from the Beier-Neely paper controlling relative
                         line morph strength by line length.

        supersampling (int): The patch size for supersampling. 

        bilinear (bool): If True, will use bilinear interpolation on the pixels.
        vectorize (bool): If True, will use the vectorized version of the code (optional).

     Returns:
         (np.ndarray): [H, W, 4] image with the source image projected onto the destination image.
    """

    h, w, _ = destination_image.shape
    assert(source_image.shape[0] == h)
    assert(source_image.shape[1] == w)

    # The h, w, 4 buffer to populate and return
    output_buffer = np.zeros_like(destination_image)
    
    # The integer coordinates which you can access via xs_int[r, c]
    xs_int = img_ops.create_coordinates(h, w)

    # The float coordinates [-1, 1] which you can access via xs[r, c]
    # To avoid confusion, you should always denormalize this using img_ops.denormalize_coordinates(xs, h, w)
    # which will bring it back to pixel space, and avoid doing any pixel related operations (like filtering,
    # interpolation, etc) in normalized space. Normalized space however is nice for heavy numerical operations
    # for floating point precision reasons.
    xs = img_ops.normalize_coordinates(xs_int, h, w) 

    # Unpack the line tensors into the start and end points of the line
    ps = destination_morph_lines[:, 0]
    qs = destination_morph_lines[:, 1]
    ps_prime = source_morph_lines[:, 0]
    qs_prime = source_morph_lines[:, 1]

    
    if not vectorize:
        print("Algorithm running without vectorization...")
        print("")

        # calculate the size of a single grid
        # pixel_ri_cj: pixel at row i column j
        pixel_r0_c0 = xs[0, 0]
        pixel_r0_c1 = xs[0, 1]
        pixel_r1_c0 = xs[1, 0]
        len_grid_x = pixel_r0_c1[0] - pixel_r0_c0[0]
        len_grid_y = pixel_r1_c0[1] - pixel_r0_c0[1]

        for r in range(h):
            # tqdm (a progress bar library) doesn't work great with kivy, 
            # so we implment our own progress bar here.
            # you should ignore this code for the most part.
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            percent_done = float(r)/float(h-1)
            print(f"[{'#' * int(percent_done*30)}{'-' * (30-int(percent_done*30))}] {int(100*percent_done)}% done")

            for c in range(w):
                
                x = xs[r, c][None]  # get the pixel coordinate
            
                # Case 1: supersampling -> True
                if supersampling > 1:

                    len_small_grid_x, len_small_grid_y = len_grid_x / supersampling, len_grid_y / supersampling
                    raw_x, raw_y = x[0][0], x[0][1]

                    # Desribe the start / end of the sampled points on X
                    x_starts = raw_x - len_grid_x / 2 + len_small_grid_x / 2
                    
                    # Desribe the start / end of the sampled points on Y
                    y_starts = raw_y - len_grid_y / 2 + len_small_grid_y / 2

                    processedColor = np.zeros(4)

                    patchSize = supersampling

                    for i in range(patchSize):
                        for j in range(patchSize):
                            currX = x_starts + i * len_grid_x
                            currY = y_starts + j * len_grid_y
                            currSample = np.array([currX, currY])

                            if source_morph_lines.shape[0] != 1:
                                new_sample = multiple_line_pair_algorithm(currSample, ps, qs, ps_prime, qs_prime, param_a, param_b, param_p)
                            else:
                                new_sample = single_line_pair_algorithm(currSample, ps, qs, ps_prime, qs_prime)[0]
                                
                            # Perform the interpolation
                            processedColor += interpolate_at_x(source_image, new_sample, bilinear)

                    output_buffer[r, c] = processedColor / (supersampling ** 2)

                # Case 2: supersampling -> False
                else:
                    if source_morph_lines.shape[0] == 1:
                        # Case 2-1: One Morph Line
                        xy = single_line_pair_algorithm(x, ps, qs, ps_prime, qs_prime)[0]
                    else:
                        # Case 2-2: Several Morph Line
                        xy = multiple_line_pair_algorithm(x, ps, qs, ps_prime, qs_prime, param_a, param_b, param_p)[0]

                    new_x, new_y = img_ops.unnormalize_coordinates(xy, h, w)

                    if not bilinear:
                        if int(new_y) < h and int(new_x) < w:
                            output_buffer[r, c] = source_image[int(new_y), int(new_x)]
                        else:
                            output_buffer[r, c] = np.array([0, 255, 0, 1])
                    
                    else:
                        temp = np.array([new_x, new_y])
                        coord = img_ops.normalize_coordinates(temp, h, w)
                        color = interpolate_at_x(source_image, coord, True)
                        output_buffer[r, c] = color

    return output_buffer