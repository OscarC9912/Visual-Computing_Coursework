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


# import packages
from typing import List, Tuple, Union
from nnf import *
from algorithm import *
import cv2 as cv
import numpy as np
import time


def profile(fn):
    """
    A decorator function for elapsed-time profiling.
    """
    def with_profiling(*args, **kwargs):
        start_time = time.time()

        ret = fn(*args, **kwargs)
        elapsed_time = time.time() - start_time

        print('\tFunction {} ran {:.4f}'.format(fn.__name__, elapsed_time))

        return ret

    return with_profiling


#####################################################################
#
# The PatchMatch Class
#
#####################################################################


class PatchMatch:
    """
    The PatchMatch Class.

    This class contains the basic methods required for implementing
    the PatchMatch algorithm. See the docstrings of the individual methods.

    To run PatchMatch, one must create an instance of this class. See
    function run() in file run.py for an example of how it is called.
    """
    _images: Dict[str, Union[None, np.ndarray]]

    def __init__(self):
        """
        The class constructor.

        When called, it creates a private dictionary object that acts
        as a container for all input and all output images of
        the inpainting algorithm. These images are initialized to None
        and populated/accessed by calling the readImage(), writeImage(),
        and run_iterations() methods.
        """
        self._images = {
            'source': None,
            'target': None,
            'NNF-image': None,
            'NNF-vectors': None,
            'rec-source': None,
        }
        # set default parameters
        self._iters = None
        self._patch_size = None
        self._alpha = None
        self._w = None
        self._im_shape = None
        self._f = None
        self._best_D = None
        self._disable_random = None
        self._disable_propagation = None
        self._output = None
        self._partial_results = None
        self._NNF_vectors = None
        self._NNF_image = None
        self._rec_source = None
        self._server = None
        self._NNF_subsampling = None
        self._NNF_line_width = None
        self._NNF_line_color = None
        self._tmpdir = None
        # internal algorithm variables
        self._need_init = True
        self._source_patches = None
        self._target_patches = None
        self._current_iteration = None
        self._init_NNF_filename = None
        self._global_vars = None
        self._random_enabled = None
        self._propagation_enabled = None

    def read_image(self, filename: str, key: str) \
            -> Tuple[bool, Union[str, None]]:
        """
        Read image from a file and copy its contents to the PatchMatch instance.

        The method should use OpenCV to read an image from a file and copy
        its contents to the PatchMatch instance's private dictionary object.
        The key specifies the image variable and should be one of the
        strings in lines 82-86.

        If the image was not read successfully, the corresponding dictionary
        entry in the PatchMatch instance should remain unaffected.

        Args:
            filename:
                Path to the image.
            key:
                A string specifying the type of image to be read.

        Returns:
            A tuple containing (1) a boolean variable that is True if the
              image-reading succeeded and (2) an error message if reading was
              unsuccessful.
        """
        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        # readFormat = {'source': cv.IMREAD_UNCHANGED, 'alpha': cv.IMREAD_GRAYSCALE}  
        # Define the mode for image reading
        image = cv.imread(filename, cv.IMREAD_UNCHANGED)
        if image is not None:

            # make sure all images are in RGB
            if image.shape[2] == 4:
                image = cv.cvtColor(image, cv.COLOR_RGBA2RGB)

            if key in self._images:
                self._images[key] = image
                success, msg = True, 'reading is successful'
            else:
                success, msg = False, 'Key Error while opening the image'
        else:
            success, msg = False, 'image is None'

        #########################################

        return success, msg

    def write_image(self, filename: str, key: str) \
            -> Tuple[bool, Union[str, None]]:
        """
        Write image to a file from the PatchMatch instance's private dictionary.

        Args:
            filename:
                Path to the image.
            key:
                A string that specifies which image should be written from
                  the instance's private dictionary.
        Returns:
            A tuple containing (1) a boolean variable that is True if and only
              if writing succeeded and (2) an error message if it was
              unsuccessful.
        """

        success = False
        msg = 'No Image Available'

        #########################################
        ## PLACE YOUR CODE BETWEEN THESE LINES ##
        #########################################
        imageExist = (self._images[key] is not None)

        if imageExist:
            try:
                if key in self._images:
                    cv.imwrite(filename, self._images[key])
                    success, msg = True, 'Image Written Success !!'
                else:
                    success, msg = False, 'Key Error while opening the image'
            except:
                Exception('Writing Image Error')
        else:
            success, msg = False, 'No Image Data Available'
        #########################################
        return success, msg

    @profile
    def _reconstruct_source(self):
        """
        Reconstruct the source image using pixels from the target.

        See algorithm.py for details. You will need to complete the function
        in that file.
        """
        self._images['rec-source'] = \
            reconstruct_source_from_target(self._images['target'], self._f)

    @profile
    def _propagation_and_random_search(self):
        """
        Run one iteration of the PatchMatch algorithm's main loop.

        See algorithm.py for details. You will need to complete the function
        in that file.
        """
        odd_iter = self._current_iteration % 2 != 0
        self._f, self._best_D, self._global_vars = \
            propagation_and_random_search(
                self._source_patches,
                self._target_patches,
                self._f,
                self._alpha, self._w,
                self._propagation_enabled,
                self._random_enabled,
                odd_iter,
                self._best_D,
                self._global_vars
            )

    def initialize_algorithm(self):
        """
        Initialize the variables required for PatchMatch.
        """
        if self._images['source'] is not None:
            self.set_im_shape()
            self._source_patches = make_patch_matrix(self._images['source'],
                                                     self._patch_size)
        else:
            self._source_patches = None
        if self._images['target'] is not None:
            self._target_patches = make_patch_matrix(self._images['target'],
                                                     self._patch_size)
        else:
            self._target_patches = None
        if self._w == 0:
            # if the maximum search radius was not specified, we use the
            # maximum image dimension of the source image
            self._w = np.max(self._images['source'].shape[0:2])
        self._current_iteration = 1
        self._best_D = None
        self._need_init = False

    #
    # Helper functions for implementing one iteration of the
    # PatchMatch algorithm.
    #

    def _validate(self) -> bool:
        """
        Parameter validation method.

        Return True if and only if all the algorithm's required parameters
        and images are available and valid.
        """
        return ((self._images['source'] is not None)
                and (self._images['target'] is not None)
                and (self._source_patches is not None)
                and (self._target_patches is not None)
                and (self._f is not None)
                and (self._patch_size > 0)
                and (self._images['source'].shape[0] ==
                     self._images['target'].shape[0])
                and (self._images['source'].shape[1] ==
                     self._images['target'].shape[1])
                and (self._f.shape[0] == self._images['source'].shape[0])
                and (self._f.shape[1] == self._images['source'].shape[1]))

    def step_algorithm(self):
        """
        Run one PatchMatch iteration and output its partial results.
        """
        # initialize the algorithm data structures if this is the first run
        if self._need_init:
            self.initialize_algorithm()
        success = False
        # make sure all the data we need are available
        if self._validate():
            if self._current_iteration <= self._iters:
                print('Running iteration {}...'.format(self._current_iteration))
                self._propagation_and_random_search()
                self._current_iteration += 1
                success = True
        else:
            return success
        if (self._current_iteration > self._iters) or self.partial_results():
            # write the output files
            if self.NNF_image():
                self._images['NNF-image'] = create_NNF_image(self._f)
                ok, msg = self.write_image(self.make_filename('nnf-col',
                                                              'png',
                                                              not success),
                                           'NNF-image')
                if not ok:
                    print('Error: write_image: ', msg)

            if self.NNF_vectors():
                # this is a kludge: the need to use matplotlib to write the
                # image to a file, then we re-read it into an openCV image,
                # then finally write that openCV image into the desired file
                ok, msg = self.read_image(
                    create_NNF_vectors_image(self._images['source'],
                                             self._images['target'],
                                             self._f,
                                             self._patch_size,
                                             subsampling=self._NNF_subsampling,
                                             line_width=self._NNF_line_width,
                                             line_color=self._NNF_line_color,
                                             tmpdir=self._tmpdir),
                    'NNF-vectors')
                ok, msg = self.write_image(self.make_filename('nnf-vec',
                                                              'png',
                                                              not success),
                                           'NNF-vectors')
                if not ok:
                    print('Error: write_image: ', msg)
            if self.rec_source():
                self._reconstruct_source()
                ok, msg = self.write_image(self.make_filename('rec-src',
                                                              'png',
                                                              not success),
                                           'rec-source')
                if not ok:
                    print('Error: write_image: ', msg)
            ok, msg = save_NNF(self.NNF(), self.make_filename('nnf',
                                                              'npy',
                                                              not success),)
            if not ok:
                print('Error: save_NNF: ', msg)

        return success

    def print_parameters(self):
        """
        Display the algorithm's parameters in a terminal window.
        """
        print('---------------------------------------------------------------')
        print('PatchMatch parameters:')
        if self._init_NNF_filename is not None:
            nnf_str = self._init_NNF_filename
        else:
            nnf_str = 'Generated internally'
        print('\tInitial NNF: \t\t', nnf_str)
        print('\tIterations: \t\t', self.iterations())
        print('\tPatch size: \t\t', self.patch_size())
        print('\tAlpha: \t\t\t', self.alpha())
        print('\tW: \t\t\t', self.w())
        print('\tPropagation enabled: \t', not self.propagation_enabled())
        print('\tRandom search enabled: \t', not self.random_enabled())
        print('Output path and base filename: \t', self.output())
        output_str = ''
        if self.NNF_vectors():
            output_str += "correspondences, "
        if self.NNF_image():
            output_str += "color nnf, "
        if self.rec_source():
            output_str += "rec'd source "
        print('Visualization parameters:')
        if len(output_str) > 0:
            print('\tOutput files: \t\t', output_str)
        print('\tNNF subsampling: \t', self.NNF_subsampling())
        print('\tNNF line width: \t', self.NNF_line_width())
        print('\tNNF line color: \t', self.NNF_line_color())
        print('\tMatplotlib server mode:', self.server())
        print('\tTmp directory: \t\t', self.tmpdir())
        print('---------------------------------------------------------------')

    def make_filename(self, label: str, suffix: str, lastIter: bool = False):
        """
        Helper function for creating filenames of the PatchMatch output.
        """
        if not lastIter:
            iter_str = 'iter%s' % (self._current_iteration-1)
        else:
            iter_str = 'last'
        return self.output() + '.%s.p%s.a%s.w%s.prop%s.rand%s.%s.%s' \
                                % (label,
                                   self.patch_size(),
                                   self.alpha(),
                                   self.w(),
                                   not self.propagation_enabled(),
                                   not self.random_enabled(),
                                   iter_str,
                                   suffix)

    def run_iterations(self):
        """
        Execute k iterations of the PatchMatch algorithm and save its results.
        """
        # initialize the algorithm data structures if this is the first run
        if self._need_init:
            self.initialize_algorithm()
        self.print_parameters()

        ok = True
        while ok:
            ok = self.step_algorithm()

        return

    #
    # Helper methods for setting the algorithm's input, output and
    # control parameters
    #

    # accessor methods for private variables
    def set_iterations(self, i):
        if i >= 0:
            self._iters = i

    def iterations(self):
        return self._iters

    def set_patch_size(self, s):
        if s % 2 == 1:
            # patch sizes must be odd
            self._patch_size = s
            self._need_init = True
        else:
            print('Warning: Patch size must be odd, '
                  'reset to %d' % self._patch_size)

    def patch_size(self):
        return self._patch_size

    def set_alpha(self, a):
        self._alpha = a

    def alpha(self):
        return self._alpha

    def set_w(self, r):
        if r >= 0:
            self._w = r

    def w(self):
        return self._w

    def set_random(self, val):
        self._random_enabled = val

    def random_enabled(self):
        return self._random_enabled

    def set_propagation(self, val):
        self._propagation_enabled = val

    def propagation_enabled(self):
        return self._propagation_enabled

    def set_init_NNF(self, nnf_file: str = None):
        if self._images['source'] is None:
            print('Warning: NNF cannot be loaded before loading a source image')
            return
        if nnf_file is not None:
            ok, f, msg = load_NNF(nnf_file,
                                  shape=self._images['source'].shape[0:2])
            if not ok:
                print('Warning: load_NNF: ', msg)
                print('Generating NNF internally instead')
                self._f = init_NNF(self._images['source'])
                self._init_NNF_filename = None
            else:
                self._init_NNF_filename = nnf_file
                self._f = f
        else:
            self._init_NNF_filename = None
            self._f = init_NNF(self._images['source'])

    def set_output(self, filename):
        self._output = filename

    def output(self):
        return self._output

    def NNF(self):
        return self._f

    def set_im_shape(self):
        if self._images['source'] is not None:
            self._im_shape = self._images['source'].shape
            self._need_init = True

    # variables controlling the image display of NNFs

    def set_server(self, val):
        self._server = val

    def server(self):
        return self._server

    def set_partial_results(self, val):
        self._partial_results = val

    def partial_results(self):
        return self._partial_results

    def set_NNF_subsampling(self, val):
        self._NNF_subsampling = val

    def NNF_subsampling(self):
        return self._NNF_subsampling

    def set_NNF_line_width(self, val):
        self._NNF_line_width = val

    def NNF_line_width(self):
        return self._NNF_line_width

    def set_NNF_line_color(self, val):
        self._NNF_line_color = val

    def NNF_line_color(self):
        return self._NNF_line_color

    def set_NNF_image(self, val):
        self._NNF_image = val

    def NNF_image(self):
        return self._NNF_image

    def set_NNF_vectors(self, val):
        self._NNF_vectors = val

    def NNF_vectors(self):
        return self._NNF_vectors

    def set_rec_source(self, val):
        self._rec_source = val

    def rec_source(self):
        return self._rec_source

    def set_tmpdir(self, tmpdir):
        self._tmpdir = tmpdir

    def tmpdir(self):
        return self._tmpdir
