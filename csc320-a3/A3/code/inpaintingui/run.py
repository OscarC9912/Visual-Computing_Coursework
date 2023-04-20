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
# DO NOT MODIFY THIS FILE
#


import sys
import argparse


# import the UI-related modules
from inpaintingui import viewer
from inpaintingui.widgets import VisCompApp


def main(argv, prog=''):
    VisCompApp().run()
        

if __name__ == '__main__':
    main(sys.argv[1:], sys.argv[0])
