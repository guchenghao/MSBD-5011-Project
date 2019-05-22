_Code#!/usr/bin/env python3
# -*- coding:utf-8 -*-
###
# File: /Users/guchenghao/CodeWarehouse/MSBD5011_Project/method1/auxiliary.py
# Project: /Users/guchenghao/CodeWarehouse/MSBD5011_Project/method1
# Created Date: Wednesday, May 1st 2019, 6:57:54 pm
# Author: Harold Gu
# -----
# Last Modified: Wednesday, 1st May 2019 6:57:54 pm
# Modified By: Harold Gu
# -----
# Copyright (c) 2019 HKUST
# #
# All shall be well and all shall be well and all manner of things shall be well.
# We're doomed!
###

import cv2 as cv
import time
from PIL import Image
import numpy as np


class Mouth_Decector(object):

    def __init__(self):
        # ! use opencv2s to load classifiers
        self.harr_face = cv.CascadeClassifier(
            '/Users/guchenghao/CodeWarehouse/MSBD5011_Project_Code/model/haarcascade_frontalface_default.xml')
        self.harr_mouth = cv.CascadeClassifier(
            '/Users/guchenghao/CodeWarehouse/MSBD5011_Project_Code/model/haarcascade_mouth.xml')

    # ! FA detected face as detected face

    def find_max_square(self, squares, tag='square', verbose=0):
        max_square_size = 0
        max_square = []
        if len(squares) > 0:
            for (x, y, w, h) in squares:
                if w * h > max_square_size:
                    max_square_size = w * h
                    max_square = [x, y, w, h]

            if verbose > 0:
                if len(max_square) == 0:  # ! if did not detect the face in the window
                    print('no %s found')
                else:
                    print('find %s @ %s' % (tag, str(max_square)))
        return max_square

    def find_lowest_square(self, squares, tag='square', verbose=0):
        max_y = 0
        lowest_square = []
        if len(squares) > 0:
            # ! squares: [0]: x; [1]: y; [2]: width; [3]: height
            for (x, y, w, h) in squares:
                if y > max_y:
                    max_y = y
                    lowest_square = [x, y, w, h]

            if verbose > 0:
                if len(lowest_square) == 0:  # ! if did not detect face
                    print('no %s found in')
                else:
                    print('find %s @ %s' % (tag, str(lowest_square)))

        return lowest_square

    def find_mouth(self, img, is_square_face=False, is_square_mouth=False):

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        faces = self.harr_face.detectMultiScale(gray, 1.3, 5)
        max_face = self.find_max_square(faces, tag='face')

        if len(max_face) > 0:
            x, y, w, h = max_face
            face_gray = gray[y:y+h, x:x+w]
            face_color = img[y:y+h, x:x+w]
            if is_square_face:
                cv.rectangle(img, (x, y), (x+w, y+h), (255, 255, 255), 2)

            mouths = self.harr_mouth.detectMultiScale(face_gray)
            mouth = self.find_lowest_square(mouths, tag='mouth')

            if len(mouth) > 0:
                mx, my, mw, mh = mouth
                if is_square_mouth:
                    cv.rectangle(face_color, (mx-1, my-1),
                                 (mx+mw+1, my+mh+1), (100, 230, 255), 2)

                return x+mx, y+my, mw, mh
        return []

    def get_partial(self, img, x, y, w, h):
        return img[y:y+h, x:x+w]

    def normalize_img(self, img, is_grey=True, is_vectorize=False, width=28, height=10):
        input_shape = width, height  # (width, height)
        im = Image.fromarray(img)  # Image.open(filename)
        resized_im = im.resize(input_shape, Image.ANTIALIAS)  # resize image
        output_res = np.array(resized_im)
        if is_grey:
            # convert the image to *greyscale*
            im_grey = resized_im.convert('L')
            im_array = np.array(im_grey)  # convert to np array
            output_res = im_array
        if is_vectorize:
            oned_array = output_res.reshape(input_shape[0] * input_shape[1])
            output_res = oned_array
        return output_res
