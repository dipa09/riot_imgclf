#!/usr/bin/env python3

#
# Test C bindings for LBP texture extraction.
#

import argparse
import numpy as np
import os
import sys
import time

from common import *


def PrintElapsedTime(mesg, elapsed):
    suffix = "s"
    if elapsed < 0.001:
        elapsed *= 1000
        suffix = "ms"

    print("{}{}{}".format(mesg, elapsed, suffix))


def OpenGrayImage(path):
    image = cv2.imread(path)
    image = cv2.resize(image, (SIZE, SIZE))
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("-i", "--input", type=str, required=True, help="Image/directory path")
    prs.add_argument("-l", "--limit", type=int, help="Limit the number of images to test")
    args = prs.parse_args()

    lib = InitLBPLib()
    if lib is not None:
        if os.path.isfile(args.input):
            image = OpenGrayImage(args.input)

            start = time.time()
            A = ExtractLBP_slow(image, SIZE, SIZE)
            end = time.time()
            PrintElapsedTime("Python LBP extraction: ", end - start)

            start = time.time()
            B = ExtractLBP(lib, image, SIZE, SIZE)
            end = time.time()
            PrintElapsedTime("C LBP extraction:      ", end - start)

            # print(A)
            # print('\n');
            # print(B)
        else:
            elapsed = [0, 0]

            filenames = os.listdir(args.input)
            maxImages = args.limit
            if maxImages is None or maxImages > len(filenames):
                maxImages = len(filenames)

            for i in range(0, maxImages):
                imagePath = os.path.join(args.input, filenames[i])
                image = OpenGrayImage(imagePath)

                start = time.time()
                A = ExtractLBP_slow(image, SIZE, SIZE)
                elapsed[0] += time.time() - start

                start = time.time()
                B = ExtractLBP(lib, image, SIZE, SIZE)
                elapsed[1] += time.time() - start


            PrintElapsedTime("Python LBP extration (total): ", elapsed[0])
            PrintElapsedTime("C LBP extration (total):      ", elapsed[1])
