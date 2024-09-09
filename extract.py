#!/usr/bin/env python3

import argparse
import numpy as np
import mahotas
import sys
import cv2
import os
import h5py

from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from common import *


def SaveH5Dataset(dirPath, filename, data):
    path = os.path.join(dirPath, filename)
    f = h5py.File(path, 'w')
    f.create_dataset('dataset_1', data=np.array(data))
    f.close()
    print("Wrote {}".format(path))


def ParseFeatureFlags(value):
    error = "Provide a comma separated list of features (hara, lbp)"

    try:
        features = 0

        for feature in value.split(','):
            if feature == "hara":
                features |= FEATURE_FLAG_HARALICK
            elif feature == "lbp":
                features |= FEATURE_FLAG_LBP
            else:
                raise argparse.ArgumentTypeError(error)

        return features
    except ValueError:
        raise argparse.ArgumentTypeError(error)


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("-i", "--input", type=str, default="DeepWeed1009", help="Path to the reduced dataset or image")
    prs.add_argument("-o", "--output", type=str, default="features", help="Specify the path for the extracted image's features")
    prs.add_argument("-c", "--class-count", type=int, default=len(LABELS), help="Specify the number of classes for classification")
    prs.add_argument("-f", "--features", type=ParseFeatureFlags, default=FEATURE_FLAG_HARALICK, help="Specify which features extract")
    args = prs.parse_args()

    lbp = None
    if args.features & FEATURE_FLAG_LBP:
        lbp = lbp_extractor()

    if os.path.isfile(args.input):
        # Extract features from a single image

        if not IsJPEG(args.input):
            print("Error: expected JPEG file")
            sys.exit(1)

        features = ExtractFeaturesFromFile(args.input, args.features, lbp)
        np.ndarray.tofile(features.astype("float32"), args.output)
    else:
        # Extract image features from the dataset created by prepare.py

        outputPath = os.path.join(args.input, "output" + str(args.class_count))
        os.makedirs(outputPath, exist_ok=True)

        features = []
        labels = []

        for labelName in GetLabels(args.class_count):
            categoryPath = os.path.join(args.input, labelName)

            print("Extracting {}".format(labelName))
            for filename in os.listdir(categoryPath):
                if not IsJPEG(filename):
                    continue

                filepath = os.path.join(categoryPath, filename)
                image = cv2.imread(filepath)
                features.append(ExtractFeatures(image, flags=args.features, lbp=lbp, reshape=False))

                labels.append(labelName)
        print("Feature extraction completed")

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaledFeatures = scaler.fit_transform(features)
        SaveH5Dataset(outputPath, "data.h5", scaledFeatures)

        le = LabelEncoder()
        encodedLabels = le.fit_transform(labels)
        SaveH5Dataset(outputPath, "labels.h5", encodedLabels)

        # Save which features have been used, so that train.py can detect that it later
        featuresPath = os.path.join(outputPath, "features")
        with open(featuresPath, 'wb') as f:
            f.write(args.features.to_bytes(4, byteorder='little'))
        print("Wrote {}".format(featuresPath))
