#!/usr/bin/env python3

import argparse
import sys
import os
from shutil import copyfile, rmtree
from common import LABELS, IMAGES_PER_LABEL


def DumpDeepWeedTable(categories):
    for i in range(len(categories)):
        label = LABELS[i]
        category = categories[i]
        print("{} [{}]:".format(label, len(category)))
        for filename in category:
              print("\t{}".format(filename))


def CheckDeepWeedTable(categories):
    for catIdx, category in enumerate(categories):
        expectedSize = IMAGES_PER_LABEL[catIdx]
        assert len(category) == expectedSize


def CopyImages(destPath, imagesPath, categories, countPerCategory, dryRun=False):
    totalImages = 0
    for catIdx, category in enumerate(categories):
        # Create sub-directory for each catergory
        label = LABELS[catIdx]
        categoryPath = os.path.join(destPath, label)
        if not dryRun:
            os.makedirs(categoryPath, exist_ok = True)

        # Copy the first 'copyImageCount' images
        copyImageCount = min(countPerCategory, len(category))
        if not dryRun:
            print("Copying {} images to {} ...".format(copyImageCount, categoryPath), end='')
        for filenameIdx in range(copyImageCount):
            filename = category[filenameIdx]
            sourceImagePath = os.path.join(imagesPath, filename)
            destImagePath = os.path.join(categoryPath, filename)
            if dryRun:
                print("Copy {} -> {}".format(sourceImagePath, destImagePath))
            else:
                copyfile(sourceImagePath, destImagePath)
        if not dryRun:
            print(" Done")
        totalImages += copyImageCount
    if dryRun:
        print("Copied {} images to {}".format(totalImages, destPath))


def PrepareDataset(destPath, imagesPath, labelsCSVPath, imageCountPerCategory, dryRun=False):
    # Load the CSV file with the labels into 'categories' which is an array of
    # an array of string holding the images' filenames for each plant category.
    categories = []
    for _ in range(len(LABELS)):
        categories.append([])

    with open(labelsCSVPath, 'r') as f:
        f.readline()            # Skip the first line
        for line in f.readlines():
            #print(line)
            tokens = line.split(',')
            #print(tokens)
            filename = tokens[0]
            labelIdx = int(tokens[1])
            if labelIdx < len(categories):
                categories[labelIdx].append(filename)

    CheckDeepWeedTable(categories)

    # Create the new dataset
    if not dryRun:
        os.makedirs(destPath, exist_ok=True)

    CopyImages(destPath, imagesPath, categories, imageCountPerCategory, dryRun)


if __name__ == "__main__":
    prs = argparse.ArgumentParser()
    prs.add_argument("-f", "--force", action="store_true", help="Overwrite existing files if the destination directory already exists")
    prs.add_argument("-n", "--dry-run", action="store_true", help="Do not perform any actual IO, just log the operations")
    prs.add_argument("-o", "--output", type=str, help="Specify where to store the new dataset")
    prs.add_argument("-i", "--images", type=int, default=min(IMAGES_PER_LABEL), help="Specify the number of images for each category")
    prs.add_argument("-s", "--source", type=str, default="DeepWeed", help="Path to the full dataset")
    args = prs.parse_args()

    imagesPath = os.path.join(args.source, "images")
    labelsCSVPath = os.path.join(args.source, "labels.csv")

    destPath = args.output
    if destPath is None:
        destPath = "DeepWeed" + str(args.images)

    if os.path.isdir(destPath):
        if args.force:
            # TODO: Rename to temp directory and remove it at the end to not lose the previous
            # dataset if we got errors or the user interrupts the copy
            rmtree(destPath)
        else:
            print("Output directory '{}' already exists. Pass --force to overwrite it".format(destPath))
            sys.exit(1)

    PrepareDataset(destPath, imagesPath, labelsCSVPath, args.images, args.dry_run)
