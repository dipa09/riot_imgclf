#!/usr/bin/env python3

import argparse
import cv2
import h5py
import joblib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys

from random import randrange
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm, metrics, preprocessing

from common import *


#
# RF Tuning (optional)
#
class random_forest_params:
    treeCount = 0
    maxDepth = None
    seed = 42
    accuracy = 0
    modelSize = None

    def __init__(self, treeCount, maxDepth, seed, accuracy, modelSize):
        self.treeCount = treeCount
        self.maxDepth = maxDepth
        self.seed = seed
        self.accuracy = accuracy
        self.modelSize = modelSize


class random_forest_params_iterator:
    paramsTable = []
    current = 0

    def __init__(self, paramsTable):
        self.paramsTable = paramsTable

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= len(self.paramsTable):
            raise StopIteration
        params = self.paramsTable[self.current]
        self.current += 1

        result = []
        result.append(params.treeCount)
        result.append(params.maxDepth)
        result.append(params.accuracy)
        if params.modelSize:
            result.append(params.modelSize)
        result.append(params.seed)
        return result


class tune_range:
    def __init__(self, Min, Max):
        self.Min = Min
        self.Max = Max


def TuneRangeException():
    return argparse.ArgumentTypeError("Provide an integer range like so '1,3'")


def GetTuneRange(value):
    try:
        values = value.split(',')
        if len(values) == 2:
            Min = int(values[0])
            Max = int(values[1])
            if Min < Max:
                return tune_range(Min, Max)
        raise TuneRangeException()
    except ValueError:
        raise TuneRangeException()


def TuneRandomForest(args, trainX, testX, trainY, testY):
    if args.show_model_size:
        import emlearn
        from emlearn.evaluate import trees

    minTrees = 5
    maxTrees = 15
    if args.tune_trees is not None:
        minTrees = args.tune_trees.Min
        maxTrees = args.tune_trees.Max

    minDepth = 6
    maxDepth = 10
    if args.tune_depth is not None:
        minDepth = args.tune_depth.Min
        maxDepth = args.tune_depth.Max

    iterations = 0
    totalIterations = float((maxTrees - minTrees + 1)*(maxDepth - minDepth + 1))

    paramsTable = []
    for treeCount in range(minTrees, maxTrees+1):
        for maxDepth in range(minDepth, maxDepth+1):
            clf = RandomForestClassifier(n_estimators=treeCount,
                                         max_depth = maxDepth,
                                         random_state=args.seed)

            clf.fit(trainX, trainY)
            predY = clf.predict(testX)
            accuracy = metrics.accuracy_score(predY, testY)

            modelSize = None
            if args.show_model_size:
                SaveModel(clf, ".tuning_model")
                model = joblib.load(".tuning_model")
                modelSize = emlearn.evaluate.trees.model_size_bytes(model)

            params = random_forest_params(treeCount, maxDepth, args.seed, accuracy, modelSize)
            paramsTable.append(params)

            iterations += 1
            print("Tuning RF {:.2f}%".format((float(iterations) / totalIterations)*100), end='\r')

    # Remove tuning_model
    if args.show_model_size:
        os.remove(".tuning_model")

    paramsTable.sort(key = lambda x: x.accuracy, reverse = True)

    try:
        from tabulate import tabulate
        headers = ["Trees", "Max depth", "Accuracy"]
        if args.show_model_size:
            headers.append("Size")
        headers.append("Seed")

        print(tabulate(random_forest_params_iterator(paramsTable), headers))

        if args.output is not None:
            with open(args.output, 'w') as f:
                f.write(tabulate(random_forest_params_iterator(paramsTable), headers))
                f.write("\n")
            print("Wrote tuning report to {}".format(args.output))

    except ImportError:
        for params in paramsTable:
            print("trees: {}, max-depth: {}, accuracy: {:.4f}, seed: {}"
                  .format(params.treeCount, params.maxDepth, params.accuracy, params.seed))

#
# Random Image Testing (optional)
#
def TestRandomImages(args, datasetPath):
    labels = GetLabels(args.class_count)

    flags = ReadFeatureFlags(datasetPath)
    lbp = None
    if flags & FEATURE_FLAG_LBP:
        lbp = lbp_extractor()

    randomCount = args.test_random
    errorCount = 0

    filenamesMap = {}
    while randomCount > 0:
        randomLabel = labels[randrange(len(labels))]
        categoryPath = os.path.join(args.input, randomLabel)

        if randomLabel in filenamesMap:
            filenames = filenamesMap[randomLabel]
        else:
            filenames = os.listdir(categoryPath)
            filenamesMap[randomLabel] = filenames

        randomFilename = None
        while True:
            randomFilename = filenames[randrange(len(filenames))]
            if IsJPEG(randomFilename):
                break
        assert randomFilename is not None

        randomImagePath = os.path.join(categoryPath, randomFilename)
        image = cv2.imread(randomImagePath)
        features = ExtractFeatures(image, flags, lbp)
        prediction = clf.predict(features)[0]

        color = (0, 255, 0) # Green
        if labels[prediction] != randomLabel:
            errorCount += 1
            color = (0, 0, 255) # Red
            print("Incorrect prediction for {}: {} instead of {}".format(randomImagePath, labels[prediction], randomLabel))
        elif args.verbose:
            print("Correct prediction for {}".format(randomImagePath))

        if args.show_random_images:
            cv2.putText(image, labels[prediction], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 3)
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()

        randomCount -= 1
        # np.ndarray.tofile(features.astype("float32"), "features")
        # print("Wrote features of '{}' to features".format(testImagePath))
    successCount = args.test_random - errorCount
    print("Outcome {}/{}  {}%"
          .format(successCount, args.test_random, (float(successCount)/float(args.test_random))*100.0))

#
#
#

def LoadDataset(path, testSize=0.15, seed=42):
    dataFile  = h5py.File(os.path.join(path, "data.h5"), 'r')
    labelsFile = h5py.File(os.path.join(path, "labels.h5"), 'r')

    features = np.array(dataFile['dataset_1'])
    labels = np.array(labelsFile['dataset_1'])

    dataFile.close()
    labelsFile.close()

    return train_test_split(np.array(features), np.array(labels), test_size=testSize, random_state=seed)


def SaveModel(clf, path):
    # Using 'compress=3' may reduce the model size up to 5 times
    joblib.dump(clf, path, compress=3)


if __name__ == '__main__':
    prs = argparse.ArgumentParser()
    # Global options
    prs.add_argument("-i", "--input", type=str, default="DeepWeed1009", help="Path to the reduced dataset")
    prs.add_argument("-o", "--output", type=str, default="model", help="Specify model output path")
    prs.add_argument("-s", "--seed", type=int, default=9, help="Specify the random seed")
    prs.add_argument("-c", "--class-count", type=int, default=len(LABELS), help="Specify the number of classes for classification")
    prs.add_argument("-v", "--verbose", action="store_true", help="Print more info")
    # Training options
    prs.add_argument("-m", "--model-kind", type=str, choices=["rf", "svm"], default="rf", help="Specify the classifier to use")
    # NOTE: action='store_false' doesn't work
    prs.add_argument("--show-confusion-matrix", action="store_true", help="Show the confusion matrix")
    prs.add_argument("--show-training-accuracy", action="store_true", help="Show training accuracy")
    prs.add_argument("--show-balanced-accuracy", action="store_true", help="Show balanced accuracy")
    #
    prs.add_argument("--test-random", type=int, default=1, help="Test random images")
    prs.add_argument("--show-random-images", action="store_true", help="Show tested images")
    ## RF options
    prs.add_argument("--trees", type=int, default=15, help="Specify the number of trees for RF")
    prs.add_argument("--max-depth", type=int, default=10, help="Specify trees' max depth for RF")
    # Tuning options
    prs.add_argument("-t", "--tune", action="store_true", help ="Tune RF")
    prs.add_argument("--show-model-size", action="store_true",help="Show estimated model size during RF tuning")
    prs.add_argument("--tune-trees", type=GetTuneRange, help="Specify the number of trees as a range for RF tuning")
    prs.add_argument("--tune-depth", type=GetTuneRange, help="Specify the range of max depths for RF tuning")
    ##
    args = prs.parse_args()

    featuresDirPath = os.path.join(args.input, "output" + str(args.class_count))
    trainX, testX, trainY, testY = LoadDataset(featuresDirPath, seed=args.seed)

    if args.tune:
        TuneRandomForest(args, trainX, testX, trainY, testY)
    else:
        clfKind = args.model_kind.lower()
        if clfKind == 'rf':
            numTrees = args.trees
            maxDepth = args.max_depth

            clf = RandomForestClassifier(n_estimators=numTrees, max_depth=maxDepth, random_state=args.seed)
            print("RF trees: {}, max-depth: {}, seed: {}".format(numTrees, maxDepth, args.seed))
        elif clfKind == 'svm':
            # decisionFun = "ovo"
            # clf = svm.SVC(decision_function_shape=decisionFun, gamma=0.1)
            #print("SVM {}".format(decisionFun))
            #clf = svm.LinearSVC(max_iter=10000)
            clf = None
        else:
            assert False

        clf.fit(trainX, trainY)

        predY = clf.predict(testX)

        accuracy = metrics.accuracy_score(predY, testY)
        print("Accuracy: {}".format(accuracy))
        if args.show_balanced_accuracy:
            balancedAcc = metrics.balanced_accuracy_score(predY, testY)
            print("Accuracy (balanced): {}".format(balancedAcc))

        if args.show_training_accuracy:
            trainPredY = clf.predict(trainX)
            trainAcc = metrics.accuracy_score(trainPredY, trainY)
            print("Training accuracy: {}".format(trainAcc))
            balancedTrainAcc = metrics.balanced_accuracy_score(trainPredY, trainY)
            print("Training accuracy (balanced): {}".format(balacnedTrainAcc))

        if args.show_confusion_matrix:
            from sklearn.metrics import ConfusionMatrixDisplay
            disp = ConfusionMatrixDisplay.from_estimator(
                clf,
                testX,
                testY,
                display_labels=GetLabels(args.class_count),
                cmap=plt.cm.Blues,
                normalize=None,
            )
            disp.ax_.set_title("Confusion matrix, without normalization")
            #print(disp.confusion_matrix)
            plt.show()

        # Save the model
        SaveModel(clf, args.output)
        print("Wrote model to {}".format(args.output))

        if args.test_random > 0:
            TestRandomImages(args, featuresDirPath)
