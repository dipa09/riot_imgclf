import os
import cv2
import mahotas
import numpy as np
import struct

SIZE = 128 # 64, 224, 256

LABELS = [
    'Chinee Apple',
    'Lantana',
    'Parkinsonia',
    'Parthenium',
    'Prickly Acacia',
    'Rubber Vine',
    'Siam Weed',
    'Snake Weed'
]

IMAGES_PER_LABEL = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016]


def GetLabels(count):
    return LABELS[:count]


def IsJPEG(filename):
    return filename.endswith('.jpg') or filename.endswith('.jpeg')


def InitLBPLib():
    import os
    from ctypes import cdll, c_uint, c_ubyte

    try:
        lib = cdll.LoadLibrary(os.path.abspath("liblbp.so"))
        ND_POINTER_2 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=2, flags="C")
        lib.ExtractLBP.argtypes = [ND_POINTER_2, c_uint, c_uint, ND_POINTER_2]
        lib.ExtractLBP.restype = None
    except OSError as ex:
        print("{}\nWarning: unable to load LBP extraction library, fallback to (slow) python version".format(ex))
        lib = None

    return lib


def EvalPixel(image, center, x, y):
    p = 0
    try:
        if image[y][x] >= center:
            p = 1
    except:
        pass

    return p


def LBP(image, x, y):
    center = image[y][x]
    return (1   * EvalPixel(image, center, x-1, y-1) + # top left
            2   * EvalPixel(image, center, x,   y-1) + # top
            4   * EvalPixel(image, center, x+1, y-1) + # top right
            8   * EvalPixel(image, center, x+1, y)   + # right
            16  * EvalPixel(image, center, x+1, y+1) + # bottom right
            32  * EvalPixel(image, center, x,   y+1) + # bottom
            64  * EvalPixel(image, center, x-1, y+1) + # bottom left
            128 * EvalPixel(image, center, x-1, y))    # left


def ExtractLBP_slow(grayImage, width, height):
    out = np.zeros((height, width), np.uint8)
    for y in range(0, height):
        for x in range(0, width):
            out[y][x] = LBP(grayImage, x, y)

    lbp = out.mean(axis=0)
    return lbp


def ExtractLBP(lib, grayImage, width, height):
    data = np.ascontiguousarray(grayImage, np.uint8)
    out = np.zeros((width, height), np.uint8)
    lib.ExtractLBP(data, width, height, out)
    return out.mean(axis=0)


class lbp_extractor:
    lib = None

    def __init__(self):
        self.lib = InitLBPLib()

    def Extract(self, grayImage, width, height):
        if self.lib:
            return ExtractLBP(self.lib, grayImage, width, height)
        return ExtractLBP_slow(grayImage, width, height)


FEATURE_FLAG_HARALICK = 0x1
FEATURE_FLAG_LBP      = 0x2
def ExtractFeatures(image, flags, lbp, reshape=True):
    image = cv2.resize(image, (SIZE, SIZE))
    hsvImage = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Extract color histogram
    bins = 8
    hist = cv2.calcHist([hsvImage], [0, 1, 2], None, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist = hist.flatten()

    features = [hist]

    # Extract haralick texture
    hara = None
    if flags & FEATURE_FLAG_HARALICK:
        hara = mahotas.features.haralick(grayImage).mean(axis=0)
        features.append(hara)

    # Extract local binary patterns
    lbpData = None
    if flags & FEATURE_FLAG_LBP:
        lbpData = lbp.Extract(grayImage, SIZE, SIZE)
        features.append(lbpData)

    # Extract Hu Moments
    hum = cv2.HuMoments(cv2.moments(grayImage)).flatten()
    features.append(hum)

    features = np.hstack(features)

    if reshape:
        features = features.reshape(1, -1)
    return features


def ExtractFeaturesFromFile(path, flags, lbp):
    image = cv2.imread(path)
    return ExtractFeatures(image, flags, lbp)


def ReadFeatureFlags(datasetPath):
    path = os.path.join(datasetPath, "features")

    features = 0
    with open(path, 'rb') as f:
        features = struct.unpack('i', f.read(4))[0]
    if features == 0:
        raise ValueError('no feature flags')
    return features
