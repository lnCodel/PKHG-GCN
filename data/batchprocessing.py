#!/usr/bin/env python

from __future__ import print_function

import collections
import csv
import logging
import os
import pandas as pd
import SimpleITK as sitk
import numpy as np
import radiomics
from radiomics import featureextractor
from functools import partial
from multiprocessing import Pool
from scipy.io import savemat
import scipy.sparse as sp

def main(args):
    inputname, outputname = args
    outPath = r''
    pat_id = inputname.split('\\')[-1][:-4]

    inputCSV = inputname
    outputFilepath = outputname
    progress_filename = os.path.join(r'', f'{pat_id}.txt')
    params = os.path.join(outPath, 'exampleSettings', 'BraTS19_2013_11_1_seg.yaml')

    # Configure logging
    rLogger = logging.getLogger('radiomics')

    # Set logging level
    # rLogger.setLevel(logging.INFO)  # Not needed, default log level of logger is INFO

    # Create handler for writing to log file
    handler = logging.FileHandler(filename = progress_filename, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)

    # Initialize logging for batch log messages
    logger = rLogger.getChild('batch')

    # Set verbosity level for output to stderr (default level = WARNING)
    radiomics.setVerbosity(logging.INFO)

    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading CSV')

    flists = []
    try:
        with open(inputCSV, 'r') as inFile:
            cr = csv.DictReader(inFile, lineterminator='\n')
            flists = [row for row in cr]
    except Exception:
        logger.error('CSV READ FAILED', exc_info=True)

    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists))

    if os.path.isfile(params):
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
    else:  # Parameter file not found, use hardcoded settings instead
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None  # [3,3,3]
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True

        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
        # extractor.enableInputImages(wavelet= {'level': 2})

    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    headers = None
    print(flists)
    for idx, entry in enumerate(flists, start=1):

        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)", idx, len(flists), entry['Image'], entry['Mask'])

        imageFilepath = entry['Image']
        maskFilepath = entry['Mask']
        label = entry.get('lable', None)
        #print(label)

        if str(label).isdigit() and int(label) == 1:
            label = int(label)
        else:
            label = None
        if not os.path.exists(imageFilepath):
            return
        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = collections.OrderedDict(entry)
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

            try:
                featureVector.update(extractor.execute(imageFilepath, maskFilepath, label))

                with open(outputFilepath, 'a') as outputFile:
                    writer = csv.writer(outputFile, lineterminator='\n')
                    if headers is None:
                        headers = list(featureVector.keys())
                        writer.writerow(headers)

                    row = []
                    for h in headers:
                        row.append(featureVector.get(h, "N/A"))
                    writer.writerow(row)
            except Exception:
                logger.error('FEATURE EXTRACTION FAILED', exc_info=True)
def read_features(pat_path):

    df = pd.read_csv(pat_path)
    df = df.iloc[:, 40:]
    df = df.values

    print("___________________________________________________________")
    print(df.shape)
    return df



    #return right
def readexcel1(path):
    raw_data = pd.read_excel(path, header=0)
    data = raw_data.values
    print(data)
    return data

def subtraction(path, left, right):
    arr = read_features(path)
    i = 7
    if left == 0 and right == 0:
        return

    if left == 0 or (left == 1 and right == 1):
        pass
    elif right == 0:
        for k in range(10):
            arr[[k, k + 10], :] = arr[[k + 10, k], :]
    print(arr[0].shape[0])

    arr1 = np.zeros((2, arr[0].shape[0]))
    arr1[0] = arr[i]
    arr1[1] = arr[i + 10]


    matrix = np.array(arr1)
    matrix[1] = matrix[0] - matrix[1]
    save_path = path.replace("features", "subtraction")
    save_path = save_path.replace(".csv", ".mat")
    savemat(save_path, {f"feature": matrix})

def normalization1(feature):
  
    de = list()
    columnmin = feature.min(axis=0)
    columnmax = feature.max(axis=0)
    for i in range(len(columnmin)):
        if abs(columnmax[i] - columnmin[i]) <= 0.0001:
            de.append(i)
    feature = np.delete(feature, de, axis=1)
    return (feature - feature.mean(axis=0)) / feature.std(axis=0)

def normalization(feature):
 
    de = list()

    columnmin = feature.min(axis=0)
    columnmax = feature.max(axis=0)
    for i in range(len(columnmin)):
        if columnmax[i] == columnmin[i]:
            de.append(i)
    print()
    feature = np.delete(feature, de, axis=1)
    columnmin = feature.min(axis=0)
    columnmax = feature.max(axis=0)
    print(feature.shape)
    print(columnmin.shape)

    ranges = columnmax - columnmin
    row = feature.shape[0]
    norma = feature - np.tile(columnmin, (row, 1))
    norma = norma / np.tile(ranges, (row, 1))
    return norma



def preprocess_features(features):
    """Row-normalize feature matrix """
    rowsum = np.array(features.sum(axis=0))
    print(rowsum[0])
    print(rowsum.shape)
    r_inv = np.power(rowsum, -1).flatten()
    print(r_inv.shape)

    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features
def preprocess_features1(features):
    rowmax = np.max(features,axis=1)
    rowmin = np.min(features,axis=1)
    rowsub = rowmax - rowmin
    r_inv = np.power(rowsub, -1).flatten()





if __name__ == '__main__':

    info = data[:, 0]
    info1 = data1[:, 0]
    feature_path = os.path.join(path_base, info[1] + ".csv")

    print(info1.shape)
    print(info.shape)
    for i in range(1, 100):
       if os.path.exists(os.path.join(path, info[i])):

            feature_path = os.path.join(path_base, info[i] + ".csv")
            print(feature_path)
            #print(os.path.join(path, info[i]))
            subtraction(feature_path, data[i][1], data[i][2])

    for i in range(1, len(info1)):
        if os.path.exists(os.path.join(path, f"{info1[i]}")):
            if info1[i] == 2186645:
                continue


            feature_path = os.path.join(path_base, f"{info1[i]}" + ".csv")
            print(feature_path)
            if data1[i][2] == "Left":
                left = 0
                right = 1
            elif data1[i][2] == "Right":
                right = 0
                left = 1
            elif data1[i][2] == "neither":
                right = 0
                left = 1
            else:
                continue
            subtraction(feature_path, left, right)
