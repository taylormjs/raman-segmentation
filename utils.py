# Script for holding useful helper functions

import os
from matplotlib import pyplot as plt
import numpy as np
import xlrd # for excel files
import pickle
import gzip # for compression/decompression
import pandas as pd
from tqdm import tqdm # for progress bar
import scipy.io # for plotting .mat files



def TODO(message=None):
    ''' raises a ValueError flag to remind you to complete the todo'''
    if not message: raise NotImplementedError('TODO')
    raise NotImplementedError('TODO: {}'.format(message))
    
    

def check_matrix_info(matrix, matrix_name=None):
    ''' print out matrix name, shape, and type '''
    # get shape
    shape = matrix.shape
    # get type
    if isinstance(matrix, np.ndarray):
        type_ = matrix.dtype
    else: type_ = type(matrix)
    if matrix_name:
        print("Name : {}  Shape : {}  Type : {}".format(matrix_name, shape, type_))
    else:
        print("Shape : {}  Type : {}".format(shape, type_))
        
####################################################
########### Loading and unloading files ############    
####################################################

def write_zipped_pickle(obj, filename, protocol=-1):
    """

    :param obj:
    :param filename:
    :param protocol:
    :return:
    """
    with gzip.open(filename, 'wb') as f:
        pickle.dump(obj, f, protocol)


def read_pickle(filename):
    """
    Read pickle file which may be zipped or not
    :param filename:
    :return:
    """
    try:
        with gzip.open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
    except OSError:
        with open(filename, 'rb') as f:
            loaded_object = pickle.load(f)
            return loaded_object
    
    