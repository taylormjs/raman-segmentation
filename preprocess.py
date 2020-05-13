'''
Script containing useful pre-processing functions on matrices prior
to input into ML models. This assumes matrices have already been loaded
from any previous files (e.g. excel, .dat, .m, etc)
'''


# native imports
import os
from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import scipy.io # for plotting .mat files
import seaborn as sns
import scanpy as sc
import anndata as ann


# third party imports
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, QuantileTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score
from scipy.signal import savgol_filter
from functools import partial
import imblearn
import pickle
import gzip # for compression/decompression
import rampy as rp


# local imports


''' Make sure module is imported correctly'''
def module_imported():
    print('preprocess imported')
    print('module name : {} module package: {}'.format(__name__, __package__))
    
module_imported()
    
    
#####################################################
####### Basic functions for Dataframe Editing #######
#####################################################
    
    
def split_column(df, col_to_split, delim='_', col1_name='day', col2_name='position',
                    col1_ix=1, col2_ix=2):
    ''' 
        Given a df, splits column col_to_split by delimiter
        and adds to df as two columns
    '''
    assert (col1_name not in df.columns), 'column 1 already exists'
    assert (col2_name not in df.columns), 'columns 2 already exists'
    split_name = [str.split(name, sep=delim) for name in df[col_to_split]]
    col1 = [e[0] for e in split_name]
    col2 = [e[1] if len(e)>1 else e[0] for e in split_name]
    df.insert(col1_ix, col1_name, col1, False)
    df.insert(col2_ix, col2_name, col2, False)
    return df


def to_dataframe(matrix, label_df, column_names=None):
    ''' 
        Given a matrix (numpy), converts to pandas DataFrame with given
        labels and column names. Labels are needed for easy plotting by day
    '''
    assert matrix.shape[0] == label_df.shape[0], 'matrix and labels are not the same shape, cannot be concatenated'
    if column_names:
        # cast to pandas matrix with column names
        spectra_df = pd.DataFrame(data=matrix, columns=column_names)
    else:
        spectra_df = pd.DataFrame(data=matrix)
    # concatenate labels with matrix along axis=1
    return pd.concat([label_df, spectra_df], axis=1) 


def get_fingerprint_region(matrix, start_ix, end_ix=None):
    '''
        Slices matrix to extract fingerprint region defined by df[:,start_ix:end_ix]
        Assumes df is a numpy matrix
    '''
    assert isinstance(matrix, np.ndarray), 'input matrix is of type {}, but should be of type np.ndarray'.format(type(matrix))
    
    if end_ix: return matrix[:, start_ix:end_ix]
    else: return matrix[:, start_ix:]
    
    # TODO - maybe add pandas dataframe functionality later
#     elif isinstance(df, pd.core.frame.DataFrame):
#         if end_ix: return pd.concat([df.loc[:, :last_col_label], df.loc[:,start_ix:end_ix]], axis=1)
#         else: return pd.concat([df.loc[:, :last_col_label], df.loc[:,start_ix:]], axis=1)
#     return -1

#####################################################
################### Peak Removal ####################
#####################################################


def remove_outliers_IQR(matrix, outlier_reg=1.5, axis=0):
    ''' 
    Checks for outliers based on the IQR for each column/feature (wavenumber for raman spectra),
    replaces outliers with median values for each feature
    
    Note: function works as intended, but it creates discontinuities in the spectra, significantly
    changing intensities. Consider removing this function
    
    Returns: numpy matrix with outliers removed, boundaries of the outliers
    '''
    assert isinstance(matrix, np.ndarray), 'input matrix is of type {}, but should be of type np.ndarray'.format(type(matrix))

    # get interquartile range
    upper_quartile = np.percentile(matrix, 75, axis=axis)
    lower_quartile = np.percentile(matrix, 25, axis=axis)
    IQR = (upper_quartile - lower_quartile) * outlier_reg
    outlier_bounds = (lower_quartile - IQR, upper_quartile + IQR)
    
    # replace those outside of outlier_bounds with median for each feature
    matrix_outliers_removed = np.where((matrix > outlier_bounds[0]) & (matrix < outlier_bounds[1]), matrix, np.median(matrix, axis=0)) 
    
    return matrix_outliers_removed, outlier_bounds


################## Whitaker Hayes Method #################
# uses the next three fns: get_peaks, remove_sharp_peaks_1d, and remove_sharp_peaks_2d 
# only remove


def get_peaks(intensity, threshold):
    '''
        Produces a mask where True values have a peak 
        greater than given threshold
    '''

    median_int = np.median(intensity)
    mad_int = np.median([np.abs(intensity - median_int)])
    modified_z_scores = np.abs(0.6745 * (intensity - median_int) / mad_int)
    peaks = modified_z_scores > threshold
    return peaks


def get_spectra_with_peaks(spectra, labels, threshold, make_plots=False, dir_to_save=None, 
                           n_cols=6, n_rows=None, figsize=(25, 30), verbose=True):
    '''
        Returns a list of the row indices marked as outliers, according to 
        the Whitaker-Hayes algorithm : https://chemrxiv.org/articles/A_Simple_Algorithm_for_Despiking_Raman_Spectra/5993011/2?file=10761493
        
        @params: 
            - spectra : 2D pandas numpy array, each row is a spectra
            - labels : pandas dataframe with columns for 'day', 'position', and 'cell' 
            - threshold: (try 3-7) z-score threshold above which an intensity is considered a peak 
                - threshold=5.25 worked well for cellular reprogramming dataset
            - make_plots : if True, will produce plots of all spectra (default is True)
            - dir_to_save : if path is given, will save a pdf at that path
            - n_cols : number of columns in the plot output
            - n_rows : number of rows in plot output. If none, automatically inferred from n_cols
            - fig_size: size of plotting figures
            
        @return: indices (row numbers) of spectra marked as outliers
    '''
    # make sure inputs are all okay
    assert isinstance(spectra, np.ndarray), f'input matrix is of type {type(spectra)}, but should be of type np.ndarray'
    for label in ['day', 'position', 'cell']:
        if label not in labels.columns:
            print(f'{label} does not appear in labels')
    if dir_to_save: assert make_plots, f'if providing dir_to_save, make_plots must be True, but make_plots={make_plots}'
   
    # initialize variables
    if not n_rows: n_rows = spectra.shape[0]//n_cols
    positive_ix = [] # indices of spectra marked as positives
    if dir_to_save:
        filter_path = os.path.join(dir_to_save, f'outliers_thresh_{threshold}.pdf')
    else: filter_path = os.path.join(os.getcwd(), 'dummy') # TODO: we may want to make this into two functions -- one for plotting
                                                                    # and one for finding spectra. Keeping them as one requires this dummy
    with PdfPages(filter_path) as export_pdf:
        plot_ix = 0 # counts number of positives
        if make_plots:
            # set up plots
            fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axs.flatten()
        for (ix, spec) in tqdm(enumerate(spectra)):
            
            # determine whether an outlier
            delta_int = np.diff(spec) # take first derivative of spectra
            peaks = get_peaks(delta_int, threshold=threshold)
            low_b, hi_b = np.min(spec), np.max(spec)
            num_peaks = np.sum(peaks)
            
            # if an outlier, plot
            if num_peaks > 0:
                label_ix = labels.iloc[ix,:]
                positive_ix.append(ix)
                if make_plots:
                    try:
                        day, pos, cell = label_ix.loc[['day', 'position', 'cell']]
                        title = f'Outlier detected at {day} {pos} {cell}'
                    except:
                        title = f'Outlier detected on day {label_ix.iloc[0]}'
                    axes[plot_ix].plot(spec)
                    axes[plot_ix].plot(peaks*low_b+(hi_b - low_b)/5, color='orange', marker='^')
                    axes[plot_ix].set_ylim(low_b, hi_b)
                    axes[plot_ix].set_title(title)#, size=10)
                plot_ix += 1
        if make_plots: 
            plt.tight_layout()
        if dir_to_save:
            export_pdf.savefig()
    if verbose: print(f'Found {plot_ix} spectra')

    return positive_ix


def get_spectra_without_peaks(spectra, labels, threshold, make_plots=False, dir_to_save=None,
                              n_cols=6, n_rows=None, figsize=(25,400), verbose=True):
    '''
        Plots the negatives
    '''   
    
    # initialize variables
    if not n_rows: n_rows = spectra.shape[0]//n_cols
    positive_ix = set(get_spectra_with_peaks(spectra, labels, threshold, make_plots=False, verbose=False)) #making into a set for efficient calling
    if dir_to_save:
        filter_path = os.path.join(dir_to_save, f'negatives_thresh_{threshold}.pdf')
    else: filter_path = os.path.join(os.getcwd(), 'dummy') # TODO: we may want to make this into two functions -- one for plotting
                                                                    # and one for finding spectra. Keeping them as one requires this dummy
    
    negative_ix = []
    with PdfPages(filter_path) as export_pdf:
        plot_ix = 0 # counts number of negatives
        if make_plots:
            # set up plots
            fig, axs = plt.subplots(n_rows, n_cols, figsize=figsize)
            axes = axs.flatten()
        for (ix, spec) in tqdm(enumerate(spectra)):
            # if not an outlier, plot
            if ix not in positive_ix:
                low_b, hi_b = np.min(spec), np.max(spec)
                label_ix = labels.iloc[ix,:]
                day, pos, cell = label_ix.loc[['day', 'position', 'cell']]
                negative_ix.append(ix)
                if make_plots:
                    title = f'Negative at {day} {pos} {cell}'
                    axes[plot_ix].plot(spec)
                    axes[plot_ix].set_ylim(low_b, hi_b)
                    axes[plot_ix].set_title(title)#, size=10)
                plot_ix += 1
        if make_plots: 
            plt.tight_layout()
        if dir_to_save:
            export_pdf.savefig()
    print(f'Found {plot_ix} negatives')

    return negative_ix


def remove_sharp_peaks_1d(intensity, win_size, threshold):
    '''
        Removes peaks from intensity vector by replacing them with 
        the mean of non-peak values within a given window size
        
        Threshold represents the z-score threshold. We found that a threshold
        of 5.25 works well for the cellular reprogramming Raman dataset
    '''
    n = len(intensity)
    delta_int = np.diff(intensity) #take first derivative to show sharp jumps
    peaks = get_peaks(delta_int, threshold)
    y_out = intensity.copy()
    for i in np.arange(len(peaks)):
        # if we have a peak, replace with mean of surrounding non-peaks
        if peaks[i] > 0:
            # get surrounding window indices
            # prevent out of boundary errors
            if i-win_size <0: lower = 0
            else: lower = i-win_size
            
            if i+1+win_size >= n: upper = n-1
            else: upper = i+1+win_size
            
            win_ind = np.arange(lower, upper)
            win_mask = win_ind[peaks[win_ind] == 0] # take the ones that are not peaks
            y_out[i] = np.mean(intensity[win_mask]) # take the mean of these surrounding values
    return y_out


def remove_sharp_peaks_2d(matrix, win_size, threshold, axis=1):
    '''
        Removes sharp peaks from all 1D vectors in a given matrix
    '''
    return np.apply_along_axis(remove_sharp_peaks_1d, 
                        axis=axis, 
                        arr=matrix, 
                        win_size=win_size, 
                        threshold=threshold)


def remove_sharp_peaks_1d_iter(intensity, win_size, threshold, max_iter=10):
    '''
        Removes peaks from intensity vector by replacing them with 
        the mean of non-peak values within a given window size
        
        Threshold represents the z-score threshold. We found that a threshold
        of 5.25 works well for the cellular reprogramming Raman dataset
    '''
    n = len(intensity)
    delta_int = np.diff(intensity) #take first derivative to show sharp jumps
    peaks = get_peaks(delta_int, threshold)
    num_peaks = sum(peaks)
    y_out = intensity.copy()
    win_ind = np.arange(n)
    iter_ = 0
    while iter_ < max_iter and num_peaks > 0: #include while loop to iteratively remove peaks
        for i in np.arange(len(peaks)):
            # if we have a peak, replace with mean of surrounding non-peaks
            if peaks[i] > 0:
                # get surrounding window indices
                # prevent out of boundary errors
                if i-win_size <0: lower = 0
                else: lower = i-win_size

                if i+1+win_size >= n: upper = n-1
                else: upper = i+1+win_size

                win_ind = np.arange(lower, upper)
    #             print(f'window: {win_ind}')
                win_mask = win_ind[peaks[win_ind] == 0] # take the ones that are not peaks
                y_out[i] = np.mean(intensity[win_mask]) # take the mean of these surrounding values
        # get new number of peaks after correction
        delta_int = np.diff(y_out) #take first derivative to show sharp jumps
        peaks = get_peaks(delta_int, threshold)
        num_peaks = sum(peaks)
        iter_ += 1
    return y_out


def remove_sharp_peaks_2d_iter(matrix, win_size, threshold, axis=1, max_iter=20):
    '''
        Removes sharp peaks from all 1D vectors in a given matrix
    '''
    return np.apply_along_axis(remove_sharp_peaks_1d_iter, 
                        axis=axis, 
                        arr=matrix, 
                        win_size=win_size, 
                        threshold=threshold,
                        max_iter=max_iter)


#####################################################
########## Autofluorescence correction ##############
#####################################################


def remove_fluorescence(matrix, method='als', x_axis=np.arange(410,1340)):
    ''' 
        Applies flouorescence removal using rampy baseline correction
        to each row of a matrix
        @ params: 
        - matrix is numpy array of spectra before fluorescence correction
        - method is one of ['poly', 'als', 'arPLS', 'drPLS']
        - x_axis is 
        @ returns: np array of spectra after baseline fluorescence correction (spectra_bc)
        ]
    '''
    assert len(x_axis) == matrix.shape[1], 'x_axis ({}) and matrix ({}) are not the same dimension'.format(x_axis.shape, matrix.shape)
    # find parameters for rampy.baseline
    x_axis[0], x_axis[-1]
    lower_y, upper_y = np.min(matrix), np.max(matrix)

    # apply a wrapper to rampy.baseline function using lambda function
    baseline_wrapper = lambda y, x, bir, method: rp.baseline(x, y, bir, method)

    # apply function along axis
    spectra_bc_tup = np.apply_along_axis(baseline_wrapper, axis=1, arr=matrix, x=x_axis, method=method,
                                         bir=np.array([[x_axis[0], x_axis[-1]],[lower_y, upper_y]]))

    # remove spectra and extra dimensions
    spectra_bc = spectra_bc_tup[:,0,:,:].squeeze()
    return spectra_bc


#####################################################
############## Mean substraction ####################
#####################################################

def subtract_mean_horizontal(df):
    '''
        standardizes by subtracting mean horizontally, moving all 
        to around 0
    '''
    return df - np.mean(df, axis=1, keepdims=True)





########### UNFINISHED AND MAYBE UNNCESSARY CODE ############
# ''' Parameter search -- goal is to choose parameters to find all 23 True positives '''

# for thresh in [5.25]:
#     positives_par_search = []
#     filter_path = os.path.join(root, 'results', 'cell_reprogramming', 'preprocessing_plots', f'outliers{thresh}.pdf')
#     with PdfPages(filter_path) as export_pdf:
#             fig, axs = plt.subplots(6,6, figsize=(25,20))
#             axes = axs.flatten()
#             plot_ix = 0
#             for (ix, spectra) in enumerate(spectra_fp):
#                 delta_int = np.diff(spectra)
#                 peaks = get_peaks(delta_int, threshold=thresh)
#                 low_b, hi_b = np.min(spectra_fp[ix]), np.max(spectra_fp[ix])
#                 num_peaks = np.sum(peaks)
#                 if num_peaks > 0:
#                     positives_par_search.append(f'{day} {pos} {cell}')
#                     day, pos, cell = spectra_fp_dataframe.iloc[ix, [1, 2, 4]]
#                     # if we classified a false negative as a positive, assign it green (G00D)
#                     if f'{day} {pos} {cell}' in false_negatives:
#                         title = f'Previously False Negative: \n{day} {pos} {cell}'
#                         title_color = 'green'
#                     # if we classified a True negative as positive, assign it red (BAD)
#                     elif f'{day} {pos} {cell}' not in true_positives:
#                         title = f'Previously True Negative: \n{day} {pos} {cell}'
#                         title_color = 'red'
#                     # Otherwise it was a true positive we already found, so we keep as black 
#                     else:
#                         title = f'Outlier predicted for {day} {pos} {cell}\nThreshold: {thresh}\n Image count: {plot_ix+1}'
#                         title_color = 'black'
# #                     title = f'Negative at {day} {pos} {cell}'
#                     axes[plot_ix].plot(spectra)
#                     axes[plot_ix].set_title(title, color=title_color)#, size=10)
#                     axes[plot_ix].plot(peaks*low_b+(hi_b - low_b)/10, color='orange', marker='^')
#                     axes[plot_ix].set_ylim(low_b, hi_b)
# #                     axes[plot+ix].annotate('local max', xy=(2, 1), xytext=(3, 1.5),
# #                                 arrowprops=dict(facecolor='black', shrink=0.05))
#                     plot_ix += 1
#             plt.tight_layout()
#             export_pdf.savefig()
#     print(f'Found {plot_ix} spectra classified as positive with thresh = {thresh}')