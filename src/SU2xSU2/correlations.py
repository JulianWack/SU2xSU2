# Routines to compute correlations between samples of observables 

import numpy as np
import time
from astropy.stats import jackknife_stats


def auto_window(IATs, c):
    '''Windowing procedure of Caracciolo, Sokal 1986 to truncate the sum for the IAT.

    IATs: array
        integrated autocorrelation time with increasingly late termination of the sum
    c: float
        defines the window width. For correlations that decay exponentially, a value of 4 of 5 is conventional

    Returns index for array IATs, representing the IAT estimate from the windowing procedure
    ''' 
    ts = np.arange(len(IATs)) # all possible separation endpoints
    m =  ts < c * IATs # first occurrence where this is false gives IAT 
    if np.any(m):
        return np.argmin(m)
    return len(IATs) - 1


def autocorr_func_1d(x):
    '''Computes the autocorrelation of a 1D array x using FFT and the Wiener Khinchin theorem.
    As FFTs yield circular convolutions and work most efficiently when the number of elements is a power of 2, pad the data with zeros to the next power of 2. 
    '''
    x = np.atleast_1d(x)
    if len(x.shape) != 1:
        raise ValueError("invalid dimensions for 1D autocorrelation function")

    def next_pow_two(n):
        i = 1
        while i < n:
            i = i << 1 # shifts bits to the left by one position i.e. multiplies by 2 
        return i
    
    n = next_pow_two(len(x))

    # Compute the FFT and then the auto-correlation function
    f = np.fft.fft(x - np.mean(x), n=2 * n)
    acf = np.fft.ifft(f * np.conjugate(f))[: len(x)].real
    acf /= 4 * n
    # normalize to get autocorrelation rather than autocovariance
    acf /= acf[0]

    return acf


def autocorrelator_repeats(data, c=4.0):
    '''Based on the implementation in emcee: https://emcee.readthedocs.io/en/stable/tutorials/autocorr/
    Computes the autocorrelation function and integrated autocorrelation time for passed data which is a 2D array such that each row represents a sample of observations.
    The correlations are computed between rows of the data by finding the 1D autocorrelation function for each column of the data. The overall ACF between rows is estimated
    as the average of correlations across columns. This also allows to get an estimate of the error of the ACF.
    An alternative is to average along rows first and to estimate the ACF as the autocorrelation of the final column (Goodman, Weare 2010)

    data: 2D array (M,N)
        each row contains N samples of the same observable while different rows correspond to different positions in the chain of M observations.
        The correlation is computed across axis 1 i.e. gives the AFC for samples from different positions in the chain. 

    Returns
        ts: array (M,)
            array of considered separations between two samples
        ACF: array (M,)
            autocorrelation function
        ACF_err:
            error of the autocorrelation function
        IAT: float
            integrated autocorrelation time, showing how many rows in the data lie between uncorrelated samples
        IAT_err: float
            error of the autocorrelation time
        delta_t: float
            time needed to compute the ACF and the IAT
    '''
    M, N = data.shape
    ts = np.arange(M)
    t1 = time.time()

    # get ACF and its error
    ACFs = np.zeros_like(data)
    for i in range(N):
        ACFs[:,i] = autocorr_func_1d(data[:,i])

    ACF, ACF_err = np.mean(ACFs, axis=1), np.std(ACFs, axis=1) / np.sqrt(N)

    # get all possible IAT and apply windowing
    IATs = 2.0 * np.cumsum(ACF) - 1.0 # IAT defined as 1 + sum starting from separation=1, but cumsum starts with t=0 for which ACF=1
    break_idx = auto_window(IATs, c)
    IAT = IATs[break_idx]
    IAT_err = np.sqrt((4*break_idx+2)/data.shape[0]) * IAT #  Madras, Sokal 1988

    t2 = time.time()
    delta_t = t2-t1

    return ts, ACF, ACF_err, IAT, IAT_err, delta_t


def autocorrelator(data, c=4.0):
    '''Alias for autocorrelator_repeats when the data has been observed only once.
    data is of shape (M,)'''
    return autocorrelator_repeats(data.reshape((data.shape[0], 1)))


def corr_func_1D(x, y):
    '''Computes the correlation between the equal length 1D arrays x, y using the cross correlation theorem.
    FFTs yield circular convolutions, such that x and y are assumed to have periodic boundary conditions. 
    When the data is not circular, need to pad it with zeros as done in autocorr_func_1d.

    Returns
    cf: 1D array of same length as x and y
        correlation function between the data in x and y   
    '''
    f = np.fft.fft(x)
    g = np.fft.fft(y)
    cf = np.fft.ifft(f * np.conjugate(g)).real

    return cf


def correlator_repeats(xs, ys):
    '''Find correlation function using FFTs between two equally sized 1D arrays for which several measurements exists. 
    Each column presents a new observation and correlations are computed along axis 0. The final CF is the average along axis 1.

    xs, ys: (N,M) array
        M measurements of a data vector of length N

    Returns
    CF: (N,) array
        average correlation function based on the M measurements 
    CF_err (N,) array
        IAT corrected SEM of the CF
    '''
    N, M = xs.shape # length of one data measurement, number of measurements

    # get ACF and its error
    CFs = np.zeros_like(xs)
    for i in range(M):
        CFs[:,i] = corr_func_1D(xs[:,i], ys[:,i])

    CF, CF_err = np.mean(CFs, axis=1), np.std(CFs, axis=1) / np.sqrt(M)
    
    # correct error by IAT
    for i in range(N):
        ts, ACF, ACF_err, IAT, IAT_err, delta_t = autocorrelator(CFs[i])
        CF_err[i] *= np.sqrt(IAT)

    return CF, CF_err


def correlator(xs, ys):
    '''Alias for correlator_repeats when the x and y have been observed only once.
    xs and ys are arrays of shape (N,)
    Note that this implies, no error on the correlation function can be estimated.'''
    return correlator_repeats(xs.reshape((xs.shape[0], 1)), ys.reshape((ys.shape[0], 1)))