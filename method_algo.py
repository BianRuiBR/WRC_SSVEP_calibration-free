import math

from scipy import signal
import numpy as np

def preprocess(filterB, filterA, data):
    notchedData = signal.filtfilt(filterB, filterA, data)
    fs = 250 / 2
    N, Wn = signal.ellipord(90 / fs, 100 / fs, 3, 60)
    b, a = signal.ellip(N, 1, 60, Wn)
    filtedData = signal.filtfilt(b, a, notchedData)
    return filtedData

def preprocess_TRCA(filterB, filterA, data):

    notchedData = signal.lfilter(filterB, filterA, data)
    b, a = signal.butter(8, [0.056, 0.72], btype='band')
    filtedData = signal.filtfilt(b, a, notchedData)
    return filtedData


def methmean(X, windowLength):
    meanX = np.zeros(X.shape)
    channelNum, samplsNum = X.shape
    padX = np.zeros((channelNum, samplsNum + windowLength * 2 - 2))
    padX[:, windowLength - 1: windowLength + samplsNum - 1] = X
    for i in range(channelNum):
        for j in range(samplsNum):
            meanX[i, j] = np.mean(padX[i, j:j + windowLength])
    return meanX


def fb_filter(fs, idx_fb):
    fs = fs / 2
    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    fbfilterBSet = []
    fbfilterASet = []
    for i in range(idx_fb):
        wp = [passband[i] / fs, 90 / fs]
        ws = [stopband[i] / fs, 100 / fs]
        [n, wn] = signal.cheb1ord(wp, ws, 2, 40)
        [b, a] = signal.cheby1(n, 0.1, wn, 'band')
        fbfilterBSet.append(b)
        fbfilterASet.append(a)
    return fbfilterBSet, fbfilterASet

def preprocess_all(data):
    fs = 250
    f0 = 50
    Q = 35
    b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
    filtedData = signal.filtfilt(b, a, data)
    return filtedData

def filterBankProcess(num_of_subband, data):
    srate = 250
    fs = srate / 2
    # filter bank
    filtedData = np.zeros([num_of_subband, data.shape[0], data.shape[1]])
    for Nband in range(0, num_of_subband):
        wp = 6 * (Nband + 1) / fs
        # wp = (6 + Nband*1)/fs
        ws = 100 / fs

        b, a = signal.butter(8, [wp, ws], 'bandpass')
        filtedData[Nband, :, :] = signal.filtfilt(b, a, data)
    return filtedData

def softmax0(inMatrix):
    Num = len(inMatrix)
    outP = np.zeros((Num, 1))
    soft_sum = 0
    for idx in range(0, Num):
        outP[idx] = math.exp(inMatrix[idx])
        soft_sum += outP[idx]
    for idx in range(0, Num):
        outP[idx] = outP[idx] / soft_sum
    return outP

def preprocess_LQXZ(data):
    fs = 250
    f0 = 50
    Q = 35
    b, a = signal.iircomb(f0, Q, ftype='notch', fs=fs)
    filtedData_notch = signal.filtfilt(b, a, data)

    fs = 250 / 2
    N, Wn = signal.ellipord(100 / fs, 105 / fs, 3, 60)
    b, a = signal.ellip(N, 0.1, 60, Wn)
    order = 2
    z, p, k = signal.tf2zpk(b, a)
    zn = np.tile(z, order)
    pn = np.tile(p, order)
    kn = k ** order
    bn, an = signal.zpk2tf(zn, pn, kn)

    filtedData_ellip = signal.filtfilt(bn, an, filtedData_notch)
    return filtedData_ellip