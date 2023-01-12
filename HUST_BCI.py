import pickle
import time
import os
import warnings
import numpy as np
import scipy
from scipy.linalg import sqrtm


class CCAClass_HUST:
    def __init__(self):
        pass

    def initial(self, targetTemplateSet, TRCAID, TRCAWeight, ECCAWeight, andOrChoose, subjectCalTime, blockTrain,
                yuall):
        self.targetTemplateSet = targetTemplateSet
        self.TRCAID = TRCAID
        self.TRCAWeight = TRCAWeight
        self.ECCAWeight = ECCAWeight
        self.andOrChoose = andOrChoose
        self.subjectCalTime = subjectCalTime
        self.blockTrain = blockTrain
        self.yuall = yuall
        subNum = 2
        self.eccaTemplateAll = [[[None] for i in range(subNum)] for j in range(TRCAID.shape[0])]
        self.W_TRCATemplateAll = [[[None] for i in range(subNum)] for j in range(TRCAID.shape[0])]
        for person_i in range(subNum):
            for j in range(TRCAID.shape[0]):
                filesName = os.path.dirname(__file__) + '/Model/HUST/' + str(
                    self.TRCAID[j]) + '.pkl'
                model = open(filesName, 'rb')
                model = pickle.load(model)
                self.eccaTemplateAll[j][person_i] = [np.expand_dims(c, axis=0) for c in model[1]]
                self.W_TRCATemplateAll[j][person_i] = model[0]

    def recognize(self, dataCCA, dataTRCA, personID, nowCalIime, blockNumNow, isNew):
        results = None
        '''if personID != 10:
           results = 0
           return results, subjectSmallTime, bigtag, isNew'''
        if blockNumNow < self.blockTrain[personID - 1]:
            p_FBCCA = fbcca(dataCCA,
                            self.targetTemplateSet)  # weighted sum of r from all different filter banks' result
            yFBCCA = p_FBCCA.argmax() + 1
            p_CCA = cca(dataCCA, self.targetTemplateSet)
            yCCA = p_CCA.argmax() + 1
            results = yFBCCA + 0
            yFBCCA = int(yFBCCA)
            yu = yuzhi(p_FBCCA)

            yCCA = int(yCCA)
            if yFBCCA == yCCA:
                isNew[personID - 1, yFBCCA - 1] = 1
                for j in range(self.TRCAID.shape[0]):
                    temp = np.expand_dims(dataTRCA[:, 0:self.TRCAID[j]], axis=0)
                    self.eccaTemplateAll[j][personID - 1][yFBCCA - 1] = np.concatenate(
                        (self.eccaTemplateAll[j][personID - 1][yFBCCA - 1], temp), axis=0)
                    self.W_TRCATemplateAll[j][personID - 1][:, yFBCCA - 1] = trca_matrix(
                        self.eccaTemplateAll[j][personID - 1][yFBCCA - 1])[:, 0]
            else:
                if isNew[personID - 1, yFBCCA - 1] == 0:
                    # isNew[personID - 1, yFBCCA - 1] = 1
                    for j in range(self.TRCAID.shape[0]):
                        temp = np.expand_dims(dataTRCA[:, 0:self.TRCAID[j]], axis=0)
                        self.eccaTemplateAll[j][personID - 1][yFBCCA - 1] = np.concatenate(
                            (self.eccaTemplateAll[j][personID - 1][yFBCCA - 1], temp), axis=0)
                        self.W_TRCATemplateAll[j][personID - 1][:, yFBCCA - 1] = trca_matrix(
                            self.eccaTemplateAll[j][personID - 1][yFBCCA - 1])[:, 0]
                if isNew[personID - 1, yCCA - 1] == 0:
                    for j in range(self.TRCAID.shape[0]):
                        temp = np.expand_dims(dataTRCA[:, 0:self.TRCAID[j]], axis=0)
                        self.eccaTemplateAll[j][personID - 1][yCCA - 1] = np.concatenate(
                            (self.eccaTemplateAll[j][personID - 1][yCCA - 1], temp), axis=0)
                        self.W_TRCATemplateAll[j][personID - 1][:, yCCA - 1] = trca_matrix(
                            self.eccaTemplateAll[j][personID - 1][yCCA - 1])[:, 0]
            return results, isNew

        # ecca
        ECCATemplate = self.eccaTemplateAll[nowCalIime][personID - 1]
        p_ECCA = ecca(dataTRCA, self.targetTemplateSet, ECCATemplate)
        yECCA = p_ECCA.argmax() + 1

        # etrca
        p_eTRCA = etrca_withmodel(ECCATemplate, self.W_TRCATemplateAll[nowCalIime][personID - 1], dataTRCA)
        yeTRCA = p_eTRCA.argmax() + 1

        # fbcca
        p_FBCCA = fbcca(dataCCA, self.targetTemplateSet)  # weighted sum of r from all different filter banks' result
        yFBCCA = p_FBCCA.argmax() + 1

        # cca
        p_CCA = cca(dataCCA, self.targetTemplateSet)
        yCCA = p_CCA.argmax() + 1

        pAll = p_FBCCA + p_CCA + p_eTRCA + p_ECCA
        yAll = pAll.argmax() + 1  # index indicate the maximum(most possible) target
        # yallcat = np.array([yCCA, yFBCCA, yeTRCA, yECCA])
        yu = yuzhi(pAll)
        '''if yECCA == yeTRCA and yECCA == yAll and (yeTRCA == yFBCCA or yeTRCA == yCCA) or yu < -0.003:
            # if yeTRCA==yFBCCA and yeTRCA==yCCA:
            bigtag = 1
            temp = np.expand_dims(dataTRCA[:, 0:self.TRCAID[nowCalIime]], axis=0)
            self.eccaTemplateAll[nowCalIime][personID - 1][int(yFBCCA - 1)] = np.concatenate(
                (self.eccaTemplateAll[nowCalIime][personID - 1][int(yFBCCA - 1)], temp), axis=0)
            self.W_TRCATemplateAll[nowCalIime][personID - 1][:, int(yFBCCA - 1)] = trca_matrix(
                self.eccaTemplateAll[nowCalIime][personID - 1][int(yFBCCA - 1)])[:, 0]'''
        if nowCalIime < self.subjectCalTime:
            if self.andOrChoose[nowCalIime] == 1:
                if yeTRCA == yCCA or yeTRCA == yFBCCA:
                    results = yeTRCA
                elif yu < self.yuall[nowCalIime]:
                    results = yAll
                else:
                    results = None
            '''if self.andOrChoose[nowCalIime] == 2:
                if yECCA == yeTRCA and yECCA == yAll and (yeTRCA == yFBCCA or yeTRCA == yCCA):
                    # if yeTRCA==yFBCCA and yeTRCA==yCCA:
                    results = yeTRCA
                elif yu < self.yuall[nowCalIime]:
                    results = yAll
                else:
                    results = None'''
            if self.andOrChoose[nowCalIime] == 2:
                if yECCA == yeTRCA and yECCA == yAll and (yeTRCA == yFBCCA or yeTRCA == yCCA):
                    # if yeTRCA==yFBCCA and yeTRCA==yCCA:
                    results = yeTRCA
                else:
                    results = None
            if self.andOrChoose[nowCalIime] == 3:
                if yeTRCA == yFBCCA or yeTRCA == yCCA:
                    results = yAll
                elif yu < self.yuall[nowCalIime]:
                    results = yAll
                else:
                    results = None
        else:
            results = yAll

        return results, isNew


def etrca(dataTRCA, W_trca, eTRCAtemplate):
    num_sampls = dataTRCA.shape[1]
    p_eTRCA = np.zeros(40)
    for frequencyIndex in range(0, 40):
        XTestTempw = np.dot(W_trca.T, dataTRCA)
        templateTempw = np.dot(W_trca.T, np.squeeze(eTRCAtemplate[frequencyIndex]))
        # templateTempw = np.dot(W_trca.T, np.squeeze(eTRCAtemplate[frequencyIndex].mean(axis=0)))
        ss = np.corrcoef(XTestTempw.reshape((1, 40 * num_sampls)), templateTempw.reshape((1, 40 * num_sampls)))
        p_eTRCA[frequencyIndex] = ss[0, 1]
    return p_eTRCA


def fbcca(dataCCA, targetTemplateSet):
    fb_coefs = np.power(np.arange(1, 5 + 1), (-1.25)) + 0.25
    _, num_smpls = dataCCA.shape  # 40 taget (means 40 fre-phase combination that we want to predict)
    y_ref = targetTemplateSet
    # result matrix
    r = np.zeros((5, 40))
    # deal with one target a time
    for fb_i in range(5):  # filter bank number, deal with different filter bank
        testdata = filterbank(dataCCA, 250, fb_i)  # data after filtering
        testdata = testdata.T
        [Q_temp, R_temp] = np.linalg.qr(testdata)
        for class_i in range(40):
            template = np.squeeze(y_ref[class_i])
            template = template[:, 0:testdata.shape[0]]
            template = template.T
            [Q_cs, R_cs] = np.linalg.qr(template)
            data_svd = np.dot(Q_temp.T, Q_cs)
            [U, S, V] = np.linalg.svd(data_svd)
            # rho = 1.25 * S[0] + 0.67 * S[1] + 0.5 * S[2]
            rho = 1.25 * S[0] + 0.67 * S[1] + 0.5 * S[2]
            r[fb_i, class_i] = rho

    p_FBCCA = np.dot(fb_coefs, r)
    return p_FBCCA


def cca(dataCCA, targetTemplateSet):
    p_CCA = np.zeros(40)
    dataCCA = dataCCA.T
    # qr分解,data:length*channel
    [Q_temp, R_temp] = np.linalg.qr(dataCCA)
    for frequencyIndex in range(0, len(targetTemplateSet)):
        template = targetTemplateSet[frequencyIndex]
        template = template[:, 0:dataCCA.shape[0]]
        template = template.T
        [Q_cs, R_cs] = np.linalg.qr(template)
        data_svd = np.dot(Q_temp.T, Q_cs)
        [U, S, V] = np.linalg.svd(data_svd)
        rho = 1.25 * S[0] + 0.67 * S[1] + 0.5 * S[2]
        p_CCA[frequencyIndex] = rho
    return p_CCA


def ecca(dataTRCA, targetTemplateSet, ECCATemplate):
    p_ECCA = np.zeros(40)
    testData = dataTRCA
    for target_i in range(40):
        coefficience = np.zeros(4)
        iReference = targetTemplateSet[target_i][:, 0:dataTRCA.shape[1]]
        iTemplate = np.squeeze(ECCATemplate[target_i].mean(axis=0))
        # iTemplate = np.squeeze(ECCATemplate[target_i].mean(axis=0))

        wn1, wn2 = CCA_Matrix(testData, iReference)
        weighted_train = np.dot(wn2.T, iReference)
        weighted_test = np.dot(wn1.T, testData)
        coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
        coefficience[0] = abs(coefficienceMatrix[0, 1])

        wn, _ = CCA_Matrix(testData, iTemplate)
        weighted_train = np.dot(wn.T, iTemplate)
        weighted_test = np.dot(wn.T, testData)
        coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
        coefficience[1] = coefficienceMatrix[0, 1]

        wn, _ = CCA_Matrix(testData, iReference)
        weighted_train = np.dot(wn.T, iTemplate)
        weighted_test = np.dot(wn.T, testData)
        coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
        coefficience[2] = coefficienceMatrix[0, 1]

        wn, _ = CCA_Matrix(iTemplate, iReference)
        weighted_train = np.dot(wn.T, iTemplate)
        weighted_test = np.dot(wn.T, testData)
        coefficienceMatrix = np.corrcoef(weighted_test, weighted_train)
        coefficience[3] = coefficienceMatrix[0, 1]
        p_ECCA[target_i] = np.sum(np.sign(coefficience) * (coefficience * coefficience))
    return p_ECCA


def filterbank(eeg, fs, idx_fb):
    if idx_fb == None:
        warnings.warn('stats:filterbank:MissingInput ' \
                      + 'Missing filter index. Default value (idx_fb = 0) will be used.')
        idx_fb = 0
    elif (idx_fb < 0 or 9 < idx_fb):
        raise ValueError('stats:filterbank:InvalidInput ' \
                         + 'The number of sub-bands must be 0 <= idx_fb <= 9.')

    if (len(eeg.shape) == 2):
        num_chans = eeg.shape[0]
        num_trials = 1
    else:
        num_chans, _, num_trials = eeg.shape

    # Nyquist Frequency = Fs/2N
    Nq = fs / 2

    passband = [6, 14, 22, 30, 38, 46, 54, 62, 70, 78]
    stopband = [4, 10, 16, 24, 32, 40, 48, 56, 64, 72]
    Wp = [passband[idx_fb] / Nq, 90 / Nq]
    Ws = [stopband[idx_fb] / Nq, 100 / Nq]
    [N, Wn] = scipy.signal.cheb1ord(Wp, Ws, 3, 40)  # band pass filter StopBand=[Ws(1)~Ws(2)] PassBand=[Wp(1)~Wp(2)]
    [B, A] = scipy.signal.cheby1(N, 0.5, Wn, 'bandpass')  # Wn passband edge frequency

    y = np.zeros(eeg.shape)
    if (num_trials == 1):
        for ch_i in range(num_chans):
            # apply filter, zero phass filtering by applying a linear filter twice, once forward and once backwards.
            # to match matlab result we need to change padding length
            y[ch_i, :] = scipy.signal.filtfilt(B, A, eeg[ch_i, :], padtype='odd', padlen=3 * (max(len(B), len(A)) - 1))

    else:
        for trial_i in range(num_trials):
            for ch_i in range(num_chans):
                y[ch_i, :, trial_i] = scipy.signal.filtfilt(B, A, eeg[ch_i, :, trial_i], padtype='odd',
                                                            padlen=3 * (max(len(B), len(A)) - 1))
    return y


def CCA_Matrix(signal1, signal2):
    X = signal1
    Y = signal2
    T = signal2.shape[1]
    meanX = np.mean(X, axis=1)
    meanX = meanX.reshape((meanX.size, 1))
    meanY = np.mean(Y, axis=1)
    meanY = meanY.reshape((meanY.size, 1))
    s11 = np.dot((X - meanX), (X - meanX).T)
    s22 = np.dot((Y - meanY), (Y - meanY).T)
    s21 = np.dot((Y - meanY), (X - meanX).T)
    s12 = np.dot((X - meanX), (Y - meanY).T)
    s11 = s11 / (T - 1)
    s22 = s22 / (T - 1)
    s12 = s12 / (T - 1)
    s21 = s21 / (T - 1)
    ta = np.dot(np.dot(np.linalg.inv(s11), s12), np.dot(np.linalg.inv(s22), s21))
    tb = np.dot(np.dot(np.linalg.inv(s22), s21), np.dot(np.linalg.inv(s11), s12))
    eigvaluea, eigvectora = np.linalg.eig(ta)
    eigvalueb, eigvectorb = np.linalg.eig(tb)
    idsorta = np.argsort(-eigvaluea)
    eigvectora = eigvectora[:, idsorta[0]]
    idsortb = np.argsort(-eigvalueb)
    eigvectorb = eigvectorb[:, idsortb[0]]
    return eigvectora, eigvectorb


def trca_matrix(X):
    num_trials, num_chans, num_sampls = np.shape(X)
    S = np.zeros((num_chans, num_chans))
    for trial_i in range(num_trials):
        S = S + np.dot(np.squeeze(X[trial_i, :, :]), np.transpose(np.sum(X[trial_i + 1:num_trials + 1, :, :], axis=0)))
    S2 = S + S.T
    X1 = np.squeeze(X[0, :, :])
    for i in range(num_trials - 1):
        X1 = np.append(X1, np.squeeze(X[i + 1, :, :]), axis=1)
    X1 = X1 - np.tile(np.mean(X1, axis=1), (num_trials * num_sampls, 1)).T
    Q = np.dot(X1, X1.T)
    S2Q = np.dot(np.linalg.inv(Q), S2)
    D_raw, V_raw = np.linalg.eig(S2Q)
    idsort = np.argsort(-D_raw)
    V_raw = V_raw[:, idsort]
    return V_raw


def etrca_withmodel(template, W_trca, XTest):
    num_sampls = XTest.shape[1]
    rowTemp = np.zeros(40)
    for class_i in range(40):
        XTestTempw = np.dot(W_trca.T, XTest)
        templateTemp = template[class_i].mean(axis=0)
        templateTempw = np.dot(W_trca.T, templateTemp)
        ss = np.corrcoef(XTestTempw.reshape((1, 40 * num_sampls)),
                         templateTempw.reshape((1, 40 * num_sampls)))
        rowTemp[class_i] = ss[0, 1]
    return rowTemp


def yuzhi(x):
    ss = np.argsort(-x)
    fenzi = x[ss[0]] - x[ss[1]]
    ttt = np.exp(x)
    fenmu = x.sum() - x.size * np.log(np.sum(ttt))
    result = fenzi / fenmu
    return result