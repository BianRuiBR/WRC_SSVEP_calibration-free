from scipy import signal
from HUST_BCI import CCAClass_HUST
import numpy as np
from scipy.io import loadmat, savemat

from method_algo import preprocess_TRCA, preprocess, methmean

frequencySet = [8.0, 8.2, 8.4, 8.6, 8.8, 9.0, 9.2, 9.4, 9.6, 9.8,
                10.0, 10.2, 10.4, 10.6, 10.8, 11.0, 11.2, 11.4, 11.6, 11.8,
                12.0, 12.2, 12.4, 12.6, 12.8, 13.0, 13.2, 13.4, 13.6, 13.8,
                14.0, 14.2, 14.4, 14.6, 14.8, 15.0, 15.2, 15.4, 15.6, 15.8]
frequencySet2 = [8., 9., 10., 11., 12., 13., 14., 15.,
                 8.2, 9.2, 10.2, 11.2, 12.2, 13.2, 14.2, 15.2,
                 8.4, 9.4, 10.4, 11.4, 12.4, 13.4, 14.4, 15.4,
                 8.6, 9.6, 10.6, 11.6, 12.6, 13.6, 14.6, 15.6,
                 8.8, 9.8, 10.8, 11.82, 12.8, 13.8, 14.8, 15.8]
Y = np.zeros(240)
tmp = np.int0(np.linspace(1, 40, 40))
for i in range(240):
    t1 = frequencySet2[np.mod(i, 40)]
    for j in range(40):
        if t1 == frequencySet[j]:
            Y[i] = np.int0(j + 1)
            break
Y = np.int0(Y)
srate = 250
targetTemplateSet = []
multiplicateTime = 4
# 采样点
t = np.linspace(0, (1000 - 1) / srate, int(1000), endpoint=True)
t = t.reshape(1, len(t))
# 对于每个频率
for freIndex in range(0, len(frequencySet)):
    frequency = frequencySet[freIndex]
    testFre = np.linspace(frequency, frequency * multiplicateTime, int(multiplicateTime), endpoint=True)
    testFre = testFre.reshape(1, len(testFre))
    numMatrix = 2 * np.pi * np.dot(testFre.T, t)
    cosSet = np.cos(numMatrix)
    sinSet = np.sin(numMatrix)
    csSet = np.append(cosSet, sinSet, axis=0)
    targetTemplateSet.append(csSet)
# 初始化算法
method_HUST = CCAClass_HUST()
calTime = np.array([0.6, 0.72, 0.84, 0.96, 1.08])
datalengthall = np.int0(calTime * 250 - 35)
yuall = np.array([-0.003, -0.0025, -0.002, -0.002, -0.002, -0.003, -0.002, -0.002])
TRCAID = np.int0(calTime * 250 - 35)
TRCAWeight = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
ECCAWeight = [1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5]
andOrChoose = np.array([2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
blockTrain = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
subjectCalTime = 3
# self.method = TRCAClass()TRCAWeight

norchh = np.load("Model/HUST/norch.npz")
filterB = np.squeeze(norchh["notch_b"])
filterA = np.squeeze(norchh["notch_a"])

ACC_all = np.zeros((35, 6, 40))
caltimeall = np.zeros((35, 6, 40))

for i in range(1):
    method_HUST.initial(targetTemplateSet, TRCAID, TRCAWeight, ECCAWeight, andOrChoose,
                        subjectCalTime, blockTrain, yuall)
    data = loadmat("benchmark/S"+repr(i+1)+".mat")
    X = data["X"]
    X = X[:, [3, 2, 4, 1, 5, 7, 6, 8], :]
    isNew = np.zeros((1, 40))
    nowCalIime = 0
    for block_i in range(6):
        print(i, ":", block_i)
        for trial_i in range(40):
            XTrial = X[block_i*40+trial_i, :, :]
            Ytrial = Y[block_i*40+trial_i]
            if block_i == 0:
                usedData = XTrial[:, 125+35:125+35+500]
                usedDataTRCA = preprocess_TRCA(filterB, filterA, usedData)
                usedDataCCA = preprocess(filterB, filterA, usedData)
                usedDataTRCA = usedDataTRCA - methmean(usedDataTRCA, 9)
                usedDataCCA = usedDataCCA - methmean(usedDataCCA, 9)
                resultType, isNew = method_HUST.recognize(usedDataCCA, usedDataTRCA, 1, nowCalIime, block_i, isNew)
                ACC_all[i, block_i, trial_i] = int(resultType == Y[block_i*60+trial_i])
                caltimeall[i, block_i, trial_i] = 2 + 0.14
            else:
                for cal_i in range(4):
                    usedData = XTrial[:, 125 + 35:125 + 35 + datalengthall[cal_i]]
                    usedDataTRCA = preprocess_TRCA(filterB, filterA, usedData)
                    usedDataCCA = preprocess(filterB, filterA, usedData)
                    usedDataTRCA = usedDataTRCA - methmean(usedDataTRCA, 9)
                    usedDataCCA = usedDataCCA - methmean(usedDataCCA, 9)
                    resultType, isNew = method_HUST.recognize(usedDataCCA, usedDataTRCA, 1, cal_i, block_i, isNew)
                    if resultType is not None:
                        ACC_all[i, block_i, trial_i] = int(resultType == Y[block_i * 40 + trial_i])
                        caltimeall[i, block_i, trial_i] = calTime[cal_i]
                        break

savemat("HustRe3.mat", {"ACC_all": ACC_all, "caltimeall":caltimeall})




