
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils


def getAccuracy(cm, len_y):
    accp = 0
    for i in range(0, cm.shape[0]):
        accp += cm[i, i]
    return accp/len_y


# Read data file
data = np.loadtxt("Abierto - Cerrado - Normal 1.txt")
samp_rate = 256
samps = data.shape[0]
n_channels = data.shape[1]
win_size = 256

channPow1 = {

}
channPow2 = {

}
channPowAverage1 = {

}
channPowAverage2 = {

}

# print('Número de muestras: ', data.shape[0])
# print('Número de canales: ', data.shape[1])
# print('Duración del registro: ', samps / samp_rate, 'segundos')
# print(data)

# Time channel
time = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

# Mark data
mark = data[:, 6]



frequenciesLabels = []

training_samples = {}
for i in range(0, samps):
    if mark[i] > 0:
        print("Marca", mark[i], 'Muestra', i, 'Tiempo', time[i])

        if (mark[i] > 100) and (mark[i] < 200):
            iniSamp = i
            condition_id = mark[i]
        elif mark[i] == 200:
            if not condition_id in training_samples.keys():
                training_samples[condition_id] = []
                channPow1[condition_id] = []
                channPow2[condition_id] = []
                channPowAverage1[condition_id] = []
                channPowAverage2[condition_id] = []
            training_samples[int(condition_id)].append([iniSamp, i])

# print('Rango de muestras con datos de entrenamiento:', training_samples)


# Plot data
for currentMark in training_samples:
    print("You are seeing the mark ", currentMark)
    ini_samp = training_samples[currentMark][2][0]
    end_samp = training_samples[currentMark][2][1]

    x = chann1[ini_samp: end_samp]
    t = time[ini_samp: end_samp]
    y = chann2[ini_samp: end_samp]

    fig, axs = plt.subplots(2)
    axs[0].plot(t, x, label='Canal 1')
    axs[1].plot(t, y, color='red', label='Canal 2')
    plt.xlabel('Tiempo (s)')
    plt.ylabel('micro V')
    plt.legend()
    plt.clf()

    power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
    power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)
    plt.clf()


    start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
    end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
    # print(start_freq, end_freq)
    start_freq2 = next(y for y, val in enumerate(freq) if val >= 4.0)
    end_freq2 = next(y for y, val in enumerate(freq) if val >= 60.0)
    print(start_freq2, end_freq2)

    # print("La frecuencia es", freq)
    # print("El poder es", power)
    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]

    plt.plot(freq[start_index:end_index],
             power[start_index:end_index], label='Canal 1')
    plt.plot(freq2[start_index:end_index],
             power2[start_index:end_index], color='red', label='Canal 2')
    plt.xlabel('Hz')
    plt.ylabel('Power')
    plt.legend()
    plt.clf()

frequenciesLabels = freq[start_index:end_index] 




for currentMark in training_samples:

    for i in range(0, len(training_samples[currentMark])):
        ini_samp = training_samples[currentMark][i][0]
        end_samp = ini_samp + win_size
        while(end_samp < training_samples[currentMark][i][1]):

            # Power Spectral Density (PSD) (1 second of training data)

            x = chann1[ini_samp: end_samp]
            t = time[ini_samp: end_samp]
            y = chann2[ini_samp: end_samp]

            print("You are currently at ", currentMark,
                  "initial samp ", ini_samp, "end_samp ", end_samp)
            plt.plot(t, x, label='Canal 1')
            plt.plot(t, y, color='red', label='Canal 2')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('micro V')
            plt.legend()
            plt.clf()

            power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
            power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)

            plt.clf()

            # print("Power ",power, " freq ",freq)

            # start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
            # end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
            # print(start_freq, end_freq)

            start_index = np.where(freq >= 4.0)[0][0]
            end_index = np.where(freq >= 60.0)[0][0]

            # 128: 2 - 30
            # 256: 4 - 60
            # 512: 8 - 120
            temp1 = []
            temp2 = []
            for hz in range(start_index, end_index+1):
                temp1.append(power[hz])
                temp2.append(power2[hz])

            channPow1[currentMark].append(temp1)
            channPow2[currentMark].append(temp2)

            # plt.plot(freq[start_index:end_index], power[start_index:end_index], label='Canal 1')
            # plt.plot(freq[start_index:end_index], power2[start_index:end_index], color='red', label='Canal 2')
            # plt.xlabel('Hz')
            # plt.ylabel('Power')
            # plt.legend()
            # plt.clf()
            ini_samp = end_samp
            end_samp = ini_samp + win_size


print("chanpow1", channPow1)
print("chanpow2", channPow2)


# for mark in training_samples:
#     averageTemp = np.array(channPow1[mark])
#     for i in range(0, averageTemp.size):
#         channPowAverage1[mark].append(sum(averageTemp[:, i])/len(averageTemp[:, i]))


# for mark in training_samples:
#     averageTemp = np.array(channPow2[mark])
#     for i in range(0, averageTemp.size):
#         channPowAverage2[mark].append(sum(averageTemp[:, i])/len(averageTemp[:, i]))


# for mark in channPowAverage1:
#     plt.plot(frequenciesLabels,
#              channPowAverage1[mark], label=mark)


# plt.title("Canal 1")
# plt.xlabel('Hz')
# plt.ylabel('Power')
# plt.legend()
# plt.clf()


# for mark in channPowAverage2:
#     plt.plot(frequenciesLabels,
#              channPowAverage2[mark], label=mark)

# plt.title("Canal 2")
# plt.xlabel('Hz')
# plt.ylabel('Power')
# plt.legend()
# plt.clf()


y = []

x = []


for mark in training_samples:
    for i in range(len(channPow1[mark])):
        x.append(channPow1[mark][i]+channPow2[mark][i])
        y.append(mark)


y = np.array(y)
x = np.array(x)

print("x",x)
print("y", y)

bestAverage = 0
bestPrediction = ""


print("==============================LINEAL")

kf = KFold(n_splits=5, shuffle=True)

clf = svm.SVC(kernel='linear')

accp = 0


for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    clf.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    print("x_test", x_test)
    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", getAccuracy(cm, len(y_test)))
    accp += getAccuracy(cm, len(y_test))


bestAverage = accp/5
bestPrediction = "Linear"
print("Average accuracy is ", accp/5)


print("==============================RBF")


rbf = svm.SVC(kernel='rbf')

accp = 0


for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    rbf.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = rbf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", getAccuracy(cm, len(y_test)))
    accp += getAccuracy(cm, len(y_test))


if(accp/5 > bestAverage):
    bestAverage = accp/5
    bestPrediction = "RBF"


print("Average accuracy is ", accp/5)


print("==============================NEIGH")

neigh = KNeighborsClassifier(n_neighbors=3)

accp = 0

for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    neigh.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = neigh.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", getAccuracy(cm, len(y_test)))
    accp += getAccuracy(cm, len(y_test))


if(accp/5 > bestAverage):
    bestAverage = accp/5
    bestPrediction = "NEIGH"

print("Average accuracy is ", accp/5)


print("==============================DTC")
dtc = DecisionTreeClassifier(random_state=0)

accp = 0

for train_index, test_index in kf.split(x):
    x_train = x[train_index, :]
    y_train = y[train_index]
    dtc.fit(x_train, y_train)

    x_test = x[test_index, :]
    y_test = y[test_index]

    y_pred = dtc.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", getAccuracy(cm, len(y_test)))
    accp += getAccuracy(cm, len(y_test))


if(accp/5 > bestAverage):
    bestAverage = accp/5
    bestPrediction = "DTC"

print("Average accuracy is ", accp/5)

################################################################
##  RED NEURONAL MULTICAPA ##

print("==============================RED NEURONAL MULTICAPA")

n_features = x.shape[1]

accp = 0

# Evaluate model
kf = KFold(n_splits=5, shuffle=True)


y = y-101
print("y ", y.shape)
print("x ", x.shape)
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    # Only required in multiclass problems
    

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))  # Una sola capa
    

    if(len(training_samples) < 3):
        clf.add(Dense(1, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        y_train = np_utils.to_categorical(y_train)
        clf.add(Dense(3, activation='softmax'))
        clf.compile(loss='categorical_crossentropy',
                    optimizer='adam') 

    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]

    if(len(training_samples) < 3):
        y_pred = (clf.predict(x_test) > 0.5).astype("int32")
    else:
        y_pred = np.argmax(clf.predict(x_test), axis=-1)
    
    cm = confusion_matrix(y_test, y_pred)

    accp += getAccuracy(cm, len(y_test))


print("Red neuronal multicapa")

accp = accp/5


print("Average accuracy is ", accp)

if(accp > bestAverage):
    bestAverage = accp
    bestPrediction = "Red neuronal multicapa"

################################################################

##  RED NEURONAL UNICAPA ##
print("==============================RED NEURONAL UNICAPA")

n_features = x.shape[1]

accp = 0

# Evaluate model
kf = KFold(n_splits=5, shuffle=True)

for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    if(len(training_samples) < 3):
        clf.add(Dense(1, activation='sigmoid'))
        clf.compile(loss='binary_crossentropy', optimizer='adam')
    else:
        y_train = np_utils.to_categorical(y_train)
        clf.add(Dense(3, activation='softmax'))
        clf.compile(loss='categorical_crossentropy',
                    optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    if(len(training_samples) < 3):
            y_pred = (clf.predict(x_test) > 0.5).astype("int32")
    else:
        y_pred = np.argmax(clf.predict(x_test), axis=-1)

    cm = confusion_matrix(y_test, y_pred)

    accp += getAccuracy(cm, len(y_test))


accp = accp/5

print("Average accuracy is ", accp)
################################################################

if(accp > bestAverage):
    bestAverage = accp
    bestPrediction = "Red neuronal"


print("The best classifier was ", bestPrediction)
