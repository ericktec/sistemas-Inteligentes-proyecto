
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


# Read data file
data = np.loadtxt("Izquierda, derecha, cerrado.txt")
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
                channPow1[condition_id] = {}
                channPow2[condition_id] = {}
                channPowAverage1[condition_id] = {}
                channPowAverage2[condition_id] = {}
            training_samples[int(condition_id)].append([iniSamp, i])

# print('Rango de muestras con datos de entrenamiento:', training_samples)
print(channPowAverage1)

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
    # print(start_freq2, end_freq2)

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


for mark in training_samples:
    for hz in range(4, 61):
        channPow1[mark][hz] = []
        channPow2[mark][hz] = []
        channPowAverage1[mark][hz] = 0
        channPowAverage2[mark][hz] = 0


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

            for hz in range(start_index, end_index+1):
                channPow1[currentMark][hz].append(power[hz])
                channPow2[currentMark][hz].append(power2[hz])

            # plt.plot(freq[start_index:end_index], power[start_index:end_index], label='Canal 1')
            # plt.plot(freq[start_index:end_index], power2[start_index:end_index], color='red', label='Canal 2')
            # plt.xlabel('Hz')
            # plt.ylabel('Power')
            # plt.legend()
            # plt.clf()
            ini_samp = end_samp
            end_samp = ini_samp + win_size


for mark in training_samples:
    for hz in range(4, 61):
        channPowAverage1[mark][hz] = sum(
            channPow1[mark][hz])/len(channPow1[mark][hz])
        channPowAverage2[mark][hz] = sum(
            channPow2[mark][hz])/len(channPow2[mark][hz])


for mark in channPowAverage1:
    plt.plot(channPowAverage1[mark].keys(),
             channPowAverage1[mark].values(), label=mark)


plt.title("Canal 1")
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()
plt.clf()


for mark in channPowAverage2:
    plt.plot(channPowAverage2[mark].keys(),
             channPowAverage2[mark].values(), label=mark)

plt.title("Canal 2")
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()
plt.clf()


y = []

x = []

for mark in training_samples:
    for i in range(len(channPow1[mark][4])):
        y.append(mark)
        temp = []
        for hz in range(4, 61):
            temp.append(channPow1[mark][hz][i])
        x.append(temp)


y = np.array(y)
x = np.array(x)


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

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test))
    accp += (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test)


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

    print("acc = ", (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test))
    accp += (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test)


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

    print("acc = ", (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test))
    accp += (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test)


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

    print("acc = ", (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test))
    accp += (cm[0, 0]+cm[1, 1]+cm[2, 2])/len(y_test)


if(accp/5 > bestAverage):
    bestAverage = accp/5
    bestPrediction = "DTC"

print("Average accuracy is ", accp/5)

################################################################
##  RED NEURONAL ##



n_features = 56




print("x ",x)

# Define MLP model
clf = Sequential()
clf.add(Dense(8, input_dim=n_features, activation='relu'))
clf.add(Dense(8, activation='relu'))
clf.add(Dense(3, activation='softmax')) # for 2-class problems, use clf.add(Dense(1, activation='sigmoid'))

# Compile model
clf.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

output_y = np_utils.to_categorical(y) 

# Fit model
clf.fit(x, output_y, epochs=150, batch_size=5)

# Predict class of a new observation
prob = clf.predict( [[1.,2.,3.,4.]] )
print("Probablities", prob)
print("Predicted class", np.argmax(prob, axis=-1))

# Evaluate model
kf = KFold(n_splits=5, shuffle = True)

acc = 0
recall = np.array([0., 0., 0.])
for train_index, test_index in kf.split(x):

    # Training phase
    x_train = x[train_index, :]
    y_train = y[train_index]
    y_train = np_utils.to_categorical(y_train)  # Only required in multiclass problems

    clf = Sequential()
    clf.add(Dense(8, input_dim=n_features, activation='relu'))
    clf.add(Dense(8, activation='relu'))
    clf.add(Dense(3, activation='softmax'))
    clf.compile(loss='categorical_crossentropy', optimizer='adam') # For 2-class problems, use clf.compile(loss='binary_crossentropy', optimizer='adam')
    clf.fit(x_train, y_train, epochs=150, batch_size=5, verbose=0)    

    # Test phase
    x_test = x[test_index, :]
    y_test = y[test_index]
    y_pred = np.argmax(clf.predict(x_test), axis=-1)  # For 2-class problems, use (clf.predict(x_test) > 0.5).astype("int32")

    cm = confusion_matrix(y_test, y_pred)

    acc += (cm[0,0]+cm[1,1]+cm[2,2])/len(y_test)    

    recall[0] += cm[0,0]/(cm[0,0] + cm[0,1] + cm[0,2])
    recall[1] += cm[1,1]/(cm[1,0] + cm[1,1] + cm[1,2])
    recall[2] += cm[2,2]/(cm[2,0] + cm[2,1] + cm[2,2])

print("Red neuronal")

accp = acc/5
print('ACC = ', accp)

recall = recall/5
print('RECALL = ', recall)

#################################################################

if(accp > bestAverage):
    bestAverage = accp
    bestPrediction = "Red neuronal"


print("Average accuracy is ", accp)


print("The best classifier was ", bestPrediction)
