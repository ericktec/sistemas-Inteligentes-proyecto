#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt


from sklearn import svm
from sklearn.model_selection import KFold
from sklearn.metrics import confusion_matrix


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
timestamp = data[:, 0]

# Data channels
chann1 = data[:, 1]
chann2 = data[:, 3]

# Mark data
mark = data[:, 6]



frequenciesLabels = []

training_samples = {}
for i in range(0, samps):
    if mark[i] > 0:
        print("Marca", mark[i], 'Muestra', i, 'Tiempo', timestamp[i])

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
    t = timestamp[ini_samp: end_samp]
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
            t = timestamp[ini_samp: end_samp]
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

            ini_samp = end_samp
            end_samp = ini_samp + win_size


print("chanpow1", channPow1)
print("chanpow2", channPow2)


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

    y_pred = clf.predict(x_test)
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    print("acc = ", getAccuracy(cm, len(y_test)))
    accp += getAccuracy(cm, len(y_test))


bestAverage = accp/5
print("Average accuracy is ", accp/5)



# Data configuration
n_channels = n_channels-2
print("chanels", n_channels)
emg_data = [[] for i in range(n_channels)]
samp_count = 0
contador  = 0
time_samp = []

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)
print("prediction", clf.predict(x_test))

# Data acquisition
start_time = time.time()
while True:
    try:
        data, addr = sock.recvfrom(1024*1024)                        
        
        values = np.frombuffer(data)       
        ns = int(len(values)/n_channels)
        samp_count+=ns
        contador+=ns        
        time_samp.append(samp_count/samp_rate)
        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])
            
        # print("time samp ", time_samp)
        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            #print ("Muestras: ", ns)
            #print ("Cuenta: ", samp_count)
            #print ("Última lectura: ", [row[samp_count-1] for row in emg_data])  
            ventana_actual = emg_data
            if (contador > win_size):
                
                chann1 = emg_data[0]
                chann2 = emg_data[2]

                # print("Channel 1: ", chann1)
                # print("Channel 2: ", chann2)
      
                t = []
                aumento = 1/samp_rate
                for i in range(0,samp_rate):
                    if(len(t) == 0):
                        t.append(time_samp[-1])
                    else:
                        t.append(t[-1]-aumento)
                    
                
                # print("El array t es", t)
                # print("Tamaño del array ", len(t))
                x = chann1[-win_size:]
                y = chann2[-win_size:]
                #print("X ", x)
                #print("Y", y)
                contador = 0
                

                power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
                power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)
                plt.clf()

                start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
                end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
    
                
                start_index = np.where(freq >= 4.0)[0][0]
                end_index = np.where(freq >= 60.0)[0][0]

                temp1 = []
                temp2 = []
                for hz in range(start_index, end_index+1):
                    temp1.append(power[hz])
                    temp2.append(power2[hz])

                #print("temp1", temp1)
                #print("temp2", temp2)
                print("prediction", clf.predict([temp1+temp2]))

            
    except socket.timeout:
        pass  
    
    

