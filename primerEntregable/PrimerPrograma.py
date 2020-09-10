
import numpy as np
import matplotlib.pyplot as plt

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
    plt.show()

    power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
    power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)
    plt.clf()

    start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
    end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
    #print(start_freq, end_freq)
    start_freq2 = next(y for y, val in enumerate(freq) if val >= 4.0)
    end_freq2 = next(y for y, val in enumerate(freq) if val >= 60.0)
    #print(start_freq2, end_freq2)

    #print("La frecuencia es", freq)
    #print("El poder es", power)
    start_index = np.where(freq >= 4.0)[0][0]
    end_index = np.where(freq >= 60.0)[0][0]



    plt.plot(freq[start_index:end_index],
             power[start_index:end_index], label='Canal 1')
    plt.plot(freq2[start_index:end_index],
             power2[start_index:end_index], color='red', label='Canal 2')
    plt.xlabel('Hz')
    plt.ylabel('Power')
    plt.legend()
    plt.show()



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

            print("You are currently at ", currentMark,"initial samp ", ini_samp, "end_samp ", end_samp)
            plt.plot(t, x, label='Canal 1')
            plt.plot(t, y, color='red', label='Canal 2')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('micro V')
            plt.legend()
            plt.clf()

            power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
            power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)

            plt.clf()

            #print("Power ",power, " freq ",freq)

            # start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
            # end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
            #print(start_freq, end_freq)

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
        channPowAverage1[mark][hz] = sum(channPow1[mark][hz])/len(channPow1[mark][hz])
        channPowAverage2[mark][hz] = sum(channPow2[mark][hz])/len(channPow2[mark][hz])


for mark in channPowAverage1:
    plt.plot(channPowAverage1[mark].keys(), channPowAverage1[mark].values(), label = mark)


plt.title("Canal 1")
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()
plt.show()


for mark in channPowAverage2:
    plt.plot(channPowAverage2[mark].keys(), channPowAverage2[mark].values(), label = mark)
    
plt.title("Canal 2")
plt.xlabel('Hz')
plt.ylabel('Power')
plt.legend()
plt.show()


