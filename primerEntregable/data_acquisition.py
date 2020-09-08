#------------------------------------------------------------------------------------------------------------------
#   Sample program for data acquisition and recording.
#------------------------------------------------------------------------------------------------------------------
import time
import socket
import numpy as np
import matplotlib.pyplot as plt

# Data configuration
n_channels = 5
samp_rate = 256
emg_data = [[] for i in range(n_channels)]
samp_count = 0
time_samp = []

# Socket configuration
UDP_IP = '127.0.0.1'
UDP_PORT = 8000
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.settimeout(0.01)

# Data acquisition
start_time = time.time()

while True:
    try:
        data, addr = sock.recvfrom(1024*1024)                        
            
        values = np.frombuffer(data)       
        ns = int(len(values)/n_channels)
        samp_count+=ns        
        time_samp.append(samp_count/samp_rate)
        for i in range(ns):
            for j in range(n_channels):
                emg_data[j].append(values[n_channels*i + j])
            
        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            print ("Muestras: ", ns)
            print ("Cuenta: ", samp_count)
            print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
            print("")
            ventana_actual = emg_data
            print("EMG_DATA ", ventana_actual)
            chann1 = ventana_actual[:, 0]
            chann2 = ventana_actual[:, 2]

            # print("Channel 1: ", chann1)
            # print("Channel 2: ", chann2)

            ini_samp = samp_count-samp_rate
            end_samp = samp_count
            

            x = chann1[ini_samp: end_samp]
            t = time_samp[ini_samp: end_samp]
            y = chann2[ini_samp: end_samp]
            
            fig, axs = plt.subplots(2)
            axs[0].plot(t, x, label='Canal 1')
            axs[1].plot(t, y, color='red', label='Canal 2')
            plt.xlabel('Tiempo (s)')
            plt.ylabel('micro V')
            plt.legend()
            plt.show()
            

            # power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
            # power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)
            # plt.clf()

            # start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
            # end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
            
            # start_freq2 = next(y for y, val in enumerate(freq) if val >= 4.0)
            # end_freq2 = next(y for y, val in enumerate(freq) if val >= 60.0)
            
            # start_index = np.where(freq >= 4.0)[0][0]
            # end_index = np.where(freq >= 60.0)[0][0]


            # plt.plot(freq[start_index:end_index], power[start_index:end_index], label='Canal 1')
            # plt.plot(freq[start_index:end_index], power2[start_index:end_index], color='red', label='Canal 2')
            # plt.xlabel('Hz')
            # plt.ylabel('Power')
            # plt.legend()
            # plt.show()

            
    except socket.timeout:
        pass  
    
    

