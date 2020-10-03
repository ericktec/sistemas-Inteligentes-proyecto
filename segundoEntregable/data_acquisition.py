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
win_size = 256
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
            
        print("time samp ", time_samp)
        elapsed_time = time.time() - start_time
        if (elapsed_time > 0.1):
            start_time = time.time()
            print ("Muestras: ", ns)
            print ("Cuenta: ", samp_count)
            print ("Última lectura: ", [row[samp_count-1] for row in emg_data])
            ventana_actual = emg_data
            
            if (contador > 256):
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
                    
                print("El array t es", t)
                print("Tamaño del array ", len(t))
                x = chann1[-256:]
                y = chann2[-256:]
                fig, axs = plt.subplots(2)
                axs[0].plot(t, x, label='Canal 1')
                axs[1].plot(t, y, color='red', label='Canal 2')
                plt.xlabel('Tiempo (s)')
                plt.ylabel('micro V')
                plt.legend()
                plt.show()
                contador = 0
                

                power, freq = plt.psd(x, NFFT=win_size, Fs=samp_rate)
                power2, freq2 = plt.psd(y, NFFT=win_size, Fs=samp_rate)
                plt.clf()

                start_freq = next(x for x, val in enumerate(freq) if val >= 4.0)
                end_freq = next(x for x, val in enumerate(freq) if val >= 60.0)
                
                start_freq2 = next(y for y, val in enumerate(freq) if val >= 4.0)
                end_freq2 = next(y for y, val in enumerate(freq) if val >= 60.0)
                
                start_index = np.where(freq >= 4.0)[0][0]
                end_index = np.where(freq >= 60.0)[0][0]


                plt.plot(freq[start_index:end_index], power[start_index:end_index], label='Canal 1')
                plt.plot(freq[start_index:end_index], power2[start_index:end_index], color='red', label='Canal 2')
                plt.xlabel('Hz')
                plt.ylabel('Power')
                plt.legend()
                plt.show()

            
    except socket.timeout:
        pass  
    
    

