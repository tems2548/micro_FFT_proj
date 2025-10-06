import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import heapq

ser = serial.Serial("/dev/ttyUSB0",115200)
fs = 4000
N = 128
plt.ion()
fig, axs = plt.subplots(3)


while True:
    #FFT
    data = []
    for _ in range(N):
        line = ser.readline().decode().strip()
        if line.isdigit():
            data.append(int(line)/1000)
    if len(data) == N:
	    #FFT
        fft_vals = np.fft.fft(data)
       
        freqs = np.fft.fftfreq(N,1/fs)
        mag = np.abs(fft_vals[:N//2])*(2.0 / N)
       
       
        #find max-min
        max_fft_value = np.argmax(mag)
        #find max-min locate 
        max_x,max_y = freqs[max_fft_value],mag[max_fft_value]
        
        #find second largest 
        #skip interval 
        start_index = 10
        freqs2 = freqs[start_index:]
        mag2 = mag[start_index:]
        
        mag_copy = mag2.copy()
        #sorted
        sorted_indic = np.argsort(mag2)[::-1]
        
        #find main peak index
        main_peak_index = sorted_indic[0] + start_index
        
        #Exclude ±exclude_bins around this main peak
        exclude_bins = 15
        mag_copy[max(0, main_peak_index - exclude_bins) : min(len(mag_copy), main_peak_index + exclude_bins + 1)] = 0
        #Find 2nd peak index
        sorted_indic_fil = np.argsort(mag_copy)[::-1]
        second_peak_index = sorted_indic_fil[0] + start_index
        
        main_peak_freq, main_peak_mag = freqs[main_peak_index], mag[main_peak_index]
        second_peak_freq, second_peak_mag = freqs[second_peak_index], mag[second_peak_index]
       
        axs[0].clear()
        axs[1].clear()
        axs[2].clear()
        
        
        #From nyquist 
        axs[0].set_xlim((0,fs/2))
        axs[1].set_xlim((0,fs/2))
        
        
        axs[0].plot(freqs[:N//2],20*np.log10(mag)) 
        axs[0].set_ylim((-80,80))
        axs[0].set_xlabel("frequecy [HZ]")
        axs[0].set_ylabel("Decibel [dB]")
        axs[0].grid(True)
        
        #show largest and 2nd largest data 
        
        dB_max_main = 20*np.log10(main_peak_mag)
        dB_max_second = 20*np.log10(second_peak_mag)
        
        axs[0].scatter(main_peak_freq,dB_max_main,color="red",s=20,label=f"1st largest = {dB_max_main:.2f} dB / {main_peak_freq:.2f} Hz")
        axs[0].text(main_peak_freq,dB_max_main,f"{dB_max_main:.2f}",color = "red",va="bottom")
        
        axs[0].scatter(second_peak_freq,dB_max_second,color="black",s=20,label=f"2nd largest = {dB_max_second:.2f} dB / {second_peak_freq:.2f} Hz")
        axs[0].text(second_peak_freq,dB_max_second,f"{dB_max_second:.2f}",color = "black",va="bottom")
        
        axs[0].legend()
        
        dB_max_y = 20*np.log10(max_y)
        bbox_prop = dict(boxstyle = "square,pad = 0.3 ",fc ="w",ec="k",lw=0.72)
        arrowprops = dict(arrowstyle = "->",connectionstyle="angle,angleA=0,angleB=60")
        kw = dict(xycoords = 'data',textcoords = "axes fraction",
				  arrowprops = arrowprops,bbox = bbox_prop,ha="right",va="top") 
        axs[0].annotate(f'max :{dB_max_y:.2f}',xy = (max_x,dB_max_y)
					   ,xytext=(0.3,0.8),**kw)
        
        
        
        
        axs[1].plot(freqs[:N//2],mag)
        axs[1].set_ylim((0,1))
        axs[1].set_xlabel("frequecy [HZ]")
        axs[1].set_ylabel("Volt [V]")
        axs[1].grid(True)
        
        axs[1].scatter(main_peak_freq,main_peak_mag,color="red",s=20,label=f"1st largest = {main_peak_mag:.2f} V / {main_peak_freq:.2f} Hz")
        axs[1].text(main_peak_freq,main_peak_mag,f"{main_peak_mag:.2f}",color = "red",va="bottom")
        
        axs[1].scatter(second_peak_freq,second_peak_mag,color="black",s=20,label=f"2nd largest = {second_peak_mag:.2f} V / {second_peak_freq:.2f} Hz ")
        axs[1].text(second_peak_freq,second_peak_mag,f"{second_peak_mag:.2f}",color = "black",va="bottom")
        
        axs[1].legend()
        
        history = np.zeros((100, N//2))  # e.g., last 100 FFT frames
        history = np.roll(history, -1, axis=0)
        history[-1, :] = 20*np.log10(mag+1e-6)
        img = axs[2].imshow(history.T, aspect='auto', origin='lower',
					  extent=[0, 100, 0, fs/2], cmap='viridis',vmin = -100,vmax=0)
        plt.pause(0.001)

        
    
            
    
