import time
import numpy as np
import matplotlib.pyplot as plt
import serial
import heapq
import psutil  # For CPU usage 

ser = serial.Serial("/dev/ttyUSB0",115200)
fs = 4000
N = 256
plt.ion()
fig, axs = plt.subplots(3)

def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            temp_str = f.readline()
            return float(temp_str) / 1000.0  # Convert millidegree to degree
    except:
        return 0.0  # fallback in case of error

while True:
    cpu_temp = get_cpu_temp()
	# --- Start timing ---
    loop_start_time = time.time()
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
        
        axs[0].axvspan(20, 250, color='blue', alpha=0.1)  # Bass
        axs[0].axvspan(250, 2000, color='green', alpha=0.1)  # Mid
        axs[0].axvspan(2000, 8000, color='red', alpha=0.1)  # Treble
        
        #axs[0].set_xscale('log')
        #axs[1].set_xscale('log')
        
        #From nyquist 
        axs[0].set_xlim((0,fs/2))
        axs[1].set_xlim((0,fs/2))
        
        
        axs[0].plot(freqs[:N//2],20*np.log10(mag+1e-6)) 
        axs[0].set_ylim((-80,80))
        axs[0].set_xlabel("frequecy [HZ]")
        axs[0].set_ylabel("Decibel [dB]")
        axs[0].grid(True)
        
        #show largest and 2nd largest data 
        
        dB_max_main = 20*np.log10(main_peak_mag+1e-6)
        dB_max_second = 20*np.log10(second_peak_mag+1e-6)
        
        axs[0].scatter(main_peak_freq,dB_max_main,color="red",s=20,label=f"1st largest = {dB_max_main:.2f} dB / {main_peak_freq:.2f} Hz")
        axs[0].text(main_peak_freq,dB_max_main,f"{dB_max_main:.2f}",color = "red",va="bottom")
        
        axs[0].scatter(second_peak_freq,dB_max_second,color="black",s=20,label=f"2nd largest = {dB_max_second:.2f} dB / {second_peak_freq:.2f} Hz")
        axs[0].text(second_peak_freq,dB_max_second,f"{dB_max_second:.2f}",color = "black",va="bottom")
        
        axs[0].legend()
        
        dB_max_y = 20*np.log10(max_y+1e-6)
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
        loop_end_time = time.time()
        fps = 1.0 / (loop_end_time - loop_start_time + 1e-6)  # Add epsilon to avoid div by 0
        # Optional: Get current CPU usage
        cpu_usage = psutil.cpu_percent(interval=0.01)
        axs[0].text(0.98, 0.02,
            f'FPS: {fps:.1f}\nCPU: {cpu_usage:.1f}%\nTemp: {cpu_temp:.1f}°C',
            transform=axs[0].transAxes,
            ha='right', va='bottom',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.5))

					  
					  
			  
        plt.pause(0.001)

        
