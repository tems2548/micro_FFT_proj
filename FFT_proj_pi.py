import serial
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import psutil, time, os

# ----------------- Settings -----------------
PORT = "/dev/ttyACM0"          # Change to your ESP32 port
BAUD = 1000000
N = 1024
HEADER = b'\xCD\xAB'
VREF = 3.3
FS = 18860.0                   # default constant sampling frequency
HISTORY_LEN = 100
FRAME_SIZE = 2 + N*2           # HEADER + ADC

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Plot -----------------
plt.ion()
fig, axes = plt.subplots(3, 1, figsize=(10, 9), constrained_layout=True)
ax_time, ax_fft, ax_spec = axes

spec_history = deque(np.zeros(N//2), maxlen=HISTORY_LEN)
prev_time = time.time()
frame_count = 0
fps_display = 0

def get_cpu_temp():
    """Read CPU temperature (Raspberry Pi or Linux)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except:
        return np.nan

while True:
    try:
        start_frame = time.time()
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        # --- ADC to voltage ---
        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        # --- Time-domain plot ---
        ax_time.cla()
        ax_time.plot(np.arange(N)/FS, volt, color='black')
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Voltage [V]")
        ax_time.set_title("Time Domain Signal")
        ax_time.grid(True)

        # --- FFT in dB ---
        fft_vals = np.fft.rfft(volt * np.hanning(N))  # windowed FFT
        mag = np.abs(fft_vals) * 2 / N
        mag_db = 20 * np.log10(mag + 1e-12)
        freq_axis = np.fft.rfftfreq(N, 1/FS)

        ax_fft.cla()
        ax_fft.plot(freq_axis, mag_db, color='purple')
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude [dB]")
        ax_fft.set_title("FFT Spectrum (in dB)")
        ax_fft.grid(True)

        # --- Find 2nd & 3rd largest peaks ---
        sorted_indices = np.argsort(mag_db)[::-1]
        exclude_bins = 5
        main_idx = sorted_indices[0]
        mag_copy = mag_db.copy()
        mag_copy[max(0, main_idx-exclude_bins):main_idx+exclude_bins+1] = -np.inf
        second_idx = np.argmax(mag_copy)
        mag_copy[max(0, second_idx-exclude_bins):second_idx+exclude_bins+1] = -np.inf
        third_idx = np.argmax(mag_copy)

        # --- Mark peaks on FFT ---
        ax_fft.scatter(freq_axis[second_idx], mag_db[second_idx], color='red', s=60, zorder=3)
        ax_fft.text(freq_axis[second_idx], mag_db[second_idx],
                    f"2nd: {freq_axis[second_idx]:.1f} Hz\n{mag_db[second_idx]:.1f} dB",
                    color='red', fontsize=9)
        ax_fft.scatter(freq_axis[third_idx], mag_db[third_idx], color='blue', s=60, zorder=3)
        ax_fft.text(freq_axis[third_idx], mag_db[third_idx],
                    f"3rd: {freq_axis[third_idx]:.1f} Hz\n{mag_db[third_idx]:.1f} dB",
                    color='blue', fontsize=9)

        # --- Spectrogram ---
        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T

        ax_spec.cla()
        im = ax_spec.imshow(spec_array, aspect='auto', origin='lower',
                            extent=[0, HISTORY_LEN, 0, FS/2],
                            cmap='inferno', vmin=-100, vmax=0)
        ax_spec.set_xlabel("Frame Index")
        ax_spec.set_ylabel("Frequency [Hz]")
        ax_spec.set_title("Spectrogram (dB)")
        ax_spec.grid(False)

        # --- Highlight current 2nd & 3rd peaks ---
        ax_spec.axhline(freq_axis[second_idx], color='red', lw=1.5, linestyle='--')
        ax_spec.axhline(freq_axis[third_idx], color='blue', lw=1.5, linestyle='--')

        # --- FPS + CPU temperature ---
        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps_display = frame_count / (time.time() - prev_time)
            cpu_temp = get_cpu_temp()
            prev_time = time.time()
            frame_count = 0
            print(f"FPS: {fps_display:.1f}, CPU Temp: {cpu_temp:.1f}°C")

        # Show FPS & CPU temp text overlay
        fig.suptitle(f"Real-Time FFT | FPS: {fps_display:.1f} | CPU Temp: {get_cpu_temp():.1f}°C",
                     fontsize=12, color='darkgreen')

        plt.pause(0.001)

    except Exception as e:
        print("Frame skipped:", e)
        continue
