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
FS = 18860.0                   # Sampling frequency (Hz)
HISTORY_LEN = 100
TRACK_LEN = 200                # Frequency tracking length
FRAME_SIZE = 2 + N*2           # HEADER + ADC data

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Plot Setup -----------------
plt.ion()
fig, axes = plt.subplots(4, 1, figsize=(10, 11), constrained_layout=True)
ax_time, ax_fft, ax_spec, ax_track = axes

spec_history = deque(np.zeros(N//2), maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(np.linspace(-TRACK_LEN, 0, TRACK_LEN), maxlen=TRACK_LEN)

prev_time = time.time()
frame_count = 0
fps_display = 0

# ----------------- Helper -----------------
def get_cpu_temp():
    """Read CPU temperature (Linux or Raspberry Pi)."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except:
        return np.nan

def smooth(data, w=5):
    """Simple moving average smoothing."""
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='same')

# ----------------- Main Loop -----------------
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        # --- Convert ADC to voltage ---
        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        # --- Time-domain plot ---
        ax_time.cla()
        ax_time.plot(np.arange(N)/FS, volt, color='black')
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Voltage [V]")
        ax_time.set_title("Time Domain Signal")
        ax_time.grid(True)

        # --- FFT ---
        fft_vals = np.fft.rfft(volt * np.hanning(N))
        mag = np.abs(fft_vals) * 2 / N
        mag_db = 20 * np.log10(mag + 1e-12)
        freq_axis = np.fft.rfftfreq(N, 1/FS)

        # --- Find main frequency & harmonics ---
        main_idx = np.argmax(mag_db)
        main_freq = freq_axis[main_idx]
        main_amp = mag[main_idx]

        harmonic_freqs = []
        harmonic_mags = []
        for h in [2, 3]:
            target = h * main_freq
            if target < FS/2:
                idx = np.argmin(np.abs(freq_axis - target))
                harmonic_freqs.append(freq_axis[idx])
                harmonic_mags.append(mag[idx])

        # --- Compute RMS and THD ---
        rms = np.sqrt(np.mean(volt**2))
        if harmonic_mags:
            thd = np.sqrt(np.sum(np.array(harmonic_mags)**2)) / main_amp
        else:
            thd = 0.0

        # --- FFT plot ---
        ax_fft.cla()
        ax_fft.plot(freq_axis, mag_db, color='purple')
        ax_fft.scatter(main_freq, 20*np.log10(main_amp), color='green', s=60)
        ax_fft.text(main_freq, 20*np.log10(main_amp)+3,
                    f"Main: {main_freq:.1f} Hz", color='green', fontsize=9)

        for i, f in enumerate(harmonic_freqs):
            ax_fft.scatter(f, 20*np.log10(harmonic_mags[i]), color='orange', s=60)
            ax_fft.text(f, 20*np.log10(harmonic_mags[i])+3,
                        f"{i+2}nd: {f:.1f} Hz", color='orange', fontsize=8)

        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude [dB]")
        ax_fft.set_title("FFT Spectrum with Harmonics")
        ax_fft.grid(True)

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
        ax_spec.axhline(main_freq, color='green', lw=1.5, linestyle='--')

        # --- Frequency tracking ---
        freq_history.append(main_freq)
        time_history.append(time_history[-1] + 1)

        ax_track.cla()
        ax_track.plot(list(time_history), smooth(freq_history, 5),
                      color='teal', linewidth=1.8)
        ax_track.set_title("Main Frequency Tracking")
        ax_track.set_xlabel("Frame Index")
        ax_track.set_ylabel("Frequency [Hz]")
        ax_track.grid(True)
        ax_track.set_xlim(time_history[0], time_history[-1])
        ax_track.set_ylim(0, FS/2)

        # --- FPS & CPU Temp ---
        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps_display = frame_count / (time.time() - prev_time)
            cpu_temp = get_cpu_temp()
            prev_time = time.time()
            frame_count = 0
            print(f"FPS: {fps_display:.1f}, CPU Temp: {cpu_temp:.1f}°C")

        # --- Overlay info ---
        fig.suptitle(
            f"Real-Time FFT | FPS: {fps_display:.1f} | CPU: {get_cpu_temp():.1f}°C | "
            f"Main: {main_freq:.1f} Hz | RMS: {rms:.3f} V | THD: {thd*100:.2f}%",
            fontsize=11, color='darkgreen'
        )

        plt.pause(0.001)

    except Exception as e:
        print("Frame skipped:", e)
        continue
