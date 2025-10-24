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
FS = 18860.0
HISTORY_LEN = 100
TRACK_LEN = 200
FRAME_SIZE = 2 + N*2
MIN_FREQ = 5.0

# ----------------- Serial -----------------
ser = serial.Serial(PORT, BAUD, timeout=0.05)

# ----------------- Plot Setup -----------------
plt.ion()
fig, axes = plt.subplots(5, 1, figsize=(10, 13), constrained_layout=True)
ax_time, ax_fft, ax_spec, ax_track, ax_text = axes

spec_history = deque(np.zeros(N//2), maxlen=HISTORY_LEN)
freq_history = deque([0]*TRACK_LEN, maxlen=TRACK_LEN)
time_history = deque(np.linspace(-TRACK_LEN, 0, TRACK_LEN), maxlen=TRACK_LEN)

prev_time = time.time()
frame_count = 0
fps_display = 0

# ----------------- Helper Functions -----------------
def get_cpu_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return float(f.read()) / 1000.0
    except:
        return np.nan

def smooth(data, w=5):
    if len(data) < w:
        return data
    return np.convolve(data, np.ones(w)/w, mode='same')

def find_main_and_secondary_freq(volt):
    volt = volt - np.mean(volt)
    fft_vals = np.fft.rfft(volt * np.hanning(N))
    mag = np.abs(fft_vals) * 2 / N
    mag_db = 20 * np.log10(mag + 1e-12)
    freq_axis = np.fft.rfftfreq(N, 1/FS)

    mask = freq_axis > MIN_FREQ
    mag_db_valid = mag_db[mask]
    mag_valid = mag[mask]
    freq_axis_valid = freq_axis[mask]

    # Sort by magnitude (find top 2 peaks)
    top_indices = np.argpartition(mag_db_valid, -2)[-2:]
    sorted_peaks = top_indices[np.argsort(mag_db_valid[top_indices])[::-1]]

    main_idx = sorted_peaks[0]
    second_idx = sorted_peaks[1] if len(sorted_peaks) > 1 else main_idx

    main_freq = freq_axis_valid[main_idx]
    main_amp = mag_valid[main_idx]
    second_freq = freq_axis_valid[second_idx]
    second_amp = mag_valid[second_idx]

    rms = np.sqrt(np.mean(volt**2))

    # Harmonics and THD
    harmonic_mags = []
    for h in [2, 3, 4]:
        target = h * main_freq
        if target < FS/2:
            idx = np.argmin(np.abs(freq_axis_valid - target))
            harmonic_mags.append(mag_valid[idx])
    thd = np.sqrt(np.sum(np.array(harmonic_mags)**2)) / main_amp if harmonic_mags else 0.0

    # SNR in dB
    snr_db = 20 * np.log10(main_amp / (second_amp + 1e-12))

    return freq_axis, mag_db, main_freq, main_amp, second_freq, second_amp, rms, thd, snr_db

# ----------------- Main Loop -----------------
while True:
    try:
        frame = ser.read(FRAME_SIZE)
        if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
            continue

        adc_vals = np.frombuffer(frame[2:], dtype=np.uint16)
        volt = adc_vals * VREF / 4095.0

        (freq_axis, mag_db, main_freq, main_amp,
         second_freq, second_amp, rms, thd, snr_db) = find_main_and_secondary_freq(volt)

        # --- Time domain ---
        ax_time.cla()
        ax_time.plot(np.arange(N)/FS, volt, color='black')
        ax_time.set_xlabel("Time [s]")
        ax_time.set_ylabel("Voltage [V]")
        ax_time.set_title("Time Domain Signal")
        ax_time.grid(True)

        # --- FFT with dynamic peak colors ---
        ax_fft.cla()
        ax_fft.plot(freq_axis, mag_db, color='purple')
        second_peak_color = 'orange' if snr_db < 20 else 'gray'
        ax_fft.scatter(main_freq, 20*np.log10(main_amp), color='green', s=60, label="Main Peak")
        ax_fft.scatter(second_freq, 20*np.log10(second_amp), color=second_peak_color, s=50, label="2nd Peak (Noise)")
        ax_fft.legend(loc='upper right')
        ax_fft.set_xlabel("Frequency [Hz]")
        ax_fft.set_ylabel("Magnitude [dB]")
        ax_fft.set_title("FFT Spectrum (Main + Noise)")
        ax_fft.grid(True)

        # --- Spectrogram with dynamic lines ---
        spec_history.append(mag_db[:N//2])
        spec_array = np.array(spec_history).T
        ax_spec.cla()
        ax_spec.imshow(spec_array, aspect='auto', origin='lower',
                       extent=[0, HISTORY_LEN, 0, FS/2],
                       cmap='inferno', vmin=-100, vmax=0)
        ax_spec.set_xlabel("Frame Index")
        ax_spec.set_ylabel("Frequency [Hz]")
        ax_spec.set_title("Spectrogram (dB)")
        ax_spec.axhline(main_freq, color='green', lw=1.5, linestyle='--')
        ax_spec.axhline(second_freq, color=second_peak_color, lw=1.5, linestyle='--')

        # --- Frequency tracking ---
        freq_history.append(main_freq)
        time_history.append(time_history[-1] + 1)
        ax_track.cla()
        ax_track.plot(list(time_history), smooth(freq_history, 5), color='teal')
        ax_track.set_title("Main Frequency Tracking")
        ax_track.set_xlabel("Frame Index")
        ax_track.set_ylabel("Frequency [Hz]")
        ax_track.grid(True)
        ax_track.set_xlim(time_history[0], time_history[-1])
        ax_track.set_ylim(0, FS/2)

        # --- Text info panel with colored background & dynamic highlighting ---
        ax_text.cla()
        ax_text.axis("off")
        bg = plt.Rectangle((0,0),1,1, transform=ax_text.transAxes, color='lightyellow', alpha=0.5)
        ax_text.add_patch(bg)

        thd_color = 'red' if thd*100 > 5 else 'darkblue'
        snr_color = 'orange' if snr_db < 20 else 'darkblue'

        info_text = (
            f"Main Frequency: {main_freq:.2f} Hz\n"
            f"Second Frequency (Noise): {second_freq:.2f} Hz\n"
            f"Amplitude: {main_amp:.4f} V\n"
            f"RMS: {rms:.4f} V\n"
            f"THD: {thd*100:.2f}%\n"
            f"SNR: {snr_db:.2f} dB"
        )

        lines = info_text.split("\n")
        colors = ['darkblue', 'darkblue', 'darkblue', 'darkblue', thd_color, snr_color]
        for i, (line, color) in enumerate(zip(lines, colors)):
            ax_text.text(0.05, 1-0.15*(i+1), line, fontsize=12, family='monospace', color=color, va='top')

        # --- Performance ---
        frame_count += 1
        if time.time() - prev_time >= 1.0:
            fps_display = frame_count / (time.time() - prev_time)
            prev_time = time.time()
            frame_count = 0

        plt.pause(0.001)

    except Exception as e:
        print("Frame skipped:", e)
        continue
