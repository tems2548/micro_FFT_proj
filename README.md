🧠 Real-Time FFT Spectrum Analyzer (Python + ESP32)
📘 Overview

This Python program performs real-time signal acquisition and spectrum analysis from an ESP32 (or any microcontroller) over serial communication.
It reads streaming ADC samples, computes the Fast Fourier Transform (FFT), identifies dominant frequency peaks, and displays:

Time-domain waveform

Frequency-domain (FFT) spectrum in dB

Real-time spectrogram history

Additionally, it tracks CPU temperature and frames-per-second (FPS) for performance monitoring.

⚙️ Features
Feature	Description
🎛️ Real-time Serial Acquisition	Reads binary ADC frames from ESP32 (via UART).
⚡ High-Speed Sampling	Default serial rate: 1,000,000 baud; sampling frequency ≈ 18.86 kHz.
🔍 FFT with Peak Detection	Performs windowed FFT (Hanning) and marks 2nd & 3rd harmonic peaks.
🌈 Spectrogram Visualization	Displays scrolling history of spectra over time.
💻 Performance Metrics	Displays FPS and CPU temperature (Raspberry Pi/Linux).
🧮 Auto Voltage Conversion	Converts raw ADC (12-bit) to voltage using Vref = 3.3 V.
🧩 System Requirements
Hardware

ESP32 / ESP32-C6 / Arduino / STM32, etc.
Must send raw 12-bit ADC samples in binary format with header 0xCD 0xAB.

Host PC or Raspberry Pi running Python 3.x.

Software
Library	Description	Install Command
pyserial	Serial communication	pip install pyserial
numpy	Signal processing, FFT	pip install numpy
matplotlib	Real-time plotting	pip install matplotlib
psutil	Optional CPU temperature (fallbacks to file read)	pip install psutil
🔌 Data Format (Serial Protocol)

Each frame consists of:

Field	Bytes	Description
Header	2	0xCD 0xAB
ADC Data	N * 2	12-bit unsigned integers (LSB first)

Total Frame Size = 2 + N×2 bytes

Default:
N = 1024 → Frame = 2050 bytes

🧠 How It Works
1. Serial Reading

The script continuously reads from the serial port:

frame = ser.read(FRAME_SIZE)
if len(frame) != FRAME_SIZE or frame[0:2] != HEADER:
    continue


Only valid frames starting with 0xCD 0xAB are processed.

2. ADC to Voltage Conversion

Raw 12-bit ADC values (0–4095) are converted to voltage:

volt = adc_vals * VREF / 4095.0

3. FFT Computation

Performs real FFT using a Hanning window:

fft_vals = np.fft.rfft(volt * np.hanning(N))
mag = np.abs(fft_vals) * 2 / N
mag_db = 20 * np.log10(mag + 1e-12)

4. Peak Detection

Finds 2nd and 3rd highest frequency peaks:

sorted_indices = np.argsort(mag_db)[::-1]
main_idx = sorted_indices[0]
# Exclude main peak range before finding next peaks

5. Visualization

Plots:

Time Domain: Voltage vs. time

FFT Spectrum: Magnitude (dB) vs. frequency

Spectrogram: Historical FFT frames over time

6. Performance Monitoring

FPS and CPU temperature are updated every second:

fps_display = frame_count / (time.time() - prev_time)
cpu_temp = get_cpu_temp()

🖥️ Usage

Upload Firmware
Program your ESP32 to send frames in this format:

Serial.write(0xCD);
Serial.write(0xAB);
for (int i = 0; i < 1024; i++) {
    uint16_t sample = analogRead(34);
    Serial.write((uint8_t*)&sample, 2);
}


Run Python Script

python3 realtime_fft.py


Adjust Settings

Variable	Description
PORT	Your serial port (/dev/ttyUSB0, /dev/ttyACM0, etc.)
BAUD	Baud rate (default: 1000000)
FS	Sampling frequency (default: 18860.0 Hz)
N	Number of samples per frame (default: 1024)
VREF	ADC reference voltage (default: 3.3 V)
🧰 Output Example
Console
FPS: 28.3, CPU Temp: 51.2°C
FPS: 29.1, CPU Temp: 51.4°C

Plots

Time Domain – raw voltage waveform

FFT Spectrum – magnitude in dB

Spectrogram – scrolling heatmap of frequency intensity

Red & Blue dashed lines mark the 2nd and 3rd harmonic peaks.

🧩 Tips

If the plot flickers too much, try replacing cla() with set_ydata() for smoother updates.

For better performance, disable the spectrogram or reduce HISTORY_LEN.

Works best with hardware-synchronized sampling from the ESP32 ADC DMA.

🧾 License

This project is open-source under the MIT License.
Use it freely for research, educational, or prototyping purposes.

👤 Author

Developed by Ungsuchaval Samitchart
Adapted & documented by GPT-5 for ESP32 real-time FFT applications.
