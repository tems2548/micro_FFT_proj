#include <Arduino.h>
#include "driver/adc.h"

#define N 1024                     
#define ADC_CHANNEL ADC1_CHANNEL_3
const uint16_t HEADER = 0xABCD;   
#define VREF 3.3

uint16_t adc_buf[N];

void setup() {
  Serial.begin(1000000); // USB streaming
  while (!Serial) { delay(10); }

  // Configure ADC
  adc1_config_width(ADC_WIDTH_BIT_12);              
  adc1_config_channel_atten(ADC_CHANNEL, ADC_ATTEN_DB_11);
}

void loop() {
  // --- Measure start time ---
  unsigned long start = micros();

  // --- Sample N points ---
  for (int i = 0; i < N; i++) {
    adc_buf[i] = adc1_get_raw(ADC_CHANNEL);
  }

  // --- Measure end time ---
  unsigned long end = micros();

  // --- Compute elapsed time safely ---
  float elapsed_us;
  if (end >= start) {
      elapsed_us = float(end - start);
  } else {
      // micros() wrap-around
      elapsed_us = float((0xFFFFFFFFUL - start) + end);
  }

  // Avoid zero elapsed time
  if (elapsed_us < 1.0f) elapsed_us = 1.0f;

  float FS_actual = N * 1e6f / elapsed_us;

  // --- Send frame ---
  Serial.write((uint8_t *)&HEADER, sizeof(HEADER));      // 2 bytes
  Serial.write((uint8_t *)adc_buf, sizeof(adc_buf));     // N*2 bytes
  Serial.write((uint8_t *)&FS_actual, sizeof(FS_actual));// 4 bytes
}
