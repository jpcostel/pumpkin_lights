#!/home/jpcostel/Projects/pumpkin_lights/neopixel_venv/bin/python3

#!/usr/bin/env python3
"""
neopixel_equalizer.py

Requirements:
    pip3 install numpy scipy sounddevice adafruit-circuitpython-neopixel
    Additionally install and enable Adafruit Blinka for CircuitPython compatibility:
    pip3 install adafruit-blinka

Hardware:
    - Raspberry Pi 4B+ with a microphone (USB or HAT) set as default input, or specify device index.
    - 64 WS2812/NeoPixel LEDs wired as one continuous strip in ring order:
        ring0 indexes 0..15 (closest to Pi)
        ring1 indexes 16..31
        ring2 indexes 32..47
        ring3 indexes 48..63 (farthest)
    - Data pin: default is board.D18 (PWM). Use a level shifter & proper 5V power supply.
    - Be sure to supply enough current for maximum brightness (64 LEDs * ~60mA ≈ 3.84A at full white).
"""

import time
import math
import numpy as np
import sounddevice as sd
from scipy.signal import get_window
import board
import neopixel
import threading
from collections import deque

# ---------- CONFIG ----------
SAMPLE_RATE = 44100
BLOCKSIZE = 2048            # audio frames per buffer (power-of-two)
CHANNELS = 1
FFT_SIZE = BLOCKSIZE
HOP_TIME = BLOCKSIZE / SAMPLE_RATE
UPDATE_HZ = 60              # target visual update rate
PIXEL_COUNT = 64
LED_PIN = board.D18         # change if needed
BRIGHTNESS = 0.4            # global brightness (0.0 - 1.0), scaled by ring intensity too
AUTO_GAIN = True            # apply simple per-band normalization
MAX_EXTRA_PIXELS = 4        # how many extra pixels per side maximum on beat
BEAT_SENSITIVITY = 2.0     # energy must exceed baseline * this factor to count as beat
BEAT_DECAY = 0.95          # how fast the beat growth decays each update
ENERGY_EMA_ALPHA = 0.2     # smoothing for band energy baseline
INTENSITY_SMOOTH_ALPHA = 0.25  # smoothing for intensity -> rotation speed/brightness

# Frequency bands (Hz) low to high: typical instrument ranges
BANDS = [
    (20, 120),      # Bass: kick, bass guitar
    (120, 500),     # Low-mid: guitars, lower keys
    (500, 2000),    # Mid: vocals, keys
    (2000, 8000)    # High: cymbals, presence
]

# Halloween color cycle (R,G,B)
COLOR_CYCLE = [
    (128, 0, 128),   # purple
    (255, 128, 0),   # orange
    (0, 200, 0)      # green
]

# Mapping ring -> global pixel indices
RING_SIZE = 16
def ring_base_index(ring):
    return ring * RING_SIZE

# ---------- END CONFIG ----------

# Initialize Neopixels
pixels = neopixel.NeoPixel(LED_PIN, PIXEL_COUNT, brightness=BRIGHTNESS, auto_write=False, pixel_order=neopixel.GRB)

# Precompute FFT frequency bin centers
bin_freqs = np.fft.rfftfreq(FFT_SIZE, 1.0 / SAMPLE_RATE)

# Precompute bin index ranges for each band
band_bins = []
for (f0, f1) in BANDS:
    idx0 = np.searchsorted(bin_freqs, f0, side='left')
    idx1 = np.searchsorted(bin_freqs, f1, side='right')
    band_bins.append((idx0, max(idx1, idx0+1)))

# Thread-safe audio buffer storage
audio_queue = deque(maxlen=4)

# Input stream callback — push blocks into audio_queue
def audio_callback(indata, frames, time_info, status):
    if status:
        # Non-fatal status messages from sounddevice
        print("Audio status:", status, flush=True)
    # Flatten to mono
    block = np.array(indata[:, 0], dtype=np.float32)
    audio_queue.append(block)

# Start audio input stream
stream = sd.InputStream(channels=CHANNELS, callback=audio_callback, blocksize=BLOCKSIZE, samplerate=SAMPLE_RATE)
stream.start()

# Per-band state
class BandState:
    def __init__(self, ring_index):
        self.ring = ring_index
        self.energy_ema = 1e-8       # baseline energy (avoid zero)
        self.intensity = 0.0        # recent smoothed intensity (0..)
        self.angle = 0.0            # rotation angle in pixel-space (0..15)
        self.direction = 1          # 1 or -1
        self.color_idx = ring_index % len(COLOR_CYCLE)
        self.extra_pixels = 0.0     # growth amount (0..MAX_EXTRA_PIXELS)
        self.last_beat_time = 0.0

bands = [BandState(i) for i in range(len(BANDS))]

# Hann window for FFT
window = get_window('hann', FFT_SIZE)

# Utility: compute band energies from a mono frame (frame length = FFT_SIZE)
def compute_band_energies(frame):
    # if frame shorter, pad
    if len(frame) < FFT_SIZE:
        frame = np.pad(frame, (0, FFT_SIZE - len(frame)))
    # apply window
    xw = frame * window
    X = np.fft.rfft(xw, n=FFT_SIZE)
    mag = np.abs(X)
    power = mag ** 2
    band_powers = []
    for (i0, i1) in band_bins:
        # sum power in band (could also use mean)
        band_power = power[i0:i1].sum()
        band_powers.append(band_power)
    return band_powers

# HSV-like brightness scaling helper (no external lib)
def scale_color(rgb, scale):
    r, g, b = rgb
    r = int(max(0, min(255, r * scale)))
    g = int(max(0, min(255, g * scale)))
    b = int(max(0, min(255, b * scale)))
    return (r, g, b)

# Main update loop
def main_loop():
    last_time = time.time()
    target_dt = 1.0 / UPDATE_HZ
    try:
        while True:
            t0 = time.time()
            # Collect enough audio blocks to fill FFT frame length
            # We'll combine the queued blocks into one frame; if not enough, zero-pad
            collected = np.zeros(FFT_SIZE, dtype=np.float32)
            if len(audio_queue) > 0:
                # pop newest blocks until we've filled FFT_SIZE or queue empty
                # we'll build a rolling buffer by concatenation of most recent blocks
                blocks = list(audio_queue)
                combined = np.concatenate(blocks, axis=0)
                if len(combined) >= FFT_SIZE:
                    # take last FFT_SIZE samples
                    frame = combined[-FFT_SIZE:]
                else:
                    # pad left with zeros so recent samples at end
                    frame = np.pad(combined, (FFT_SIZE - len(combined), 0))
            else:
                frame = np.zeros(FFT_SIZE, dtype=np.float32)

            # compute band energies
            band_powers = compute_band_energies(frame)

            # normalize / compute per-band processing
            for i, power in enumerate(band_powers):
                b = bands[i]
                # update baseline EMA
                b.energy_ema = (1 - ENERGY_EMA_ALPHA) * b.energy_ema + ENERGY_EMA_ALPHA * (power + 1e-10)
                # relative intensity
                if AUTO_GAIN:
                    rel = power / (b.energy_ema + 1e-12)  # ratio to baseline
                    # map to 0..some value; subtract 1 so quiet -> ~0
                    intensity = max(0.0, rel - 1.0)
                else:
                    intensity = power
                # smooth intensity
                b.intensity = (1 - INTENSITY_SMOOTH_ALPHA) * b.intensity + INTENSITY_SMOOTH_ALPHA * intensity

                # beat detection: if raw power significantly exceeds baseline
                beat = False
                if power > b.energy_ema * BEAT_SENSITIVITY and (time.time() - b.last_beat_time) > 0.08:
                    beat = True
                    b.last_beat_time = time.time()

                # on beat: switch color, reverse direction, grow extra_pixels proportional to how strong
                if beat:
                    b.color_idx = (b.color_idx + 1) % len(COLOR_CYCLE)
                    b.direction *= -1
                    # intensity ratio -> growth
                    # stronger beats produce more extra pixels
                    growth = min(MAX_EXTRA_PIXELS, 1 + (power - b.energy_ema * BEAT_SENSITIVITY) / (b.energy_ema + 1e-12))
                    growth = max(1.0, growth)
                    b.extra_pixels = min(MAX_EXTRA_PIXELS, b.extra_pixels + growth)

                # decay extra_pixels
                b.extra_pixels *= BEAT_DECAY
                if b.extra_pixels < 0.01:
                    b.extra_pixels = 0.0

                # update rotation angle: speed proportional to intensity (and a base speed)
                base_speed = 0.6   # pixels per second baseline
                speed = base_speed + (b.intensity * 5.0)  # tweak multiplier to taste
                # convert speed to pixel-step for this frame
                dt = time.time() - last_time if last_time else target_dt
                b.angle = (b.angle + b.direction * speed * dt) % RING_SIZE

            last_time = time.time()

            # Build pixel frame
            pixels_to_set = [(0,0,0)] * PIXEL_COUNT

            for i, b in enumerate(bands):
                ring_start = ring_base_index(i)
                center = int(round(b.angle)) % RING_SIZE
                opp = (center + (RING_SIZE//2)) % RING_SIZE  # opposite (8 offset)
                # compute number of pixels to light on each end
                extra = int(round(b.extra_pixels))
                # base is 0 meaning center pixel only, but user wanted "one pixel on across the ring from each other (8 pixel offset)"
                # We'll light center and opposite, plus extra on each side up to MAX_EXTRA_PIXELS
                indices = set()
                # for center side
                for e in range(-extra, extra+1):
                    idx = (center + e) % RING_SIZE
                    indices.add(idx)
                # for opposite side
                for e in range(-extra, extra+1):
                    idx = (opp + e) % RING_SIZE
                    indices.add(idx)

                # brightness scale from intensity (0..1-ish), cap
                # map b.intensity (smoothed) to 0..1 via arctan-like scaling to avoid bursts
                intensity_scale = math.tanh(b.intensity * 0.7)  # tuned value
                # overall brightness multipler
                ring_brightness = 0.15 + 0.85 * intensity_scale  # never completely dark
                # compute color
                color_rgb = COLOR_CYCLE[b.color_idx]
                # final color scaled
                color_final = scale_color(color_rgb, ring_brightness)

                for pix_in_ring in indices:
                    global_idx = ring_start + pix_in_ring
                    pixels_to_set[global_idx] = color_final

                # Add a subtle trailing pixel to show motion: light a pixel slightly behind current center
                trail_idx = (center - b.direction) % RING_SIZE
                global_trail = ring_start + trail_idx
                # trail color with lower brightness
                trail_color = scale_color(color_rgb, max(0.08, 0.35 * intensity_scale))
                # only set if not already set (so no override)
                if pixels_to_set[global_trail] == (0,0,0):
                    pixels_to_set[global_trail] = trail_color

            # Write to hardware
            for idx, col in enumerate(pixels_to_set):
                pixels[idx] = col
            pixels.show()

            # throttle to target rate
            elapsed = time.time() - t0
            sleep_time = max(0, target_dt - elapsed)
            time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        # turn off LEDs
        pixels.fill((0,0,0))
        pixels.show()
        stream.stop()
        stream.close()

if __name__ == "__main__":
    print("Starting NeoPixel equalizer. Press Ctrl+C to stop.")
    print(f"Sample rate: {SAMPLE_RATE}, blocksize: {BLOCKSIZE}, FFT size: {FFT_SIZE}")
    main_loop()
