# tune

Realtime FFT-based autotune for Windows.

Picks an input audio device, runs the signal through a phase-vocoder pitch
shifter that snaps detected pitch to the nearest note in a chosen key/scale,
and sends the corrected audio to a selectable output device.

## Install

```sh
pip install -r requirements.txt
```

`sounddevice` ships with a bundled PortAudio binary on Windows, so nothing
else is required.

## Run

```sh
python -m tune
```

## Routing into other apps

To use the corrected audio as a microphone in another app (Discord, OBS,
etc.), install [VB-Audio CABLE](https://vb-audio.com/Cable/), pick `CABLE
Input` as the output device in tune, then select `CABLE Output` as the
microphone in the receiving app.

## Controls

- **Input / Output**: audio devices (Refresh re-enumerates).
- **Sample rate**: 44.1 or 48 kHz. Both devices must support it.
- **FFT size**: 1024 / 2048 / 4096. Larger = better frequency resolution but
  more latency. Hop = FFT/4 (75% overlap, Hann window).
- **Key + Scale**: target notes the detected pitch will snap to.
- **Retune speed**: 0% = instant snap (T-Pain), 100% = slow glide.
- **Strength**: 0% = passthrough (still through the FFT), 100% = full
  correction.
- **Input / Output gain**: ±24 dB trim.
- **Bypass**: copy input straight to output (no pitch processing).

## Algorithm

1. Slide each block into a windowed analysis buffer.
2. `rfft` → magnitude + phase spectra.
3. Pitch detection via Harmonic Product Spectrum (4 harmonics, 70–1000 Hz).
4. Snap detected f0 to the nearest scale note; smooth the shift ratio with an
   exponential filter governed by Retune speed.
5. Phase vocoder: compute true bin frequencies from phase deltas, scatter
   magnitudes and frequencies to ratio-shifted bins, advance synthesis
   phases.
6. `irfft` → window → overlap-add → emit one hop.

## Latency

At FFT=2048 / 48 kHz, the inherent FFT delay is ~43 ms (analysis frame size).
Hop is 512 samples (~10.7 ms) — that's the callback period. Total round-trip
latency depends on the OS audio stack; on WASAPI shared mode, expect
60–80 ms. Use ASIO drivers (if your interface ships them) for tighter
numbers.

## Limitations

- Monophonic only (HPS pitch detector assumes one fundamental).
- No formant preservation in v1.
- Both selected devices must support the chosen sample rate.
