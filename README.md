# tune

Realtime and file-based FFT autotune for Windows.

Two modes:

- **Live**: capture audio from a selectable input device, snap detected pitch
  to a chosen key/scale (or a user-picked target), emit on a selectable output
  device — usually a virtual cable so other apps see it as a mic.
- **Song processing**: take a voice file (or any audio/video container) and a
  MIDI file, repitch the voice to follow the song, write the result to a new
  audio file. Optional vibrato, formant preservation, reverb, dry/wet mix,
  time-stretch (prepitch speed), volume shaping, vocal-range restriction,
  song overlay, key-centered snapping, etc.

## Install

```sh
pip install -r requirements.txt
```

`sounddevice` ships with a bundled PortAudio binary on Windows, so the live
engine has no extra runtime requirements.

For song processing with **MP3 / M4A / AAC / MP4 / MOV / MKV / WEBM** input
or MP3 output, `ffmpeg` must be on PATH. WAV / FLAC / OGG / AIFF go through
`soundfile` directly and don't need ffmpeg. Get a Windows build from
<https://www.gyan.dev/ffmpeg/builds/> and add the `bin` folder to PATH, or
`winget install ffmpeg`.

On Python 3.13+, `audioop-lts` (auto-installed) replaces the removed stdlib
`audioop` module that `pydub` depends on.

## Run

```sh
python -m tune
```

A console-script entry point is also installed via `pyproject.toml`:

```sh
tune
```

## Routing into other apps

To use the corrected audio as a microphone in another app (Discord, OBS, …),
install [VB-Audio CABLE](https://vb-audio.com/Cable/), pick `CABLE Input` as
the output device in tune, then select `CABLE Output` as the microphone in
the receiving app.

## Live controls

### Audio devices

- **Input / Output**: enumerated PortAudio devices. *Refresh* re-scans.
- **Sample rate**: 44.1 or 48 kHz. Both devices must support it.
- **FFT size**: 1024 / 2048 / 4096. Larger = better frequency resolution but
  more latency. Hop = FFT/4 (75 % overlap, Hann window).

### Pitch correction

- **Mode**: `Auto` snaps to the nearest scale note; `Bar` lets a slider pick
  one scale degree directly and pins the target to the chosen octave.
- **Key + Scale**: target pitch class set. Major, Natural Minor, Harmonic
  Minor, Major / Minor Pentatonic, Blues, Dorian, Mixolydian, or Chromatic.
- **Octave + Additional step range**: bounds the snap candidates to a window
  of `[12·(octave+1) − range, 12·(octave+2) − 1 + range]` semitones. With
  range = 0, output is locked to the chosen octave; raise it to give the
  snap more freedom.
- **Target** *(Bar mode only)*: slider snaps to the scale degrees of the
  current key.
- **Retune speed**: 0 % = instant snap (T-Pain), 100 % = slow glide.
- **Strength**: 0 % = pass-through (still through the FFT), 100 % = full
  correction.

### Levels & monitoring

- **Input / Output gain**: ±24 dB trim.
- **Input / Output meters**: per-block peak.
- **Detected**: live readout of the detected fundamental, the closest scale
  note, and the cents offset.

### Transport

- **Start / Stop**: opens / closes the audio stream.
- **Bypass**: pass input straight to output (still goes through the FFT
  pipeline, just at unity ratio).
- **Status line**: sample rate / FFT / hop / dropout count. Non-zero
  dropouts mean the OS audio stack is starving — usually a buffer-size or
  driver issue, not the algorithm.

### Settings (File → Settings…)

- **Interface theme**: Light / Dark. Persisted to `~/.tune/config.json`.

## Song processing (File → Song processing…)

Pick a voice file, a MIDI file, an output path; the output is a new audio
file with the voice repitched to follow the MIDI. The voice file on disk is
never modified.

### Files

- **Voice file**: WAV / FLAC / OGG / AIFF (libsndfile native) or MP3 / M4A /
  AAC / MP4 / MOV / MKV / WEBM (via pydub + ffmpeg — first audio track is
  extracted, video is discarded).
- **MIDI file**: standard `.mid` / `.midi`.
- **Output**: WAV / FLAC / OGG (soundfile) or MP3 (pydub). Default if
  blank: `~/Documents/wondrous/<voice_stem>_tuned.wav` (folder created on
  save).

Last-used paths are remembered in `~/.tune/config.json` and preloaded the
next time the dialog opens (only if the files still exist on disk).

### MIDI track / channel

- **Track**: dropdown listing every (track, channel) that has notes, sorted
  by note count. The default `Auto melody (all tracks)` aggregates every
  non-drum note across every track and uses the highest-pitched active note
  at each instant — keeps the voice tracking the song's melodic top line
  even when the chosen instrument's part ends mid-song.
- **Play preview / Stop**: synthesises the chosen track as additive sine
  tones so you can identify the right melody by ear.

### Mix

- **Song overlay**: render the full MIDI as additive-sine audio and mix it
  into the output, looped on the same cycle as the pitch changes. The
  output then plays *with* an instrumental backing rather than just the
  bare voice.
  - **Vol** (0–100 %): how loud the overlay sits in the mix.

### Voice shaping

Each toggle is independent and persists separately.

- **Volume shaping**: spike envelope at note onsets (+6 dB after ≥50 ms of
  silence, +3 dB at note-to-note pitch changes), exponential decay back to
  unity over ~120 ms. Makes the voice "lean into" each note like a singer.
- **Restrict to human vocal range (C1..C4)**: two-layer enforcement. The
  song-wide alignment picks an octave that fits the MIDI inside C1..C4 if
  possible; a per-frame post-clamp folds any outliers by whole octaves
  until they're inside. Hard ceiling/floor — the chipmunk-prevention knob.
- **Key centered logic**: replace per-frame melody following with
  Krumhansl-Schmuckler key estimation (sliding 6 s window, half-second
  steps) and snap the voice to the nearest scale note in the current key.
  Useful when you want the voice to *be in the song's key* without trying
  to track every melodic leap.
- **Lower MIDI one octave**: drops the entire output by 12 semitones from
  whatever the alignment / snap would have produced. Combines with the
  vocal-range clamp — if it would push below C1, the clamp folds back up.
- **Prepitch speed (50–100 %)**: time-stretch the voice via linear
  interpolation before the autotune step. The autotune naturally restores
  the pitch (uses an override pitch contour pre-detected on the original
  voice, since the slowed fundamental is below the autotuner's vocal-range
  mask). Output ends up `1 / pct`× longer at the original pitch.
- **Dry/wet mix**: blend the processed voice with the original at 0–100 %.
  At 80 %, 20 % of the dry voice keeps the singer's natural timbre and
  breath while the remainder carries the melody — softens the "robotic"
  edge a lot.
- **Vibrato**: sine LFO modulating target Hz. Per-note depth ramp so
  attacks stay crisp (50 ms hold + 200 ms ramp). Rate 3–7 Hz, depth 0–50 ¢.
- **Retune speed (0–100)**: same parameter as the live engine's smoother;
  exposes the EWMA time constant on the per-frame ratio. 0 = instant snap,
  100 = ~250 ms glide (heavy portamento).
- **Formant preservation**: cepstral lifter that keeps the vocal-tract
  resonances at their original frequencies regardless of pitch shift —
  fixes the chipmunk effect on upshifts and the mumble on downshifts.
  Frequency-dependent soft blend (full correction at DC, fading to none at
  Nyquist) prevents the high-frequency thinning that plagues naive lifter
  implementations on big shifts.
- **Reverb (small room)**: convolution reverb with a calibrated random-tap
  IR (~80 ms tail). Wet 0–30 %.

### Process

Progress bar + status line. Background worker thread keeps the dialog
responsive; cancel via the X / Cancel.

### Loop behaviour

The output is always voice-length. The MIDI looped on a cycle of
`song_end + 0.5 s` (a half-second silence between repeats) — voice longer
than song wraps the song around as many times as needed. Voice shorter
than song just stops where the voice does. Pitch transitions across the
loop wrap and across the silence pause use the smoother for a soft fade.

## DSP overview

Live and offline share the same `Autotuner`:

1. Slide each block into a windowed analysis buffer (Hann, 75 % overlap).
2. `rfft` → magnitude + phase spectra.
3. **Pitch detection**: ACF via FFT (`|X(f)|² → ifft`), debiased by the
   analysis window's own autocorrelation. Vocal-range bandpass (90–2500 Hz)
   suppresses sub-bass rumble and HVAC hiss before correlation. Octave-error
   correction walks down sub-multiples (4, 3, 2). Voicing hysteresis
   (enter 0.4, exit 0.25 of `e₀`). 5-frame median + outlier hold guard against
   single-frame jumps.
4. **Target selection**: depending on mode, snap with hysteresis, place at the
   user-picked Bar slider position, or use a per-frame target supplied by
   song mode.
5. **Phase vocoder pitch shift**: compute true bin frequencies from phase
   deltas, spectral-interpolate magnitudes and frequencies to the source
   bins `K_out / ratio`, energy-normalise.
6. **Laroche–Dolson loose phase locking**: detect peaks in the input
   magnitude, lock each non-peak output bin's synthesis phase to its owning
   peak's phase + the input-side offset.
7. **Optional formant preservation** (cepstral lifter): extract low-quefrency
   envelope from input log-mag, multiply output by `env_in / env_out`,
   blended toward 1.0 at high frequencies.
8. `irfft` → window → overlap-add → emit one hop.

Per-frame ratio is clamped to ±3 octaves; the song-wide octave alignment
and vocal-range constraints are what musically bound it in normal use.

## Latency

At FFT = 2048 / 48 kHz the inherent FFT delay is ~43 ms (analysis frame
size). Hop is 512 samples (~10.7 ms) — that's the callback period. Total
round-trip latency depends on the OS audio stack; on WASAPI shared mode,
expect 60–80 ms. ASIO drivers (if your interface ships them) give tighter
numbers.

Song processing has no real-time constraint, so it runs at whatever speed
the CPU allows — usually a few seconds per minute of audio at default
settings.

## Limitations

- **Monophonic only**: ACF assumes a single fundamental.
- **Pitch detector floor ≈ 70 Hz**: voices below that (extreme bass) will
  fall back to a generic mid-vocal default for the median-alignment
  pre-pass; song-wide octave choice may be off for those voices.
- **Wide-range MIDI vs. narrow vocal window**: when the chosen track's
  pitch range exceeds the active vocal-range constraint, the per-frame
  clamp folds outliers by octaves — preserves the range guarantee at the
  cost of breaking the voice-direction-matches-melody invariant for those
  notes.
- **Very large pitch shifts** (> ~24 semitones either direction) push the
  phase vocoder past its clean range. Output stays usable but loses
  brightness / gains aliasing.
- Both live audio devices must support the chosen sample rate.
