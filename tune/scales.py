"""Musical scale definitions and frequency/MIDI helpers."""
from __future__ import annotations

import numpy as np

NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

SCALES: dict[str, list[int]] = {
    "Chromatic":        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    "Major":            [0, 2, 4, 5, 7, 9, 11],
    "Natural Minor":    [0, 2, 3, 5, 7, 8, 10],
    "Harmonic Minor":   [0, 2, 3, 5, 7, 8, 11],
    "Major Pentatonic": [0, 2, 4, 7, 9],
    "Minor Pentatonic": [0, 3, 5, 7, 10],
    "Blues":            [0, 3, 5, 6, 7, 10],
    "Dorian":           [0, 2, 3, 5, 7, 9, 10],
    "Mixolydian":       [0, 2, 4, 5, 7, 9, 10],
}

SCALE_NAMES = list(SCALES.keys())


def freq_to_midi(f: float) -> float:
    return 69.0 + 12.0 * np.log2(f / 440.0)


def midi_to_freq(m: float) -> float:
    return 440.0 * (2.0 ** ((m - 69.0) / 12.0))


def snap_freq(f: float, key: int, scale_name: str) -> float:
    """Snap a frequency to the nearest note in (key, scale)."""
    if f <= 0.0:
        return f
    intervals = SCALES.get(scale_name, SCALES["Chromatic"])
    midi = freq_to_midi(f)
    octave = int(round(midi / 12.0))
    candidates = [
        o * 12 + key + i
        for o in (octave - 1, octave, octave + 1)
        for i in intervals
    ]
    arr = np.asarray(candidates, dtype=np.float64)
    nearest = arr[int(np.argmin(np.abs(arr - midi)))]
    return float(midi_to_freq(nearest))


def midi_to_name(m: float) -> str:
    n = int(round(m))
    octave = n // 12 - 1
    return f"{NOTE_NAMES[n % 12]}{octave}"


def cents_off(f: float, target_f: float) -> float:
    if f <= 0.0 or target_f <= 0.0:
        return 0.0
    return 1200.0 * float(np.log2(f / target_f))
