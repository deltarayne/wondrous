"""Live parameters shared between GUI and audio callback.

Reads in the audio callback are atomic (single attribute load under the GIL),
so no lock is needed for primitive fields. The GUI thread mutates these
fields directly and the callback picks them up on its next block.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Params:
    # Device + stream config (only read at engine start; restart to change)
    input_device: int = -1
    output_device: int = -1
    samplerate: int = 48000
    fft_size: int = 2048

    # Live tweakables
    mode: str = "Auto"           # "Auto" (snap to scale) | "Bar" (user-picked target)
    key: int = 0                 # 0=C ... 11=B
    scale: str = "Chromatic"
    bar_target_semitone: int = 0  # 0..11, semitones above key root; snapped to scale by GUI
    octave: int = 4              # MIDI octave (C4=60 ⇒ octave 4) the output is anchored to
    additional_range: int = 0    # extra semitones allowed above/below the chosen octave
    retune_speed: float = 20.0   # 0=instant snap, 100=slow glide
    strength: float = 100.0      # 0..100 percent of full correction
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    bypass: bool = False
