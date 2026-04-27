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
    key: int = 0                 # 0=C ... 11=B
    scale: str = "Chromatic"
    retune_speed: float = 20.0   # 0=instant snap, 100=slow glide
    strength: float = 100.0      # 0..100 percent of full correction
    input_gain_db: float = 0.0
    output_gain_db: float = 0.0
    bypass: bool = False
