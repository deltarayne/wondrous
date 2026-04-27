"""Offline song processing: read voice audio + MIDI, repitch the voice so
its detected fundamental tracks the notes on a chosen MIDI track/channel.

The voice file is left untouched on disk; a new file is written at the
requested output path (default ``~/Documents/wondrous/<voice>_tuned.wav``).
"""
from __future__ import annotations

import threading
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

import mido
import soundfile as sf

from .dsp import Autotuner
from .scales import freq_to_midi, midi_to_freq, midi_to_name


# Audio formats we read with libsndfile via soundfile. Anything else gets
# routed through pydub (which needs ffmpeg on PATH for MP3/AAC).
_SOUNDFILE_EXTS = {".wav", ".flac", ".ogg", ".aiff", ".aif"}
# Output formats we accept via the save dialog.
OUTPUT_EXTS = (".wav", ".flac", ".ogg", ".mp3")
# Codec quality knobs for MIDI preview synthesis.
_PREVIEW_SR = 44100
_PREVIEW_ATTACK_S = 0.005
_PREVIEW_RELEASE_S = 0.05
# Half-second silence between MIDI loops, per the user spec.
LOOP_PAUSE_S = 0.5
# Retune-speed used in song mode. The smoother gives the in/out ramp the
# user asked for; ~76 ms is short enough to feel snappy on note changes
# but long enough to avoid clicks at silence boundaries.
SONG_RETUNE_SPEED = 30.0
# Voice pre-pass: only sample every Nth hop when estimating median pitch.
# Cuts a 5-minute file's pre-pass from ~30 s to a few seconds without
# meaningfully degrading the median estimate.
VOICE_MEDIAN_HOP_STRIDE = 4
# When the voice file has no detectable pitch (silence/noise only), fall
# back to a typical mid-vocal pitch. A3 ≈ 220 Hz is a reasonable centre
# for both speaking and singing voices.
VOICE_MEDIAN_FALLBACK_MIDI = 57.0


# ---------- MIDI ----------------------------------------------------------

@dataclass
class MidiNote:
    start: float  # seconds
    end: float    # seconds
    pitch: int    # MIDI note number


@dataclass
class TrackOption:
    """One selectable target option from a MIDI file. Either a real
    (track, channel) pair, or the synthetic "auto melody" pseudo-option
    that aggregates the highest-pitched note across every non-drum
    track/channel at each instant — flagged via ``track_idx == -1``.
    """
    track_idx: int
    channel: int
    track_name: str
    notes: list[MidiNote] = field(default_factory=list)

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def end_time(self) -> float:
        return max((n.end for n in self.notes), default=0.0)

    @property
    def is_drum_channel(self) -> bool:
        # MIDI channel 10 in 1-indexed convention = channel 9 in 0-indexed.
        return self.channel == 9

    @property
    def is_auto_melody(self) -> bool:
        return self.track_idx < 0

    @property
    def label(self) -> str:
        if self.is_auto_melody:
            return f"{self.track_name}  —  {self.note_count} notes"
        nm = self.track_name or f"Track {self.track_idx}"
        suffix = " [drums]" if self.is_drum_channel else ""
        return f"{nm} (ch {self.channel + 1})  —  {self.note_count} notes{suffix}"


def load_midi_options(path: str) -> tuple[float, list[TrackOption]]:
    """Parse ``path`` and return (overall_end_time, [TrackOption, ...])
    sorted by note count descending so likely-melody tracks float to the top.
    """
    mid = mido.MidiFile(path)
    tpb = mid.ticks_per_beat or 480

    # Build a global tempo map (abs_tick -> tempo in microseconds-per-beat),
    # since tempo events can live on any track but apply globally.
    tempo_events: list[tuple[int, int]] = [(0, 500000)]  # default 120 BPM
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((abs_tick, msg.tempo))
    tempo_events.sort()

    def tick_to_sec(tick: int) -> float:
        seconds = 0.0
        prev_tick = 0
        prev_tempo = 500000
        for t, tempo in tempo_events:
            if t >= tick:
                break
            seconds += (t - prev_tick) * prev_tempo / (tpb * 1_000_000)
            prev_tick = t
            prev_tempo = tempo
        seconds += (tick - prev_tick) * prev_tempo / (tpb * 1_000_000)
        return seconds

    bins: dict[tuple[int, int], TrackOption] = {}
    track_names: dict[int, str] = {}
    overall_end = 0.0

    for track_idx, track in enumerate(mid.tracks):
        abs_tick = 0
        # (channel, pitch) -> start_tick
        active: dict[tuple[int, int], int] = {}
        for msg in track:
            abs_tick += msg.time
            if msg.type == "track_name":
                track_names[track_idx] = (msg.name or "").strip()
            elif msg.type == "note_on" and msg.velocity > 0:
                active[(msg.channel, msg.note)] = abs_tick
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                key = (msg.channel, msg.note)
                start_tick = active.pop(key, None)
                if start_tick is None:
                    continue
                start = tick_to_sec(start_tick)
                end = tick_to_sec(abs_tick)
                if end <= start:
                    continue
                opt_key = (track_idx, msg.channel)
                opt = bins.setdefault(opt_key, TrackOption(
                    track_idx=track_idx, channel=msg.channel, track_name="",
                ))
                opt.notes.append(MidiNote(start, end, msg.note))
                overall_end = max(overall_end, end)
        # Force-close any notes left hanging at end-of-track.
        for (channel, pitch), start_tick in active.items():
            start = tick_to_sec(start_tick)
            end = tick_to_sec(abs_tick)
            if end <= start:
                continue
            opt_key = (track_idx, channel)
            opt = bins.setdefault(opt_key, TrackOption(
                track_idx=track_idx, channel=channel, track_name="",
            ))
            opt.notes.append(MidiNote(start, end, pitch))
            overall_end = max(overall_end, end)

    options: list[TrackOption] = []
    for opt in bins.values():
        opt.track_name = track_names.get(opt.track_idx, "")
        opt.notes.sort(key=lambda n: n.start)
        options.append(opt)
    options.sort(key=lambda o: -o.note_count)

    # Synthesise the "auto melody" option: every non-drum note from every
    # track/channel, merged. ``MidiTimeline.target_hz_at`` already picks the
    # highest active pitch, so this gives the cross-track melody envelope
    # the user wants by default. Inserted at index 0 so it's the dropdown
    # default.
    melody_notes: list[MidiNote] = []
    for opt in options:
        if opt.is_drum_channel:
            continue
        melody_notes.extend(opt.notes)
    melody_notes.sort(key=lambda n: n.start)
    auto_melody = TrackOption(
        track_idx=-1, channel=-1,
        track_name="Auto melody (all tracks)",
        notes=melody_notes,
    )
    options.insert(0, auto_melody)
    return overall_end, options


class MidiTimeline:
    """Fast lookup of "the highest active MIDI pitch at time t" for a track.

    Used both to decide the per-frame target during voice processing and to
    drive the silence/wrap-pause ramp envelope: when no note is active, this
    returns ``None`` and the caller treats that as "no shift".
    """

    def __init__(self, notes: list[MidiNote]) -> None:
        self._notes: list[MidiNote] = sorted(notes, key=lambda n: n.start)
        self.starts = np.array([n.start for n in self._notes], dtype=np.float64)
        self.ends = np.array([n.end for n in self._notes], dtype=np.float64)
        self.pitches = np.array([n.pitch for n in self._notes], dtype=np.int32)

    def target_hz_at(self, t: float) -> float | None:
        hz, _ = self.target_at(t)
        return hz

    def target_at(self, t: float) -> tuple[float | None, MidiNote | None]:
        """Same lookup but also returns the chosen ``MidiNote`` (the highest
        active pitch at time ``t``). Vibrato/onset-aware features need the
        note's start time so they can ramp per-note."""
        if self.starts.size == 0:
            return None, None
        i = int(np.searchsorted(self.starts, t, side="right"))
        if i == 0:
            return None, None
        active_mask = self.ends[:i] > t
        if not active_mask.any():
            return None, None
        active_indices = np.where(active_mask)[0]
        local_idx = int(np.argmax(self.pitches[active_indices]))
        chosen = active_indices[local_idx]
        note = self._notes[int(chosen)]
        return float(midi_to_freq(float(note.pitch))), note


def _render_notes_audio(
    notes: list[MidiNote], total_duration_s: float, sr: int,
    per_note_amplitude: float = 0.22,
) -> np.ndarray:
    """Render ``notes`` to an additive-sine mono buffer of length
    ``total_duration_s`` seconds at ``sr``. Each note gets a 5 ms attack
    and 50 ms release. Output is tanh-saturated to keep dense polyphony
    from clipping.
    """
    n = max(1, int(total_duration_s * sr))
    audio = np.zeros(n, dtype=np.float32)
    if not notes:
        return audio
    attack_n = max(1, int(_PREVIEW_ATTACK_S * sr))
    release_n = max(1, int(_PREVIEW_RELEASE_S * sr))
    for note in notes:
        freq = float(midi_to_freq(float(note.pitch)))
        i0 = int(note.start * sr)
        i1 = int(note.end * sr)
        if i1 <= i0 or i0 >= n:
            continue
        i1 = min(i1, n)
        nn = i1 - i0
        t = np.arange(nn, dtype=np.float32) / sr
        sig = (per_note_amplitude * np.sin(2.0 * np.pi * freq * t)).astype(np.float32)
        env = np.ones(nn, dtype=np.float32)
        a = min(attack_n, nn // 2)
        r = min(release_n, nn // 2)
        if a > 0:
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if r > 0:
            env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
        audio[i0:i1] += sig * env
    return np.tanh(audio).astype(np.float32)


def render_midi_preview(option: TrackOption, sr: int = _PREVIEW_SR) -> np.ndarray:
    """Render a track/channel as additive sine tones for the preview player.

    Polyphonic content is rendered as actual chords (notes sum) rather than
    just the highest pitch — the user uses this to identify the right
    track, so they should hear what's actually there.
    """
    if not option.notes:
        return np.zeros(0, dtype=np.float32)
    return _render_notes_audio(option.notes, option.end_time + 0.25, sr)


def estimate_voice_pitch_contour(
    audio_mono: np.ndarray, sr: int, fft_size: int = 2048,
) -> np.ndarray:
    """Per-hop fundamental-frequency contour for the entire audio. Length =
    number of full ``hop`` windows that fit in ``audio_mono``. Zeros for
    unvoiced frames. Used by song mode when the voice has been time-stretched
    into a region the autotuner's per-frame detector can't see — we run the
    detector on the *original* voice, then translate to the slowed timeline.
    A fresh ``Autotuner`` is used so its internal voicing/median-filter
    state doesn't bleed into the channel-processing tuners.
    """
    tuner = Autotuner(fft_size, sr)
    hop = tuner.hop
    n = int(audio_mono.shape[0])
    if n < fft_size:
        return np.zeros(0, dtype=np.float64)
    n_hops = (n - fft_size) // hop + 1
    contour = np.zeros(n_hops, dtype=np.float64)
    for h in range(n_hops):
        block = audio_mono[h * hop : h * hop + fft_size]
        windowed = block * tuner.window
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec)
        contour[h] = float(tuner._detect_pitch_robust(mag))
    return contour


def _linear_resample(audio: np.ndarray, factor: float) -> np.ndarray:
    """Stretch ``audio`` to ``factor`` × its current length via linear
    interpolation. ``factor > 1`` stretches (slows + pitch-drops at constant
    sr); ``factor < 1`` compresses. Output frame ``k`` is sampled at input
    position ``k / factor`` — exact uniform stride matters: ``np.linspace``
    rolls a tiny non-uniform step into the result that shows up as a beat
    a few cents off the true pitch when the autotuner shifts it.
    """
    if abs(factor - 1.0) < 1e-9 or audio.shape[0] == 0:
        return audio
    n_in = int(audio.shape[0])
    n_out = max(1, int(round(n_in * float(factor))))
    x_old = np.arange(n_in, dtype=np.float64)
    x_new = np.arange(n_out, dtype=np.float64) / float(factor)
    # Clamp the last fractional position to stay within ``x_old``'s range
    # so np.interp's right-edge constant doesn't produce a step.
    np.clip(x_new, 0.0, n_in - 1, out=x_new)
    if audio.ndim == 1:
        return np.interp(x_new, x_old, audio).astype(np.float32)
    out = np.empty((n_out, audio.shape[1]), dtype=np.float32)
    for c in range(audio.shape[1]):
        out[:, c] = np.interp(x_new, x_old, audio[:, c]).astype(np.float32)
    return out


def estimate_voice_median_midi(
    audio_mono: np.ndarray, sr: int, fft_size: int = 2048,
) -> float:
    """Pre-pass on the voice file: median MIDI pitch over voiced hops.
    Returns ``VOICE_MEDIAN_FALLBACK_MIDI`` if nothing voiced is detected.

    Uses the live engine's pitch detector so the estimate matches what
    ``Autotuner.process`` will report during the actual run. Sampling every
    ``VOICE_MEDIAN_HOP_STRIDE`` hops keeps this fast on long files without
    materially affecting the median.
    """
    tuner = Autotuner(fft_size, sr)
    hop = tuner.hop
    midis: list[float] = []
    n = int(audio_mono.shape[0])
    end = max(0, n - fft_size + 1)
    step = hop * VOICE_MEDIAN_HOP_STRIDE
    for h_start in range(0, end, step):
        block = audio_mono[h_start:h_start + fft_size]
        windowed = block * tuner.window
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec)
        hz = tuner._detect_pitch_robust(mag)
        if hz > 0.0:
            midis.append(float(freq_to_midi(hz)))
    if not midis:
        return float(VOICE_MEDIAN_FALLBACK_MIDI)
    return float(np.median(midis))


def midi_duration_weighted_median(notes: list[MidiNote]) -> float | None:
    """Median MIDI pitch weighted by note duration. A 5-second sustained
    note pulls the median harder than a 50 ms grace note — better
    representative of "where the song lives" than counting note onsets.
    Returns ``None`` for an empty list.
    """
    if not notes:
        return None
    sorted_notes = sorted(notes, key=lambda n: n.pitch)
    total_dur = sum(max(0.0, n.end - n.start) for n in sorted_notes)
    if total_dur <= 0.0:
        return float(np.median([n.pitch for n in sorted_notes]))
    target = total_dur / 2.0
    cum = 0.0
    for n in sorted_notes:
        cum += max(0.0, n.end - n.start)
        if cum >= target:
            return float(n.pitch)
    return float(sorted_notes[-1].pitch)


# Vocal range used by the "restrict to human vocal range" option. C1..C4
# (MIDI 24..60) — 3 octaves wide so most MIDI ranges fit fully under the
# song-wide shift, avoiding the median-centering fallback that lets
# extremes spill above the upper bound. A per-frame clamp in
# ``_process_channel`` (applied after vibrato) catches any outliers and
# folds them by whole octaves until they're inside the window.
VOCAL_LO_MIDI = 24.0
VOCAL_HI_MIDI = 60.0

# Krumhansl-Schmuckler tonal-hierarchy profiles (1990). Used by
# ``build_key_timeline`` to estimate the active key as a windowed
# correlation between pitch-class durations and rotated profiles.
_KRUMHANSL_MAJOR = np.array(
    [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88],
    dtype=np.float64,
)
_KRUMHANSL_MINOR = np.array(
    [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17],
    dtype=np.float64,
)
_MAJOR_INTERVALS = (0, 2, 4, 5, 7, 9, 11)
_NATURAL_MINOR_INTERVALS = (0, 2, 3, 5, 7, 8, 10)


@dataclass
class KeyTimeline:
    """Per-step (root, mode) key estimate over a song. ``mode = 0`` is
    major, ``mode = 1`` is natural minor. ``lookup(t)`` returns the
    estimate for any time in seconds.
    """
    step_s: float
    roots: np.ndarray  # int8, length == n_steps
    modes: np.ndarray  # int8, length == n_steps

    def lookup(self, t: float) -> tuple[int, int]:
        if self.roots.size == 0:
            return 0, 0
        i = int(t / self.step_s)
        i = max(0, min(self.roots.size - 1, i))
        return int(self.roots[i]), int(self.modes[i])


def build_key_timeline(
    notes: list[MidiNote], end_time: float,
    *, window_s: float = 6.0, step_s: float = 0.5,
) -> KeyTimeline:
    """Krumhansl-Schmuckler windowed key estimation. For each step
    spanning ``[0, end_time]``, correlate the duration-weighted
    pitch-class distribution within the past ``window_s`` seconds
    against the 24 (root × major/minor) profiles and keep the best.
    Empty windows reuse the previous step's key.
    """
    n_steps = max(1, int(np.ceil(max(0.0, end_time) / step_s)) + 1)
    roots = np.zeros(n_steps, dtype=np.int8)
    modes = np.zeros(n_steps, dtype=np.int8)
    if not notes:
        return KeyTimeline(step_s=step_s, roots=roots, modes=modes)

    sorted_notes = sorted(notes, key=lambda n: n.start)
    starts = np.array([n.start for n in sorted_notes], dtype=np.float64)
    ends = np.array([n.end for n in sorted_notes], dtype=np.float64)
    pcs = np.array([n.pitch % 12 for n in sorted_notes], dtype=np.int8)

    for i in range(n_steps):
        t = i * step_s
        win_start = max(0.0, t - window_s)
        win_end = t + step_s
        active_idx = np.where((ends > win_start) & (starts < win_end))[0]
        if active_idx.size == 0:
            if i > 0:
                roots[i], modes[i] = roots[i - 1], modes[i - 1]
            continue
        pc_dur = np.zeros(12, dtype=np.float64)
        for j in active_idx:
            os_ = max(starts[j], win_start)
            oe = min(ends[j], win_end)
            if oe > os_:
                pc_dur[pcs[j]] += oe - os_
        if pc_dur.sum() < 1e-6:
            if i > 0:
                roots[i], modes[i] = roots[i - 1], modes[i - 1]
            continue
        best_score = -np.inf
        best_root, best_mode = 0, 0
        for root in range(12):
            sm = float(np.dot(pc_dur, np.roll(_KRUMHANSL_MAJOR, root)))
            sn = float(np.dot(pc_dur, np.roll(_KRUMHANSL_MINOR, root)))
            if sm > best_score:
                best_score, best_root, best_mode = sm, root, 0
            if sn > best_score:
                best_score, best_root, best_mode = sn, root, 1
        roots[i], modes[i] = best_root, best_mode
    return KeyTimeline(step_s=step_s, roots=roots, modes=modes)


def scale_candidates(
    root: int, mode: int, lo_midi: int = 0, hi_midi: int = 127,
) -> list[int]:
    """All MIDI notes belonging to ``(root, mode)`` inside ``[lo, hi]``."""
    intervals = _MAJOR_INTERVALS if mode == 0 else _NATURAL_MINOR_INTERVALS
    out: list[int] = []
    for octv in range(0, 11):
        base = octv * 12 + (int(root) % 12)
        for ivl in intervals:
            m = base + ivl
            if lo_midi <= m <= hi_midi:
                out.append(m)
    return out


def compute_song_octave_shift(
    voice_median_midi: float,
    midi_median_midi: float | None,
    *,
    restrict_to_vocal_range: bool = False,
    midi_lo_midi: float | None = None,
    midi_hi_midi: float | None = None,
    vocal_lo_midi: float = VOCAL_LO_MIDI,
    vocal_hi_midi: float = VOCAL_HI_MIDI,
) -> float:
    """Whole-octave shift (in semitones, always a multiple of 12) to apply
    to every MIDI target so the song's median pitch lands in the same
    octave as the voice. Preserves the song's contour — every MIDI step is
    mirrored 1:1 in the voice — while keeping the audible range close to
    the singer's natural register.

    With ``restrict_to_vocal_range=True`` and the MIDI's pitch range
    (``midi_lo_midi``, ``midi_hi_midi``) supplied, the shift is further
    constrained to the set of whole-octaves that keep the MIDI's full
    range inside ``[vocal_lo_midi, vocal_hi_midi]``. Among those octaves
    we still pick the one closest to the voice-median alignment, so the
    singer's natural pitch is honoured whenever the constraint allows.
    Returning a single shift preserves the direction-preserving contour
    guarantee — voice always moves in the same direction as the song.
    """
    if midi_median_midi is None:
        return 0.0
    voice_pref_oct = round((voice_median_midi - midi_median_midi) / 12.0)

    if (
        not restrict_to_vocal_range
        or midi_lo_midi is None
        or midi_hi_midi is None
    ):
        return 12.0 * float(voice_pref_oct)

    # Range of shifts that keep [midi_lo + shift, midi_hi + shift] inside
    # the vocal window.
    min_shift = vocal_lo_midi - midi_lo_midi
    max_shift = vocal_hi_midi - midi_hi_midi
    if min_shift > max_shift:
        # MIDI range exceeds vocal range — no shift can hold every note.
        # Center the median on the vocal centre instead.
        center = 0.5 * (vocal_lo_midi + vocal_hi_midi)
        return 12.0 * round((center - midi_median_midi) / 12.0)
    min_oct = int(np.ceil(min_shift / 12.0))
    max_oct = int(np.floor(max_shift / 12.0))
    if min_oct > max_oct:
        # Valid window is narrower than 12 semitones — no whole-octave
        # shift fits cleanly. Pick the one closest to voice preference,
        # accepting that a few notes near the edges may spill out.
        center = 0.5 * (vocal_lo_midi + vocal_hi_midi)
        return 12.0 * round((center - midi_median_midi) / 12.0)
    chosen_oct = max(min_oct, min(max_oct, voice_pref_oct))
    return 12.0 * float(chosen_oct)


def render_overlay_cycle(
    notes: list[MidiNote],
    cycle_end_s: float,
    pause_s: float,
    sr: int,
) -> np.ndarray:
    """Render exactly one loop cycle of overlay audio: ``cycle_end_s``
    seconds of MIDI followed by ``pause_s`` seconds of silence. The result
    can be tiled to match the voice output's length.
    """
    total = max(0.0, cycle_end_s) + max(0.0, pause_s)
    return _render_notes_audio(notes, total, sr, per_note_amplitude=0.16)


def make_room_impulse_response(
    sr: int, length_ms: float = 80.0, taps: int = 240, seed: int = 0xBADCAFE,
) -> np.ndarray:
    """Random-tap room IR. Sparse impulses with exponential decay produce
    a "small room" feel without needing a convolution sample. Deterministic
    seed so the same settings always give the same room.

    The IR is calibrated so that convolving a unit-RMS noise signal with
    it yields a unit-RMS output. That way a ``wet`` knob of 0.05 actually
    adds ~5% RMS of reverb onto the dry signal — the slider feels
    proportional. Without this scaling, summing hundreds of decaying taps
    produces O(sqrt(taps)) RMS gain and small wet values clip immediately.
    """
    n = max(1, int(length_ms / 1000.0 * sr))
    rng = np.random.default_rng(seed)
    ir = np.zeros(n, dtype=np.float32)
    tap_idx = rng.integers(0, n, size=taps)
    tap_signs = rng.choice([-1.0, 1.0], size=taps).astype(np.float32)
    for idx, sign in zip(tap_idx, tap_signs):
        ir[idx] += sign
    decay = np.exp(np.linspace(0.0, -6.91, n)).astype(np.float32)  # -60 dB tail
    ir *= decay
    # Calibrate: white-noise probe through the IR, normalise so output RMS
    # matches input RMS. Cheap and deterministic given the fixed seed.
    probe = rng.standard_normal(min(sr, 4 * n)).astype(np.float32)
    probe_out = np.convolve(probe, ir, mode="same")
    in_rms = float(np.sqrt(np.mean(probe * probe)))
    out_rms = float(np.sqrt(np.mean(probe_out * probe_out)))
    if out_rms > 1e-9:
        ir *= in_rms / out_rms
    return ir


def _fft_convolve_overlap_add(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Overlap-add FFT convolution, returns the causal portion of length len(x).
    Used for reverb because direct ``np.convolve`` is O(n*m) which is slow for
    multi-minute audio with a few-thousand-sample IR.
    """
    n = int(x.shape[0])
    m = int(h.shape[0])
    if n == 0 or m == 0:
        return np.zeros(n, dtype=np.float32)
    L = max(2 * m, 4096)
    L = 1 << int(np.ceil(np.log2(L)))
    chunk = L - m + 1
    H = np.fft.rfft(h.astype(np.float32), L)
    out_full_len = n + m - 1
    out = np.zeros(out_full_len, dtype=np.float32)
    pos = 0
    while pos < n:
        seg = x[pos:pos + chunk]
        if seg.shape[0] < chunk:
            pad = np.zeros(chunk - seg.shape[0], dtype=np.float32)
            seg = np.concatenate([seg, pad])
        Y = np.fft.rfft(seg, L) * H
        y = np.fft.irfft(Y, L).astype(np.float32)
        end = min(pos + L, out_full_len)
        out[pos:end] += y[:end - pos]
        pos += chunk
    return out[:n]


def apply_reverb(audio: np.ndarray, sr: int, wet: float, length_ms: float = 80.0) -> np.ndarray:
    """Add ``wet * convolve(audio, IR)`` to ``audio``. ``wet`` is 0..1 (a
    convenience scale; even 0.05 is audible because the IR is normalised
    to unit RMS). Mono and stereo handled — stereo gets the same IR per
    channel (centred reverb).
    """
    if wet <= 0.0:
        return audio
    ir = make_room_impulse_response(sr, length_ms)
    if audio.ndim == 1:
        wet_audio = _fft_convolve_overlap_add(audio.astype(np.float32, copy=False), ir)
        return audio + wet * wet_audio
    out = np.empty_like(audio)
    for c in range(audio.shape[1]):
        col = np.ascontiguousarray(audio[:, c], dtype=np.float32)
        wet_audio = _fft_convolve_overlap_add(col, ir)
        out[:, c] = col + wet * wet_audio
    return out


def build_volume_envelope_cycle(
    notes: list[MidiNote],
    cycle_end_s: float,
    pause_s: float,
    sr: int,
    *,
    onset_db: float = 6.0,
    change_db: float = 3.0,
    decay_tau_s: float = 0.12,
    gap_threshold_s: float = 0.05,
) -> np.ndarray:
    """Multiplicative volume envelope, length ``(cycle_end_s + pause_s) * sr``.

    Spikes to ``10**(onset_db/20)`` at notes that follow more than
    ``gap_threshold_s`` of silence (a "fresh attack"), to ``10**(change_db/20)``
    at note-to-note pitch changes without a gap (an articulation kick), and
    decays exponentially back to 1.0 with time constant ``decay_tau_s``.
    Same MIDI note repeated back-to-back gets no boost.

    Tiled across the voice output length, this gives the voice the
    "leans into the new note" feel of a singer articulating each phrase.
    """
    n = max(1, int((max(0.0, cycle_end_s) + max(0.0, pause_s)) * sr))
    env = np.ones(n, dtype=np.float32)
    if not notes:
        return env

    onset_gain = float(10.0 ** (onset_db / 20.0))
    change_gain = float(10.0 ** (change_db / 20.0))

    # Pre-compute the decay shape: e^(-t/tau) for ~5 tau (drops to ~0.7%).
    decay_n = max(1, int(5.0 * decay_tau_s * sr))
    decay_curve = np.exp(
        -np.arange(decay_n, dtype=np.float32) / float(decay_tau_s * sr)
    )

    sorted_notes = sorted(notes, key=lambda n: n.start)
    prev: MidiNote | None = None
    for note in sorted_notes:
        idx = int(note.start * sr)
        if idx >= n:
            break
        if prev is None or (note.start - prev.end) > gap_threshold_s:
            gain = onset_gain
        elif note.pitch != prev.pitch:
            gain = change_gain
        else:
            gain = 1.0  # repeated identical pitch with no gap — no boost
        excess = gain - 1.0
        if excess > 0.0:
            end = min(idx + decay_n, n)
            seg = env[idx:end]
            new_seg = 1.0 + excess * decay_curve[: end - idx]
            np.maximum(seg, new_seg, out=seg)
        prev = note

    return env


# ---------- Audio I/O -----------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (samples, sr). Mono → shape (n,); stereo+ → shape (n, channels).
    Always float32 in [-1, 1]. Falls back to pydub for any extension that
    isn't a libsndfile native (e.g. .mp3, .m4a, .mp4, .mov, .mkv). Video
    files are accepted — pydub asks ffmpeg to extract the first audio
    track and drops the video, so the rest of the pipeline doesn't have
    to know or care that the source was a video.
    """
    p = Path(path)
    ext = p.suffix.lower()
    if ext in _SOUNDFILE_EXTS:
        data, sr = sf.read(str(p), always_2d=False, dtype="float32")
        return np.ascontiguousarray(data), int(sr)

    try:
        from pydub import AudioSegment
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            f"Reading {ext} files requires pydub. Install with: pip install pydub"
        ) from e

    try:
        seg = AudioSegment.from_file(str(p))
    except Exception as e:
        raise RuntimeError(
            f"Could not decode {p.name} via pydub. Audio + video formats "
            f"(.mp3 / .m4a / .aac / .mp4 / .mov / .mkv) need ffmpeg on PATH. "
            f"Underlying error: {e}"
        ) from e

    sr = int(seg.frame_rate)
    width = seg.sample_width
    samples = np.array(seg.get_array_of_samples())
    if width == 1:
        samples = (samples.astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        samples = samples.astype(np.float32) / 32768.0
    elif width == 4:
        samples = samples.astype(np.float32) / 2147483648.0
    else:  # pragma: no cover
        samples = samples.astype(np.float32) / float(2 ** (8 * width - 1))
    if seg.channels > 1:
        samples = samples.reshape(-1, seg.channels)
    return np.ascontiguousarray(samples), sr


def save_audio(path: str, data: np.ndarray, sr: int) -> None:
    p = Path(path)
    ext = p.suffix.lower()
    p.parent.mkdir(parents=True, exist_ok=True)

    if ext in _SOUNDFILE_EXTS:
        sf.write(str(p), data, sr)
        return

    if ext == ".mp3":
        try:
            from pydub import AudioSegment
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "Writing .mp3 needs pydub. Install with: pip install pydub"
            ) from e
        if data.ndim == 1:
            channels = 1
            interleaved = data
        else:
            channels = int(data.shape[1])
            interleaved = data.reshape(-1)
        i16 = np.clip(interleaved * 32767.0, -32768, 32767).astype("<i2")
        try:
            seg = AudioSegment(
                data=i16.tobytes(),
                sample_width=2,
                frame_rate=int(sr),
                channels=channels,
            )
            seg.export(str(p), format="mp3")
        except Exception as e:
            raise RuntimeError(
                f"MP3 export failed; ffmpeg must be on PATH. Underlying error: {e}"
            ) from e
        return

    raise ValueError(f"Unsupported output extension: {ext}")


def default_output_path(voice_path: str, ext: str = ".wav") -> Path:
    """``~/Documents/wondrous/<voice_stem>_tuned.<ext>``. Creates the
    folder lazily — the actual mkdir happens on save."""
    voice_stem = Path(voice_path).stem if voice_path else "output"
    return Path.home() / "Documents" / "wondrous" / f"{voice_stem}_tuned{ext}"


# ---------- Processing ----------------------------------------------------

def process_song(
    voice_path: str,
    midi_path: str,
    track_option: TrackOption,
    output_path: str,
    *,
    overlay_notes: list[MidiNote] | None = None,
    overlay_volume_pct: float = 45.0,
    full_song_end: float | None = None,
    volume_shaping: bool = False,
    dry_wet_pct: float = 100.0,
    vibrato_enabled: bool = False,
    vibrato_rate_hz: float = 5.0,
    vibrato_depth_cents: float = 25.0,
    portamento_speed: float = SONG_RETUNE_SPEED,
    formant_preserve: bool = False,
    reverb_enabled: bool = False,
    reverb_wet_pct: float = 5.0,
    prepitch_speed_enabled: bool = False,
    prepitch_speed_pct: float = 100.0,
    restrict_to_vocal_range: bool = False,
    key_centered_enabled: bool = False,
    lower_midi_octave: bool = False,
    fft_size: int = 2048,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Read voice + MIDI, repitch voice channel-by-channel to follow MIDI,
    write the result to ``output_path``. Voice file on disk is unchanged.

    The loop cycle is ``cycle_end + LOOP_PAUSE_S`` where ``cycle_end`` is
    ``full_song_end`` if given, otherwise the chosen track's last-note time.
    Using ``full_song_end`` keeps the loop tied to the whole MIDI file even
    when the chosen instrument's part stops earlier — without it the voice
    would re-pitch over the (shorter) track range and fall out of sync with
    the song.

    If ``overlay_notes`` is non-None, those notes are rendered as additive
    sines and mixed into the output, looped on the same cycle as the pitch
    target.

    Output sample rate matches the voice file (no resampling — the user
    explicitly asked to avoid any pitch/speed side-effects).
    """
    voice, sr = load_audio(voice_path)
    voice_mono_orig = (
        voice if voice.ndim == 1 else voice.mean(axis=1).astype(np.float32, copy=False)
    )

    # Pre-shift: drop every MIDI pitch by an octave before any of the
    # downstream pitch math (median alignment, vocal-range bounds, key
    # detection, per-frame target lookup, volume-envelope timing, overlay
    # rendering). Useful when the chosen track sits in a register that
    # would force the song-wide shift to land the output uncomfortably
    # high. Note start/end times are unchanged, so loop length and beat
    # alignment stay intact.
    if lower_midi_octave:
        track_option = TrackOption(
            track_idx=track_option.track_idx,
            channel=track_option.channel,
            track_name=track_option.track_name,
            notes=[
                MidiNote(n.start, n.end, n.pitch - 12)
                for n in track_option.notes
            ],
        )
        if overlay_notes is not None:
            overlay_notes = [
                MidiNote(n.start, n.end, n.pitch - 12)
                for n in overlay_notes
            ]

    # Pre-pass: estimate voice's median pitch, then choose a single
    # whole-octave shift that aligns the song's range with the voice. All
    # MIDI targets get this fixed shift, so the voice mirrors the song's
    # melodic contour exactly (up moves up, down moves down) and stays
    # close to the singer's natural register without per-note octave
    # jumps. Run on the *original* voice so the slowdown below doesn't
    # offset the alignment by an extra octave.
    voice_median_midi = estimate_voice_median_midi(voice_mono_orig, sr, fft_size)
    midi_median_midi = midi_duration_weighted_median(track_option.notes)
    midi_lo_midi: float | None = None
    midi_hi_midi: float | None = None
    if track_option.notes:
        pitches = [n.pitch for n in track_option.notes]
        midi_lo_midi = float(min(pitches))
        midi_hi_midi = float(max(pitches))
    shift_semitones = compute_song_octave_shift(
        voice_median_midi,
        midi_median_midi,
        restrict_to_vocal_range=restrict_to_vocal_range,
        midi_lo_midi=midi_lo_midi,
        midi_hi_midi=midi_hi_midi,
    )
    target_freq_factor = float(2.0 ** (shift_semitones / 12.0))

    # Prepitch slowdown: stretch the voice via linear interpolation. Time
    # extends (output ends up longer) and the fundamental drops by the
    # same factor — the autotune step naturally compensates back to
    # ``target_hz`` because we feed it the expected slowed-pitch contour
    # via ``voice_hz_override``. Detection of the slowed pitch directly
    # would fail (well below the autotuner's vocal-range mask), so we
    # pre-detect on the original voice and translate per-hop.
    #
    # The phase-vocoder's spectral resolution is fft_size/sr Hz per bin.
    # At the default 2048/44100 that's ~22 Hz, which is too coarse when
    # the slowed pitch falls into the bin-width range (e.g. 55 Hz at 50%
    # slowdown sits ±0.5 bins from the nearest integer, which becomes a
    # multi-semitone offset after the autotune ratio shift). We grow the
    # FFT proportional to the inverse speed factor so the slowed pitch
    # always has enough bin resolution.
    pitch_contour: np.ndarray | None = None
    speed_factor = 1.0
    effective_fft_size = fft_size
    prepitch_active = (
        prepitch_speed_enabled and 0.0 < prepitch_speed_pct < 100.0
    )
    if prepitch_active:
        speed_factor = 100.0 / float(prepitch_speed_pct)

    # Bump the analysis FFT size for any large pitch shift, in either
    # direction. The PV's bin-interpolation + cepstral-envelope quality
    # both improve with finer bin resolution; for shifts ≳ ±6 semitones
    # the default 2048 leaves audible thinning / distortion. We size the
    # FFT roughly proportional to the *largest* magnitude shift among
    # speed_factor (prepitch slowdown), target_freq_factor (song-wide
    # alignment), or its inverse (downshifts). Capped at 8192 so latency
    # stays bounded — beyond that, larger FFTs don't help much because
    # the bandwidth limits become fundamental.
    shift_magnitude = max(
        speed_factor,
        float(target_freq_factor),
        1.0 / max(float(target_freq_factor), 1e-9),
    )
    if shift_magnitude > 1.2:  # ≥ ~+3 semitones
        target_fft = int(np.ceil(fft_size * shift_magnitude * 2))
        effective_fft_size = min(
            8192, 1 << int(np.ceil(np.log2(max(target_fft, fft_size))))
        )

    # Pitch contour pre-pass: needed by prepitch (to override slowed-pitch
    # detection) AND by key-centered mode (to know what to snap each frame).
    if prepitch_active or key_centered_enabled:
        pitch_contour = estimate_voice_pitch_contour(voice_mono_orig, sr, fft_size)
    if prepitch_active:
        voice = _linear_resample(voice, speed_factor)

    if voice.ndim == 1:
        channels = [voice.astype(np.float32, copy=False)]
    else:
        channels = [
            np.ascontiguousarray(voice[:, c], dtype=np.float32)
            for c in range(voice.shape[1])
        ]

    timeline = MidiTimeline(track_option.notes)
    if full_song_end is not None and full_song_end > 0.0:
        cycle_end = float(full_song_end)
    else:
        cycle_end = float(track_option.end_time)
    cycle_length = cycle_end + LOOP_PAUSE_S if cycle_end > 0.0 else 0.0

    # Key timeline: per-step Krumhansl-Schmuckler estimate for key-centered mode.
    key_timeline: KeyTimeline | None = None
    if key_centered_enabled and track_option.notes:
        key_timeline = build_key_timeline(
            track_option.notes,
            end_time=cycle_end if cycle_end > 0 else track_option.end_time,
        )

    n_channels = len(channels)
    out_channels: list[np.ndarray] = []
    for c_idx, ch_audio in enumerate(channels):
        out_ch = _process_channel(
            ch_audio, sr, effective_fft_size, timeline, cycle_end, cycle_length,
            target_freq_factor=target_freq_factor,
            vibrato_enabled=vibrato_enabled,
            vibrato_rate_hz=float(vibrato_rate_hz),
            vibrato_depth_cents=float(vibrato_depth_cents),
            portamento_speed=float(portamento_speed),
            formant_preserve=bool(formant_preserve),
            speed_factor=float(speed_factor),
            original_pitch_contour=pitch_contour,
            key_centered_enabled=bool(key_centered_enabled),
            key_timeline=key_timeline,
            restrict_to_vocal_range=bool(restrict_to_vocal_range),
            lower_midi_octave=bool(lower_midi_octave),
            progress_callback=(
                lambda f, c=c_idx: (
                    progress_callback((c + f) / n_channels)
                    if progress_callback else None
                )
            ),
            cancel_event=cancel_event,
        )
        if cancel_event is not None and cancel_event.is_set():
            return
        out_channels.append(out_ch)

    if len(out_channels) == 1:
        out = out_channels[0]
    else:
        n = min(c.shape[0] for c in out_channels)
        out = np.stack([c[:n] for c in out_channels], axis=1)

    if volume_shaping and cycle_length > 0.0:
        env_cycle = build_volume_envelope_cycle(
            track_option.notes, cycle_end, LOOP_PAUSE_S, sr,
        )
        n_env = int(env_cycle.shape[0])
        n_out = int(out.shape[0])
        if n_env > 0 and n_out > 0:
            n_repeats = (n_out + n_env - 1) // n_env
            tiled_env = np.tile(env_cycle, n_repeats)[:n_out]
            if out.ndim == 1:
                out = out * tiled_env
            else:
                out = out * tiled_env[:, None]

    # Dry/wet mix with the original voice. Length-aligned to ``out``.
    wet = max(0.0, min(1.0, dry_wet_pct / 100.0))
    if wet < 1.0:
        n_align = min(int(out.shape[0]), int(voice.shape[0]))
        if voice.ndim == 1 and out.ndim == 1:
            out_dry = voice[:n_align].astype(np.float32, copy=False)
            out[:n_align] = wet * out[:n_align] + (1.0 - wet) * out_dry
        elif voice.ndim == 2 and out.ndim == 2:
            n_ch = min(out.shape[1], voice.shape[1])
            out[:n_align, :n_ch] = (
                wet * out[:n_align, :n_ch]
                + (1.0 - wet) * voice[:n_align, :n_ch].astype(np.float32, copy=False)
            )
        # Mismatched dim shapes (mono voice → stereo? not currently produced) skip dry mix.

    if reverb_enabled and reverb_wet_pct > 0.0:
        out = apply_reverb(out, sr, wet=float(reverb_wet_pct) / 100.0)

    if overlay_notes is not None and cycle_length > 0.0:
        cycle_audio = render_overlay_cycle(
            overlay_notes, cycle_end, LOOP_PAUSE_S, sr,
        )
        n_cycle = int(cycle_audio.shape[0])
        n_out = int(out.shape[0])
        if n_cycle > 0 and n_out > 0:
            n_repeats = (n_out + n_cycle - 1) // n_cycle
            tiled = np.tile(cycle_audio, n_repeats)[:n_out]
            overlay_gain = max(0.0, min(1.0, overlay_volume_pct / 100.0))
            if out.ndim == 1:
                out = out + overlay_gain * tiled
            else:
                out = out + overlay_gain * tiled[:, None]

    # Single saturation step at the very end so volume-shaping boosts and
    # overlay both get tamed without stacking clip operations.
    np.clip(out, -1.0, 1.0, out=out)

    save_audio(output_path, out, sr)


def _process_channel(
    audio: np.ndarray,
    sr: int,
    fft_size: int,
    timeline: MidiTimeline,
    cycle_end_s: float,
    cycle_length_s: float,
    *,
    target_freq_factor: float = 1.0,
    vibrato_enabled: bool = False,
    vibrato_rate_hz: float = 5.0,
    vibrato_depth_cents: float = 25.0,
    portamento_speed: float = SONG_RETUNE_SPEED,
    formant_preserve: bool = False,
    speed_factor: float = 1.0,
    original_pitch_contour: np.ndarray | None = None,
    key_centered_enabled: bool = False,
    key_timeline: KeyTimeline | None = None,
    restrict_to_vocal_range: bool = False,
    lower_midi_octave: bool = False,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> np.ndarray:
    tuner = Autotuner(fft_size, sr)
    hop = tuner.hop
    n_orig = int(audio.shape[0])

    # Pre-pend a warmup of zeros equal to the algorithm's inherent latency
    # (FFT_SIZE - hop), then post-pend whatever's needed to round to a
    # whole hop. Output samples 0..warmup are the warm-up and get sliced off.
    warmup = fft_size - hop
    n_total = warmup + n_orig
    n_padded = ((n_total + hop - 1) // hop) * hop
    audio_padded = np.concatenate(
        [
            np.zeros(warmup, dtype=np.float32),
            np.ascontiguousarray(audio, dtype=np.float32),
            np.zeros(n_padded - n_total, dtype=np.float32),
        ]
    )

    out = np.zeros(n_padded, dtype=np.float32)
    n_hops = n_padded // hop

    # State for key-centered mode: latched snap target (with 30-cent
    # hysteresis), the t_orig at which it last changed (for vibrato
    # ramp), and the (root, mode) it was chosen against (resets the
    # snap when the song modulates). Lives across iterations so the
    # hysteresis state persists.
    _key_state: dict = {
        "snap": None,        # last committed snap MIDI (float | None)
        "snap_start": 0.0,   # t_orig when 'snap' was committed
        "root": -1,
        "mode": -1,
    }

    for h in range(n_hops):
        if cancel_event is not None and (h & 0x3F) == 0 and cancel_event.is_set():
            return out[warmup:warmup + n_orig]
        block = audio_padded[h * hop : (h + 1) * hop]

        # Time in the *original* voice timeline corresponding to the centre
        # of this analysis hop. Use it to look up the MIDI target.
        t_orig = max(0.0, (h * hop - warmup)) / float(sr)
        # Look up the voice's natural-pitch hz from the contour pre-pass
        # if one was built. Used both by the prepitch slowdown
        # (voice_hz_override) and by key-centered mode (snap input).
        voice_natural_hz = 0.0
        if (
            original_pitch_contour is not None
            and original_pitch_contour.size > 0
        ):
            orig_h_float = (h * hop - warmup) / (max(1.0, speed_factor) * hop)
            orig_h = int(round(orig_h_float))
            if 0 <= orig_h < original_pitch_contour.size:
                voice_natural_hz = float(original_pitch_contour[orig_h])

        active_note: MidiNote | None = None
        if cycle_length_s > 0.0:
            t_midi = t_orig % cycle_length_s
            if t_midi >= cycle_end_s:
                # Silence ramp during the half-second pause.
                target_hz = 0.0
            elif key_centered_enabled and key_timeline is not None:
                # Snap voice's natural pitch to the nearest scale note in
                # the currently estimated key. Hysteresis keeps the snap
                # stable while voice's pitch wobbles around a borderline.
                if voice_natural_hz > 0.0:
                    voice_midi = float(freq_to_midi(voice_natural_hz))
                    root, mode = key_timeline.lookup(t_midi)
                    if root != _key_state["root"] or mode != _key_state["mode"]:
                        _key_state["snap"] = None
                        _key_state["root"] = root
                        _key_state["mode"] = mode
                    lo = int(VOCAL_LO_MIDI) if restrict_to_vocal_range else 0
                    hi = int(VOCAL_HI_MIDI) if restrict_to_vocal_range else 127
                    cands = scale_candidates(root, mode, lo, hi)
                    if cands:
                        nearest = float(min(cands, key=lambda m: abs(m - voice_midi)))
                        prev_snap = _key_state["snap"]
                        if prev_snap is None:
                            _key_state["snap"] = nearest
                            _key_state["snap_start"] = t_orig
                        elif nearest != prev_snap:
                            cents_to_prev = 100.0 * abs(voice_midi - prev_snap)
                            cents_to_new = 100.0 * abs(voice_midi - nearest)
                            if cents_to_prev - cents_to_new >= 30.0:
                                _key_state["snap"] = nearest
                                _key_state["snap_start"] = t_orig
                        target_hz = float(midi_to_freq(_key_state["snap"]))
                    else:
                        target_hz = 0.0
                else:
                    target_hz = 0.0
            else:
                tgt, active_note = timeline.target_at(t_midi)
                target_hz = 0.0 if tgt is None else float(tgt) * target_freq_factor
        else:
            t_midi = 0.0
            target_hz = 0.0

        # Vibrato: per-note ramp on depth so onsets stay crisp, then
        # continuous sine LFO. Phase uses ``t_orig`` (monotonically
        # increasing through the whole voice) so the LFO doesn't reset
        # at each loop-cycle wrap and click. In key-centered mode the
        # "note start" is the moment the snap target last changed; in
        # melody mode it's the active MidiNote's start.
        if (
            vibrato_enabled
            and target_hz > 0.0
            and vibrato_depth_cents > 0.0
        ):
            if key_centered_enabled and _key_state["snap"] is not None:
                elapsed = max(0.0, t_orig - float(_key_state["snap_start"]))
                vib_ok = True
            elif active_note is not None:
                elapsed = max(0.0, t_midi - active_note.start)
                vib_ok = True
            else:
                elapsed = 0.0
                vib_ok = False
            if vib_ok:
                ramp = max(0.0, min(1.0, (elapsed - 0.05) / 0.20))
                lfo = float(np.sin(2.0 * np.pi * vibrato_rate_hz * t_orig))
                cents_delta = ramp * vibrato_depth_cents * lfo
                target_hz *= float(2.0 ** (cents_delta / 1200.0))

        # When the voice has been time-stretched (speed_factor > 1), the
        # autotuner's own detector won't see the slowed fundamental
        # (it's below the vocal-range mask). Translate the original
        # pitch contour onto the slowed timeline and feed it as
        # ``voice_hz_override``: the slowed pitch is the original times
        # 1/speed_factor.
        voice_hz_override: float | None = None
        if speed_factor > 1.000001 and voice_natural_hz > 0.0:
            voice_hz_override = voice_natural_hz / speed_factor

        # Lower MIDI by an octave: halve target_hz so the output drops 12
        # semitones from whatever the alignment / snap produced. Applied
        # *after* both modes' targets are computed and *before* the
        # vocal-range clamp, so the clamp can still fold the lowered
        # target back up if it would push below C1. The source notes
        # were already pre-shifted in process_song so the overlay
        # rendering, key detection, and median calculations all see the
        # lowered MIDI; this step is the additional bypass that stops
        # the song-wide alignment from compensating away the lowering.
        if lower_midi_octave and target_hz > 0.0:
            target_hz *= 0.5

        # Strict per-frame vocal-range enforcement. Catches notes that
        # the song-wide shift's centering fallback couldn't fit, and any
        # outlier notes when the MIDI's range exceeds the vocal window.
        # Folds them by whole octaves until they're inside, so the upper
        # and lower bounds are hard ceilings/floors. The 0.6-semitone
        # tolerance lets vibrato (depth ≤50 cents) and pitch glide
        # through the boundary without immediately snapping back.
        if restrict_to_vocal_range and target_hz > 0.0:
            target_midi_clamp = float(freq_to_midi(target_hz))
            while target_midi_clamp > VOCAL_HI_MIDI + 0.6:
                target_midi_clamp -= 12.0
            while target_midi_clamp < VOCAL_LO_MIDI - 0.6:
                target_midi_clamp += 12.0
            target_hz = float(midi_to_freq(target_midi_clamp))

        out_block, _ = tuner.process(
            block,
            key=0,
            scale="Chromatic",
            strength=100.0,
            retune_speed=portamento_speed,
            target_hz_override=target_hz,
            voice_hz_override=voice_hz_override,
            formant_preserve=formant_preserve,
        )
        out[h * hop : (h + 1) * hop] = out_block

        if progress_callback is not None and (h & 0x1F) == 0:
            progress_callback(h / max(1, n_hops))

    if progress_callback is not None:
        progress_callback(1.0)

    return out[warmup : warmup + n_orig]
