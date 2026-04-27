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
from .scales import midi_to_freq, midi_to_name


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


# ---------- MIDI ----------------------------------------------------------

@dataclass
class MidiNote:
    start: float  # seconds
    end: float    # seconds
    pitch: int    # MIDI note number


@dataclass
class TrackOption:
    """One selectable (track, channel) pair from a MIDI file."""
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
    def label(self) -> str:
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
    return overall_end, options


class MidiTimeline:
    """Fast lookup of "the highest active MIDI pitch at time t" for a track.

    Used both to decide the per-frame target during voice processing and to
    drive the silence/wrap-pause ramp envelope: when no note is active, this
    returns ``None`` and the caller treats that as "no shift".
    """

    def __init__(self, notes: list[MidiNote]) -> None:
        notes = sorted(notes, key=lambda n: n.start)
        self.starts = np.array([n.start for n in notes], dtype=np.float64)
        self.ends = np.array([n.end for n in notes], dtype=np.float64)
        self.pitches = np.array([n.pitch for n in notes], dtype=np.int32)

    def target_hz_at(self, t: float) -> float | None:
        if self.starts.size == 0:
            return None
        i = int(np.searchsorted(self.starts, t, side="right"))
        if i == 0:
            return None
        # Among notes with start <= t, keep those still ringing.
        active = self.ends[:i] > t
        if not active.any():
            return None
        active_pitches = self.pitches[:i][active]
        return float(midi_to_freq(float(active_pitches.max())))


def render_midi_preview(option: TrackOption, sr: int = _PREVIEW_SR) -> np.ndarray:
    """Render a track/channel as additive sine tones for the preview player.

    Polyphonic content is rendered as actual chords (notes sum) rather than
    just the highest pitch — the user uses this to identify the right
    track, so they should hear what's actually there. Light tanh saturation
    keeps overlapping sines from clipping.
    """
    if not option.notes:
        return np.zeros(0, dtype=np.float32)
    end = option.end_time + 0.25  # short tail of silence
    n = max(1, int(end * sr))
    audio = np.zeros(n, dtype=np.float32)

    attack_n = max(1, int(_PREVIEW_ATTACK_S * sr))
    release_n = max(1, int(_PREVIEW_RELEASE_S * sr))

    for note in option.notes:
        freq = float(midi_to_freq(float(note.pitch)))
        i0 = int(note.start * sr)
        i1 = int(note.end * sr)
        if i1 <= i0 or i0 >= n:
            continue
        i1 = min(i1, n)
        nn = i1 - i0
        t = np.arange(nn, dtype=np.float32) / sr
        sig = 0.25 * np.sin(2.0 * np.pi * freq * t).astype(np.float32)
        env = np.ones(nn, dtype=np.float32)
        a = min(attack_n, nn // 2)
        r = min(release_n, nn // 2)
        if a > 0:
            env[:a] = np.linspace(0.0, 1.0, a, dtype=np.float32)
        if r > 0:
            env[-r:] = np.linspace(1.0, 0.0, r, dtype=np.float32)
        audio[i0:i1] += sig * env

    return np.tanh(audio).astype(np.float32)


# ---------- Audio I/O -----------------------------------------------------

def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Return (samples, sr). Mono → shape (n,); stereo+ → shape (n, channels).
    Always float32 in [-1, 1]. Falls back to pydub for any extension that
    isn't a libsndfile native (e.g. .mp3, .m4a).
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
            f"Could not decode {p.name} via pydub. For .mp3 / .m4a / .aac you "
            f"need ffmpeg on PATH. Underlying error: {e}"
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
    fft_size: int = 2048,
    progress_callback=None,
    cancel_event: threading.Event | None = None,
) -> None:
    """Read voice + MIDI, repitch voice channel-by-channel to follow MIDI,
    write the result to ``output_path``. Voice file on disk is unchanged.

    Sample rate of the output matches the voice file (no resampling — the
    user explicitly asked to avoid any pitch/speed side-effects).
    """
    voice, sr = load_audio(voice_path)
    if voice.ndim == 1:
        channels = [voice.astype(np.float32, copy=False)]
    else:
        channels = [
            np.ascontiguousarray(voice[:, c], dtype=np.float32)
            for c in range(voice.shape[1])
        ]

    timeline = MidiTimeline(track_option.notes)
    midi_end = float(track_option.end_time)
    cycle_length = midi_end + LOOP_PAUSE_S if midi_end > 0 else 0.0

    n_channels = len(channels)
    out_channels: list[np.ndarray] = []
    for c_idx, ch_audio in enumerate(channels):
        out_ch = _process_channel(
            ch_audio, sr, fft_size, timeline, midi_end, cycle_length,
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

    save_audio(output_path, out, sr)


def _process_channel(
    audio: np.ndarray,
    sr: int,
    fft_size: int,
    timeline: MidiTimeline,
    midi_end: float,
    cycle_length: float,
    *,
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

    for h in range(n_hops):
        if cancel_event is not None and (h & 0x3F) == 0 and cancel_event.is_set():
            return out[warmup:warmup + n_orig]
        block = audio_padded[h * hop : (h + 1) * hop]

        # Time in the *original* voice timeline corresponding to the centre
        # of this analysis hop. Use it to look up the MIDI target.
        t_orig = max(0.0, (h * hop - warmup)) / float(sr)
        if cycle_length > 0.0:
            t_midi = t_orig % cycle_length
            if t_midi >= midi_end:
                target_hz = 0.0  # silence ramp during the half-second pause
            else:
                tgt = timeline.target_hz_at(t_midi)
                target_hz = 0.0 if tgt is None else float(tgt)
        else:
            target_hz = 0.0

        out_block, _ = tuner.process(
            block,
            key=0,
            scale="Chromatic",
            strength=100.0,
            retune_speed=SONG_RETUNE_SPEED,
            target_hz_override=target_hz,
        )
        out[h * hop : (h + 1) * hop] = out_block

        if progress_callback is not None and (h & 0x1F) == 0:
            progress_callback(h / max(1, n_hops))

    if progress_callback is not None:
        progress_callback(1.0)

    return out[warmup : warmup + n_orig]
