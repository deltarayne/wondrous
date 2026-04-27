"""Audio I/O engine. Wraps a sounddevice.Stream around the Autotuner with
ring buffers so the driver's native blocksize is decoupled from the FFT hop.
"""
from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
import sounddevice as sd

from .dsp import Autotuner
from .params import Params


@dataclass
class DeviceInfo:
    index: int
    name: str
    hostapi: str
    channels: int
    default_samplerate: float

    @property
    def label(self) -> str:
        return f"[{self.index}] {self.name} ({self.hostapi})"


def list_input_devices() -> list[DeviceInfo]:
    return _list_devices(input=True)


def list_output_devices() -> list[DeviceInfo]:
    return _list_devices(input=False)


def _list_devices(input: bool) -> list[DeviceInfo]:
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    out: list[DeviceInfo] = []
    for i, d in enumerate(devices):
        ch = d["max_input_channels"] if input else d["max_output_channels"]
        if ch <= 0:
            continue
        out.append(
            DeviceInfo(
                index=i,
                name=d["name"],
                hostapi=hostapis[d["hostapi"]]["name"],
                channels=int(ch),
                default_samplerate=float(d["default_samplerate"] or 0.0),
            )
        )
    return out


class _Ring:
    """Single-producer/single-consumer float32 ring buffer."""

    def __init__(self, capacity: int) -> None:
        self.cap = int(capacity)
        self.buf = np.zeros(self.cap, dtype=np.float32)
        self.read = 0
        self.write = 0
        self.fill = 0

    def write_chunk(self, data: np.ndarray) -> None:
        n = data.shape[0]
        if n == 0:
            return
        end = self.write + n
        if end <= self.cap:
            self.buf[self.write : end] = data
        else:
            split = self.cap - self.write
            self.buf[self.write :] = data[:split]
            self.buf[: end - self.cap] = data[split:]
        self.write = end % self.cap
        self.fill = min(self.cap, self.fill + n)

    def read_chunk(self, n: int, out: np.ndarray) -> int:
        """Read up to n samples into `out`. Returns number actually read."""
        take = min(n, self.fill)
        if take == 0:
            return 0
        end = self.read + take
        if end <= self.cap:
            out[:take] = self.buf[self.read : end]
        else:
            split = self.cap - self.read
            out[:split] = self.buf[self.read :]
            out[split:take] = self.buf[: end - self.cap]
        self.read = end % self.cap
        self.fill -= take
        return take

    def read_into(self, dst: np.ndarray, count: int) -> None:
        """Read exactly `count` samples; pads with zeros if not enough."""
        got = self.read_chunk(count, dst)
        if got < count:
            dst[got:count] = 0.0


class Engine:
    """Audio I/O + DSP. Engine attributes (`in_level`, `out_level`,
    `detected_hz`, `last_status`) are written from the audio callback thread
    and read by the GUI thread; primitive reads under the GIL are atomic.
    """

    def __init__(self, params: Params) -> None:
        self.params = params
        self._stream: sd.Stream | None = None
        self._tuner: Autotuner | None = None
        self._start_lock = threading.Lock()

        self.in_level: float = 0.0
        self.out_level: float = 0.0
        self.detected_hz: float = 0.0
        self.last_status: str = ""
        self.dropout_count: int = 0

    @property
    def is_running(self) -> bool:
        return self._stream is not None and self._stream.active

    def start(self) -> None:
        with self._start_lock:
            self.stop()
            p = self.params
            tuner = Autotuner(p.fft_size, p.samplerate)
            hop = tuner.hop

            # Ring buffers sized to comfortably absorb driver block sizes up
            # to ~100 ms while feeding the autotuner in hop-sized chunks.
            ring_cap = max(p.fft_size * 4, int(p.samplerate * 0.2))
            in_ring = _Ring(ring_cap)
            out_ring = _Ring(ring_cap)

            # Pre-prime the output ring so the first audio callbacks always
            # have data: the algorithm has FFT_SIZE-hop samples of inherent
            # latency, so prime that much silence.
            prime = np.zeros(p.fft_size - hop, dtype=np.float32)
            out_ring.write_chunk(prime)

            scratch_in = np.empty(hop, dtype=np.float32)

            def callback(indata, outdata, frames, time, status):  # noqa: ANN001
                if status:
                    self.last_status = str(status)
                    if status.input_underflow or status.output_underflow:
                        self.dropout_count += 1

                # Down-mix to mono and apply input gain.
                if indata.ndim > 1 and indata.shape[1] > 1:
                    mono = indata.mean(axis=1).astype(np.float32, copy=False)
                else:
                    mono = indata.reshape(-1).astype(np.float32, copy=False)

                gin = float(10.0 ** (self.params.input_gain_db / 20.0))
                if gin != 1.0:
                    mono = mono * gin

                self.in_level = float(np.max(np.abs(mono))) if mono.size else 0.0

                in_ring.write_chunk(mono)

                # Process every available hop. With ring buffers, the driver
                # can hand us any blocksize and we still consume in hop units.
                bypass = self.params.bypass
                key = int(self.params.key)
                scale = str(self.params.scale)
                strength = float(self.params.strength)
                retune = float(self.params.retune_speed)
                mode = str(self.params.mode)
                bar_target = int(self.params.bar_target_semitone)
                octave = int(self.params.octave)
                additional = int(self.params.additional_range)
                while in_ring.fill >= hop:
                    in_ring.read_chunk(hop, scratch_in)
                    if bypass:
                        out_block = scratch_in.copy()
                    else:
                        out_block, hz = tuner.process(
                            scratch_in, key, scale, strength, retune,
                            mode=mode, bar_target_semitone=bar_target,
                            octave=octave, additional_range=additional,
                        )
                        self.detected_hz = hz
                    out_ring.write_chunk(out_block)

                # Hand the driver `frames` samples (zero-padding if starved).
                gout = float(10.0 ** (self.params.output_gain_db / 20.0))
                if outdata.ndim == 1:
                    out_ring.read_into(outdata, frames)
                    if gout != 1.0:
                        outdata *= gout
                    np.clip(outdata, -1.0, 1.0, out=outdata)
                    self.out_level = float(np.max(np.abs(outdata)))
                else:
                    tmp = np.empty(frames, dtype=np.float32)
                    out_ring.read_into(tmp, frames)
                    if gout != 1.0:
                        tmp *= gout
                    np.clip(tmp, -1.0, 1.0, out=tmp)
                    self.out_level = float(np.max(np.abs(tmp)))
                    outdata[:] = tmp[:, None]

            out_dev_info = sd.query_devices(p.output_device)
            out_channels = min(2, int(out_dev_info["max_output_channels"]))
            if out_channels < 1:
                out_channels = 1

            stream = sd.Stream(
                device=(p.input_device, p.output_device),
                samplerate=p.samplerate,
                # blocksize=0 lets the driver pick the optimal value.
                blocksize=0,
                channels=(1, out_channels),
                dtype="float32",
                callback=callback,
            )
            stream.start()
            self._stream = stream
            self._tuner = tuner
            self.last_status = ""
            self.dropout_count = 0

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
        if self._tuner is not None:
            self._tuner.reset()
            self._tuner = None
        self.in_level = 0.0
        self.out_level = 0.0
        self.detected_hz = 0.0
