"""FFT phase-vocoder pitch shifter with autocorrelation pitch detection
and Laroche-Dolson loose phase locking.

Block size handed to `process()` must equal `hop` (= fft_size // 4).
Hann window with 75% overlap gives a constant overlap-add factor of 1.5
(when synthesis windowing is also applied).
"""
from __future__ import annotations

import collections

import numpy as np

from .scales import SCALES, freq_to_midi, midi_to_freq


_TWO_PI = 2.0 * np.pi
_PI = np.pi
_RATIO_MIN = 2.0 ** (-4.0 / 12.0)  # -4 semitones ≈ 0.7937
_RATIO_MAX = 2.0 ** (4.0 / 12.0)   # +4 semitones ≈ 1.2599


class Autotuner:
    def __init__(self, fft_size: int, sr: int) -> None:
        if fft_size & (fft_size - 1):
            raise ValueError("fft_size must be a power of two")
        if fft_size < 256:
            raise ValueError("fft_size too small")
        self.N = int(fft_size)
        self.hop = self.N // 4
        self.sr = int(sr)
        self.n_bins = self.N // 2 + 1

        self.window = np.hanning(self.N).astype(np.float32)
        # Hann + 75% overlap with synthesis windowing → sum of overlapping w^2 = 1.5
        self.cola_norm = np.float32(1.5)

        # Autocorrelation of the analysis window. Used to debias the
        # FFT-based ACF pitch detector: ACF(windowed_x) ≈ ACF(x) * ACF(window),
        # so dividing by ACF(window) recovers the underlying signal's ACF.
        # Without this, low-frequency signals (period > N/4) lose to
        # short-lag peaks because the window's own AC is large there.
        w64 = self.window.astype(np.float64)
        w_spec = np.fft.rfft(w64)
        w_acf = np.fft.irfft(np.abs(w_spec) ** 2, n=self.N)
        self._w_acf = np.maximum(w_acf, w_acf[0] * 1e-3)

        # Vocal-range bandpass mask applied to the magnitude spectrum
        # before pitch-detection ACF. Excludes sub-bass rumble and
        # high-frequency hiss/fan noise that would otherwise contribute
        # spurious peaks to the autocorrelation. Low edge is set above
        # the Hann main-lobe leakage region of common AC/HVAC noise so
        # those don't bleed into the passband. PV pitch shift itself
        # uses the unmasked spectrum so output tonality is unaffected.
        self._pitch_band = self._make_bandpass_mask(90.0, 2500.0, transition_hz=30.0)

        self.in_buf = np.zeros(self.N, dtype=np.float32)
        self.out_buf = np.zeros(self.N, dtype=np.float32)
        self.last_phase = np.zeros(self.n_bins, dtype=np.float64)
        self.sum_phase = np.zeros(self.n_bins, dtype=np.float64)

        self.bin_idx = np.arange(self.n_bins, dtype=np.float64)
        self.expected = _TWO_PI * self.hop * self.bin_idx / self.N

        self.smoothed_ratio = 1.0
        self.last_hz = 0.0
        self._voiced = False
        self._smoothed_midi: float | None = None
        self._snapped_midi: float | None = None
        self._midi_history: collections.deque[float] = collections.deque(maxlen=5)
        self._outlier_count = 0
        self._last_valid_hz = 0.0

    def reset(self) -> None:
        self.in_buf.fill(0.0)
        self.out_buf.fill(0.0)
        self.last_phase.fill(0.0)
        self.sum_phase.fill(0.0)
        self.smoothed_ratio = 1.0
        self.last_hz = 0.0
        self._voiced = False
        self._smoothed_midi = None
        self._snapped_midi = None
        self._midi_history.clear()
        self._outlier_count = 0
        self._last_valid_hz = 0.0

    def process(
        self,
        block: np.ndarray,
        key: int,
        scale: str,
        strength: float,
        retune_speed: float,
    ) -> tuple[np.ndarray, float]:
        if block.shape[0] != self.hop:
            raise ValueError(f"expected hop={self.hop} samples, got {block.shape[0]}")

        # Slide analysis buffer left by hop, append the new block.
        self.in_buf[: -self.hop] = self.in_buf[self.hop :]
        self.in_buf[-self.hop :] = block

        windowed = self.in_buf * self.window
        spec = np.fft.rfft(windowed)
        mag = np.abs(spec)
        phase = np.angle(spec)

        hz = self._detect_pitch_robust(mag)
        self.last_hz = hz

        target_ratio = self._snap_with_hysteresis(hz, key, scale) if hz > 0.0 else 1.0

        # Retune speed: 0 → 1 ms time constant (instant), 100 → 250 ms (slow glide).
        rs = max(0.0, min(1.0, retune_speed / 100.0))
        tau = 0.001 + 0.25 * rs
        alpha = 1.0 - float(np.exp(-self.hop / (tau * self.sr)))
        self.smoothed_ratio += alpha * (target_ratio - self.smoothed_ratio)

        # Strength: log-blend ratio toward 1.0 (0% = identity, 100% = full snap).
        s = max(0.0, min(1.0, strength / 100.0))
        if self.smoothed_ratio > 0.0:
            eff_ratio = float(np.exp(s * np.log(self.smoothed_ratio)))
        else:
            eff_ratio = 1.0
        eff_ratio = max(_RATIO_MIN, min(_RATIO_MAX, eff_ratio))  # ±4 semitones

        # ---- Instantaneous frequency per input bin (in bins) ---------------
        delta_phase = phase - self.last_phase
        self.last_phase[:] = phase
        delta_phase -= self.expected
        delta_phase = (delta_phase + _PI) % _TWO_PI - _PI
        true_freq_bins = self.bin_idx + delta_phase * self.N / (_TWO_PI * self.hop)

        # ---- Pitch shift via spectral interpolation ------------------------
        # For each output bin k_out, source position = k_out / eff_ratio.
        src = self.bin_idx / eff_ratio
        src_lo = np.floor(src).astype(np.int64)
        frac = src - src_lo
        src_hi = src_lo + 1
        valid = (src_lo >= 0) & (src_hi <= self.n_bins - 1)
        src_lo_c = np.clip(src_lo, 0, self.n_bins - 1)
        src_hi_c = np.clip(src_hi, 0, self.n_bins - 1)

        new_mag = mag[src_lo_c] * (1.0 - frac) + mag[src_hi_c] * frac
        new_freq = (
            true_freq_bins[src_lo_c] * (1.0 - frac)
            + true_freq_bins[src_hi_c] * frac
        ) * eff_ratio
        new_mag = np.where(valid, new_mag, 0.0)
        new_freq = np.where(valid, new_freq, 0.0)

        # Preserve total spectral energy: linear interpolation can spread a
        # sharp peak across two output bins, losing up to 3 dB. Re-normalize.
        in_e = float(np.sum(mag * mag))
        out_e = float(np.sum(new_mag * new_mag))
        if in_e > 1e-12 and out_e > 1e-12:
            new_mag *= np.sqrt(in_e / out_e)

        # ---- Free-running synthesis phase (PV) -----------------------------
        self.sum_phase += _TWO_PI * self.hop * new_freq / self.N

        # ---- Loose phase locking (Laroche-Dolson) --------------------------
        # Locks each non-peak output bin's phase to the synthesis phase of its
        # owning peak, plus the input-side phase offset between this bin and
        # the peak. Eliminates the "phasiness" / energy decay caused by free
        # phase accumulation per bin.
        peak_mask = np.zeros(self.n_bins, dtype=bool)
        peak_mask[1:-1] = (mag[1:-1] > mag[:-2]) & (mag[1:-1] > mag[2:])
        # Conservative threshold: only count peaks with real energy. A loose
        # threshold flags Hann sidelobes as peaks, which then flicker as the
        # signal's sub-bin position drifts and causes phase discontinuities.
        peak_thresh = 0.05 * float(np.max(mag) + 1e-12)
        peak_mask &= mag > peak_thresh
        peaks = np.where(peak_mask)[0]

        if peaks.size > 0:
            # Output position of each input peak (one per peak).
            peak_out_positions = np.rint(
                peaks.astype(np.float64) * eff_ratio
            ).astype(np.int64)
            peak_out_positions = np.clip(peak_out_positions, 0, self.n_bins - 1)

            # Override the free-run sum_phase at peak output positions with
            # an advance computed from the *peak's exact* instantaneous
            # frequency (interpolation contaminates this with sidelobe noise).
            peak_freq_exact = true_freq_bins[peaks] * eff_ratio
            peak_advance = _TWO_PI * self.hop * peak_freq_exact / self.N
            # Undo the wrong (interpolated) advance and apply the correct one.
            wrong_advance = _TWO_PI * self.hop * new_freq[peak_out_positions] / self.N
            self.sum_phase[peak_out_positions] += peak_advance - wrong_advance

            # For each output bin, find the owning input peak (nearest in
            # input-bin space relative to its source position `src`).
            if peaks.size == 1:
                owner_idx = np.zeros(self.n_bins, dtype=np.int64)
            else:
                midpoints = 0.5 * (peaks[:-1] + peaks[1:]).astype(np.float64)
                owner_idx = np.searchsorted(midpoints, src)
                owner_idx = np.clip(owner_idx, 0, peaks.size - 1)
            owner_in = peaks[owner_idx]
            owner_out = peak_out_positions[owner_idx]

            # Input phase at the source of each output bin (interpolated,
            # with proper wrap to keep the linear interp stable).
            phase_lo = phase[src_lo_c]
            phase_hi = phase[src_hi_c]
            d = (phase_hi - phase_lo + _PI) % _TWO_PI - _PI
            src_phase = phase_lo + frac * d
            owner_phase = phase[owner_in]
            offset = (src_phase - owner_phase + _PI) % _TWO_PI - _PI

            locked = self.sum_phase[owner_out] + offset
            is_owner = np.zeros(self.n_bins, dtype=bool)
            is_owner[peak_out_positions] = True
            self.sum_phase = np.where(is_owner, self.sum_phase, locked)

        self.sum_phase = (self.sum_phase + _PI) % _TWO_PI - _PI

        # ---- Synthesis ------------------------------------------------------
        new_spec = (new_mag * np.exp(1j * self.sum_phase)).astype(np.complex64)
        time_frame = np.fft.irfft(new_spec, n=self.N).astype(np.float32)
        time_frame *= self.window

        self.out_buf += time_frame / self.cola_norm
        out = self.out_buf[: self.hop].copy()
        self.out_buf[: -self.hop] = self.out_buf[self.hop :]
        self.out_buf[-self.hop :] = 0.0

        return out, hz

    def _snap_with_hysteresis(
        self,
        hz: float,
        key: int,
        scale: str,
        smooth_tau_s: float = 0.2,
        switch_margin_cents: float = 30.0,
    ) -> float:
        """Smooth the detected pitch in MIDI space, then snap to a scale
        note with hysteresis. Returns the ratio that maps current hz to
        the committed target.

        Smoothing averages out vibrato so the snap target stays committed
        to one note even when the live pitch wobbles. Hysteresis prevents
        flipping when the smoothed pitch sits exactly between two scale
        notes: a new target must beat the current one by at least
        ``switch_margin_cents`` to be adopted. Without these the snap
        target oscillates each vibrato cycle, which sounds like cuts.
        """
        midi_now = freq_to_midi(hz)
        if self._smoothed_midi is None:
            self._smoothed_midi = float(midi_now)
        else:
            alpha = 1.0 - float(np.exp(-(self.hop / self.sr) / smooth_tau_s))
            self._smoothed_midi += alpha * (midi_now - self._smoothed_midi)

        intervals = SCALES.get(scale, SCALES["Chromatic"])
        smooth = self._smoothed_midi
        octave = int(round(smooth / 12.0))
        candidates = [
            o * 12 + key + i
            for o in (octave - 1, octave, octave + 1)
            for i in intervals
        ]
        nearest = float(min(candidates, key=lambda m: abs(m - smooth)))

        if self._snapped_midi is None:
            self._snapped_midi = nearest
        elif nearest != self._snapped_midi:
            cents_to_current = 100.0 * abs(smooth - self._snapped_midi)
            cents_to_nearest = 100.0 * abs(smooth - nearest)
            if cents_to_current - cents_to_nearest >= switch_margin_cents:
                self._snapped_midi = nearest

        return midi_to_freq(self._snapped_midi) / hz

    def _detect_pitch_robust(
        self,
        mag: np.ndarray,
        outlier_semitones: float = 6.0,
        outlier_recovery_frames: int = 5,
    ) -> float:
        """Robust pitch detection: ACF + 5-sample median filter + outlier
        rejection.

        The raw ACF detector occasionally publishes octave-wrong readings
        (it picks 2T or T/2 over the true period T). Without filtering,
        those bad frames feed the snap/ratio pipeline and produce sudden
        large pitch swings in the output. The median rejects single-frame
        outliers; the deviation check holds the previous valid hz when a
        single frame disagrees with the running smoothed midi by more than
        ``outlier_semitones``. After ``outlier_recovery_frames`` consecutive
        outliers, the new pitch is accepted as a real jump.
        """
        raw_hz = self._detect_pitch(mag)
        if raw_hz <= 0.0:
            self._midi_history.clear()
            self._outlier_count = 0
            return 0.0

        raw_midi = freq_to_midi(raw_hz)
        self._midi_history.append(raw_midi)

        # Warmup — accept directly until we have enough samples to median.
        if len(self._midi_history) < 3:
            self._outlier_count = 0
            self._last_valid_hz = raw_hz
            return raw_hz

        median_midi = float(np.median(self._midi_history))
        median_hz = float(midi_to_freq(median_midi))

        if (
            self._smoothed_midi is not None
            and abs(median_midi - self._smoothed_midi) > outlier_semitones
        ):
            self._outlier_count += 1
            if self._outlier_count < outlier_recovery_frames:
                # Hold previous valid pitch; don't propagate the jump.
                return self._last_valid_hz if self._last_valid_hz > 0.0 else 0.0
            # Sustained disagreement — accept as a real pitch change.
            self._outlier_count = 0
        else:
            self._outlier_count = 0

        self._last_valid_hz = median_hz
        return median_hz

    def _make_bandpass_mask(
        self,
        lo_hz: float,
        hi_hz: float,
        transition_hz: float = 20.0,
    ) -> np.ndarray:
        """Raised-cosine bandpass mask over the rfft bins. 1.0 inside the
        passband, 0.0 outside, smooth raised-cosine ramps of width
        ``transition_hz`` on each edge to avoid ringing."""
        freqs = np.arange(self.n_bins, dtype=np.float64) * self.sr / self.N
        mask = np.zeros(self.n_bins, dtype=np.float64)
        mask[(freqs >= lo_hz) & (freqs <= hi_hz)] = 1.0
        lo_ramp = (freqs >= lo_hz - transition_hz) & (freqs < lo_hz)
        if lo_ramp.any():
            mask[lo_ramp] = 0.5 * (
                1.0 - np.cos(
                    np.pi * (freqs[lo_ramp] - (lo_hz - transition_hz)) / transition_hz
                )
            )
        hi_ramp = (freqs > hi_hz) & (freqs <= hi_hz + transition_hz)
        if hi_ramp.any():
            mask[hi_ramp] = 0.5 * (
                1.0 + np.cos(np.pi * (freqs[hi_ramp] - hi_hz) / transition_hz)
            )
        return mask

    def _detect_pitch(
        self,
        mag: np.ndarray,
        fmin: float = 70.0,
        fmax: float = 700.0,
        voicing_enter: float = 0.4,
        voicing_exit: float = 0.25,
    ) -> float:
        """ACF-via-FFT pitch detector with octave-error correction and
        voicing hysteresis. Returns 0.0 if unvoiced."""
        # Apply vocal-range bandpass to the magnitude spectrum before ACF.
        # Out-of-band content (sub-bass, hiss) doesn't carry pitch info but
        # does contribute spurious ACF energy at unrelated lags.
        masked = mag.astype(np.float64) * self._pitch_band
        power = masked * masked
        acf = np.fft.irfft(power, n=self.N)
        # Debias for the analysis window's own autocorrelation so peaks at
        # the true period dominate over short-lag artifacts.
        acf = acf / self._w_acf
        e0 = float(acf[0])
        if e0 < 1e-9:
            self._voiced = False
            return 0.0

        period_min = max(2, int(self.sr / fmax))
        period_max = min(self.N // 2 - 1, int(self.sr / fmin))
        if period_max <= period_min + 2:
            self._voiced = False
            return 0.0

        region = acf[period_min : period_max + 1]
        peak_local = int(np.argmax(region))
        peak = period_min + peak_local
        peak_val = float(acf[peak])

        # Octave-error correction: ACF has near-equal peaks at every multiple
        # of the true period, and argmax can land on 2× or 3× the period
        # (reading the pitch one or two octaves below true). Walk down from
        # the picked period to the smallest sub-multiple whose ACF is at
        # least 80% of peak_val and still inside the search range.
        for divisor in (4, 3, 2):
            sub = peak / divisor
            if sub < period_min:
                continue
            sub_int = int(round(sub))
            if 1 <= sub_int < self.N // 2 - 1:
                # Take the local max of [sub_int-1, sub_int, sub_int+1] to
                # tolerate small period-ratio jitter.
                local = max(acf[sub_int - 1], acf[sub_int], acf[sub_int + 1])
                if local >= 0.8 * peak_val:
                    peak = sub_int
                    peak_val = float(local)
                    break

        # Voicing hysteresis: harder to enter voiced state than to stay in it.
        threshold = voicing_exit if self._voiced else voicing_enter
        if peak_val / e0 < threshold:
            self._voiced = False
            return 0.0
        self._voiced = True

        if 1 <= peak < self.N // 2 - 1:
            a = float(acf[peak - 1])
            b = float(acf[peak])
            c = float(acf[peak + 1])
            denom = a - 2.0 * b + c
            offset = 0.5 * (a - c) / denom if abs(denom) > 1e-12 else 0.0
            peak_f = float(peak) + offset
        else:
            peak_f = float(peak)

        if peak_f <= 0.0:
            return 0.0
        return float(self.sr) / peak_f
