"""Tkinter GUI for the realtime autotune engine."""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

import sounddevice as sd

from .audio import DeviceInfo, Engine, list_input_devices, list_output_devices
from .params import Params
from .scales import NOTE_NAMES, SCALE_NAMES, SCALES, freq_to_midi, midi_to_name


FFT_SIZES = [1024, 2048, 4096]
SAMPLE_RATES = [44100, 48000]


class TuneApp:
    def __init__(self) -> None:
        self.params = Params()
        self.engine = Engine(self.params)

        self.root = tk.Tk()
        self.root.title("tune — realtime autotune")
        self.root.geometry("520x640")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._inputs: list[DeviceInfo] = []
        self._outputs: list[DeviceInfo] = []

        self._build_ui()
        self._refresh_devices()
        self._tick()

    # ----- UI construction -----
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        frm = ttk.Frame(self.root)
        frm.pack(fill="both", expand=True, padx=8, pady=8)

        # Devices ---------------------------------------------------------
        dev = ttk.LabelFrame(frm, text="Audio devices")
        dev.pack(fill="x", **pad)

        ttk.Label(dev, text="Input").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.input_var = tk.StringVar()
        self.input_box = ttk.Combobox(dev, textvariable=self.input_var, state="readonly", width=55)
        self.input_box.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(dev, text="Output").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.output_var = tk.StringVar()
        self.output_box = ttk.Combobox(dev, textvariable=self.output_var, state="readonly", width=55)
        self.output_box.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(dev, text="Sample rate").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.sr_var = tk.IntVar(value=48000)
        self.sr_box = ttk.Combobox(
            dev, textvariable=self.sr_var, state="readonly", width=10,
            values=[str(s) for s in SAMPLE_RATES],
        )
        self.sr_box.grid(row=2, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(dev, text="FFT size").grid(row=3, column=0, sticky="w", padx=4, pady=2)
        self.fft_var = tk.IntVar(value=2048)
        self.fft_box = ttk.Combobox(
            dev, textvariable=self.fft_var, state="readonly", width=10,
            values=[str(s) for s in FFT_SIZES],
        )
        self.fft_box.grid(row=3, column=1, sticky="w", padx=4, pady=2)

        refresh = ttk.Button(dev, text="Refresh devices", command=self._refresh_devices)
        refresh.grid(row=4, column=1, sticky="e", padx=4, pady=4)
        dev.columnconfigure(1, weight=1)

        # Pitch correction ------------------------------------------------
        pc = ttk.LabelFrame(frm, text="Pitch correction")
        pc.pack(fill="x", **pad)

        ttk.Label(pc, text="Key").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.key_var = tk.StringVar(value=NOTE_NAMES[0])
        ttk.Combobox(
            pc, textvariable=self.key_var, state="readonly", width=6,
            values=NOTE_NAMES,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(pc, text="Scale").grid(row=0, column=2, sticky="w", padx=4, pady=2)
        self.scale_var = tk.StringVar(value=SCALE_NAMES[0])
        ttk.Combobox(
            pc, textvariable=self.scale_var, state="readonly", width=18,
            values=SCALE_NAMES,
        ).grid(row=0, column=3, sticky="w", padx=4, pady=2)

        self.retune_var = tk.DoubleVar(value=20.0)
        self._labeled_scale(pc, "Retune speed", self.retune_var, 0, 100, row=1, suffix="%")

        self.strength_var = tk.DoubleVar(value=100.0)
        self._labeled_scale(pc, "Strength", self.strength_var, 0, 100, row=2, suffix="%")

        # Levels ----------------------------------------------------------
        lvl = ttk.LabelFrame(frm, text="Levels")
        lvl.pack(fill="x", **pad)
        self.gain_in_var = tk.DoubleVar(value=0.0)
        self._labeled_scale(lvl, "Input gain", self.gain_in_var, -24, 24, row=0, suffix=" dB")
        self.gain_out_var = tk.DoubleVar(value=0.0)
        self._labeled_scale(lvl, "Output gain", self.gain_out_var, -24, 24, row=1, suffix=" dB")

        # Meters ----------------------------------------------------------
        meters = ttk.LabelFrame(frm, text="Monitoring")
        meters.pack(fill="x", **pad)

        ttk.Label(meters, text="Input").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.in_meter = ttk.Progressbar(meters, length=320, mode="determinate", maximum=100)
        self.in_meter.grid(row=0, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(meters, text="Output").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.out_meter = ttk.Progressbar(meters, length=320, mode="determinate", maximum=100)
        self.out_meter.grid(row=1, column=1, sticky="ew", padx=4, pady=2)

        ttk.Label(meters, text="Detected").grid(row=2, column=0, sticky="w", padx=4, pady=2)
        self.detected_var = tk.StringVar(value="—")
        ttk.Label(meters, textvariable=self.detected_var, font=("TkDefaultFont", 11, "bold"))\
            .grid(row=2, column=1, sticky="w", padx=4, pady=2)

        meters.columnconfigure(1, weight=1)

        # Transport -------------------------------------------------------
        ctrl = ttk.Frame(frm)
        ctrl.pack(fill="x", **pad)
        self.start_btn = ttk.Button(ctrl, text="Start", command=self._on_start)
        self.start_btn.pack(side="left", padx=4)
        self.stop_btn = ttk.Button(ctrl, text="Stop", command=self._on_stop, state="disabled")
        self.stop_btn.pack(side="left", padx=4)
        self.bypass_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(ctrl, text="Bypass", variable=self.bypass_var,
                        command=self._sync_params).pack(side="left", padx=12)

        self.status_var = tk.StringVar(value="Stopped.")
        ttk.Label(frm, textvariable=self.status_var, foreground="gray")\
            .pack(fill="x", padx=8, pady=(8, 0))

        # Hook params changes
        for v in (self.key_var, self.scale_var, self.retune_var, self.strength_var,
                  self.gain_in_var, self.gain_out_var, self.bypass_var):
            v.trace_add("write", lambda *_: self._sync_params())

    def _labeled_scale(self, parent, label, var, lo, hi, row, suffix="") -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=2)
        sc = ttk.Scale(parent, from_=lo, to=hi, variable=var, orient="horizontal")
        sc.grid(row=row, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        val = tk.StringVar()

        def update(*_):
            val.set(f"{var.get():+.1f}{suffix}" if "dB" in suffix else f"{var.get():.0f}{suffix}")

        var.trace_add("write", update)
        update()
        ttk.Label(parent, textvariable=val, width=10).grid(row=row, column=3, sticky="e", padx=4, pady=2)
        parent.columnconfigure(1, weight=1)

    # ----- behavior -----
    def _refresh_devices(self) -> None:
        self._inputs = list_input_devices()
        self._outputs = list_output_devices()
        self.input_box["values"] = [d.label for d in self._inputs]
        self.output_box["values"] = [d.label for d in self._outputs]

        try:
            default_in, default_out = sd.default.device
        except Exception:
            default_in = default_out = None

        if self._inputs and not self.input_var.get():
            sel = next((d for d in self._inputs if d.index == default_in), self._inputs[0])
            self.input_var.set(sel.label)
        if self._outputs and not self.output_var.get():
            sel = next((d for d in self._outputs if d.index == default_out), self._outputs[0])
            self.output_var.set(sel.label)

    def _selected_device(self, dlist: list[DeviceInfo], label: str) -> DeviceInfo | None:
        for d in dlist:
            if d.label == label:
                return d
        return None

    def _sync_params(self) -> None:
        try:
            self.params.key = NOTE_NAMES.index(self.key_var.get())
        except ValueError:
            self.params.key = 0
        self.params.scale = self.scale_var.get() or "Chromatic"
        self.params.retune_speed = float(self.retune_var.get())
        self.params.strength = float(self.strength_var.get())
        self.params.input_gain_db = float(self.gain_in_var.get())
        self.params.output_gain_db = float(self.gain_out_var.get())
        self.params.bypass = bool(self.bypass_var.get())

    def _on_start(self) -> None:
        in_dev = self._selected_device(self._inputs, self.input_var.get())
        out_dev = self._selected_device(self._outputs, self.output_var.get())
        if in_dev is None or out_dev is None:
            messagebox.showerror("tune", "Select an input and output device.")
            return

        self.params.input_device = in_dev.index
        self.params.output_device = out_dev.index
        try:
            self.params.samplerate = int(self.sr_var.get())
            self.params.fft_size = int(self.fft_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror("tune", "Invalid sample rate or FFT size.")
            return
        self._sync_params()

        try:
            self.engine.start()
        except Exception as e:
            messagebox.showerror("tune", f"Could not start audio:\n{e}")
            return

        self._set_running(True)
        self.status_var.set(
            f"Running @ {self.params.samplerate} Hz, FFT={self.params.fft_size}, "
            f"hop={self.params.fft_size // 4}"
        )

    def _on_stop(self) -> None:
        self.engine.stop()
        self._set_running(False)
        self.status_var.set("Stopped.")

    def _set_running(self, running: bool) -> None:
        if running:
            self.start_btn.state(["disabled"])
            self.stop_btn.state(["!disabled"])
            for w in (self.input_box, self.output_box, self.sr_box, self.fft_box):
                w.state(["disabled"])
        else:
            self.start_btn.state(["!disabled"])
            self.stop_btn.state(["disabled"])
            for w in (self.input_box, self.output_box, self.sr_box, self.fft_box):
                w.state(["!disabled"])

    def _tick(self) -> None:
        in_pct = min(100.0, max(0.0, self.engine.in_level * 100.0))
        out_pct = min(100.0, max(0.0, self.engine.out_level * 100.0))
        self.in_meter["value"] = in_pct
        self.out_meter["value"] = out_pct

        hz = self.engine.detected_hz
        if hz > 0.0:
            midi = freq_to_midi(hz)
            intervals = SCALES.get(self.params.scale, SCALES["Chromatic"])
            octave = int(round(midi / 12))
            cands = [o * 12 + self.params.key + i
                     for o in (octave - 1, octave, octave + 1) for i in intervals]
            nearest = min(cands, key=lambda m: abs(m - midi))
            cents = 1200.0 * (midi - nearest)
            self.detected_var.set(
                f"{hz:6.1f} Hz  →  {midi_to_name(nearest)}  ({cents:+.0f} cents)"
            )
        else:
            self.detected_var.set("—")

        if self.engine.is_running:
            base = (
                f"Running @ {self.params.samplerate} Hz, "
                f"FFT={self.params.fft_size}, "
                f"hop={self.params.fft_size // 4}  |  "
                f"dropouts: {self.engine.dropout_count}"
            )
            if self.engine.last_status:
                base += f"  |  {self.engine.last_status}"
            self.status_var.set(base)

        self.root.after(50, self._tick)

    def _on_close(self) -> None:
        self.engine.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
