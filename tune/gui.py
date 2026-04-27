"""Tkinter GUI for the realtime autotune engine."""
from __future__ import annotations

import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import sounddevice as sd

from .audio import DeviceInfo, Engine, list_input_devices, list_output_devices
from .params import Params
from .scales import NOTE_NAMES, SCALE_NAMES, SCALES, freq_to_midi, midi_to_name


FFT_SIZES = [1024, 2048, 4096]
SAMPLE_RATES = [44100, 48000]
MODES = ["Auto", "Bar"]

CONFIG_PATH = Path.home() / ".tune" / "config.json"

# clam-themed colour palettes. clam exposes the most surface area for
# customisation, so we force it as the base theme and override per element.
THEMES: dict[str, dict[str, str]] = {
    "Light": {
        "bg":         "#f0f0f0",
        "fg":         "#1a1a1a",
        "field_bg":   "#ffffff",
        "field_fg":   "#1a1a1a",
        "select_bg":  "#0078d4",
        "select_fg":  "#ffffff",
        "border":     "#a0a0a0",
        "trough":     "#d6d6d6",
        "bar":        "#0078d4",
        "muted":      "#666666",
        "disabled":   "#9a9a9a",
    },
    "Dark": {
        "bg":         "#1e1e1e",
        "fg":         "#e8e8e8",
        "field_bg":   "#2b2b2b",
        "field_fg":   "#e8e8e8",
        "select_bg":  "#264f78",
        "select_fg":  "#ffffff",
        "border":     "#3c3c3c",
        "trough":     "#3a3a3a",
        "bar":        "#3794ff",
        "muted":      "#a0a0a0",
        "disabled":   "#5a5a5a",
    },
}


def _load_config() -> dict:
    try:
        return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_config(cfg: dict) -> None:
    try:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    except Exception:
        pass


class TuneApp:
    def __init__(self) -> None:
        self.params = Params()
        self.engine = Engine(self.params)

        self._cfg = _load_config()
        self.theme: str = self._cfg.get("theme", "Light")
        if self.theme not in THEMES:
            self.theme = "Light"

        self.root = tk.Tk()
        self.root.title("tune — realtime autotune")
        self.root.geometry("520x680")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self._inputs: list[DeviceInfo] = []
        self._outputs: list[DeviceInfo] = []
        self._bar_snapping = False  # guard to break recursion when snapping the bar slider

        self._build_menu()
        self._build_ui()
        self._apply_theme(self.theme)
        self._refresh_devices()
        self._on_mode_change()  # set initial visibility of bar widgets
        self._tick()

    # ----- Menu ------------------------------------------------------------
    def _build_menu(self) -> None:
        self.menubar = tk.Menu(self.root)
        self.root.config(menu=self.menubar)

        self.file_menu = tk.Menu(self.menubar, tearoff=0)
        self.file_menu.add_command(label="Song processing...", command=self._open_song_dialog)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Settings...", command=self._open_settings)
        self.file_menu.add_separator()
        self.file_menu.add_command(label="Exit", command=self._on_close)
        self.menubar.add_cascade(label="File", menu=self.file_menu)

    # ----- UI construction -------------------------------------------------
    def _build_ui(self) -> None:
        pad = {"padx": 8, "pady": 4}
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=8, pady=8)
        frm = self.main_frame

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

        ttk.Label(pc, text="Mode").grid(row=0, column=0, sticky="w", padx=4, pady=2)
        self.mode_var = tk.StringVar(value="Auto")
        self.mode_box = ttk.Combobox(
            pc, textvariable=self.mode_var, state="readonly", width=8,
            values=MODES,
        )
        self.mode_box.grid(row=0, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(pc, text="Key").grid(row=1, column=0, sticky="w", padx=4, pady=2)
        self.key_var = tk.StringVar(value=NOTE_NAMES[0])
        ttk.Combobox(
            pc, textvariable=self.key_var, state="readonly", width=6,
            values=NOTE_NAMES,
        ).grid(row=1, column=1, sticky="w", padx=4, pady=2)

        ttk.Label(pc, text="Scale").grid(row=1, column=2, sticky="w", padx=4, pady=2)
        self.scale_var = tk.StringVar(value=SCALE_NAMES[0])
        ttk.Combobox(
            pc, textvariable=self.scale_var, state="readonly", width=18,
            values=SCALE_NAMES,
        ).grid(row=1, column=3, sticky="w", padx=4, pady=2)

        self.octave_var = tk.DoubleVar(value=4.0)
        self._labeled_scale(pc, "Octave", self.octave_var, 0, 9, row=2)

        self.range_var = tk.DoubleVar(value=0.0)
        self._labeled_scale(pc, "Additional step range", self.range_var, 0, 12, row=3,
                            suffix=" steps")

        # Bar mode target slider (row 4) — hidden in Auto mode.
        self.bar_target_label = ttk.Label(pc, text="Target")
        self.bar_target_label.grid(row=4, column=0, sticky="w", padx=4, pady=2)
        self.bar_var = tk.DoubleVar(value=0.0)
        self.bar_scale = ttk.Scale(
            pc, from_=0, to=11, variable=self.bar_var,
            orient="horizontal", command=self._on_bar_change,
        )
        self.bar_scale.grid(row=4, column=1, columnspan=2, sticky="ew", padx=4, pady=2)
        self.bar_value_var = tk.StringVar(value="—")
        self.bar_value_label = ttk.Label(pc, textvariable=self.bar_value_var, width=10)
        self.bar_value_label.grid(row=4, column=3, sticky="e", padx=4, pady=2)
        self._bar_widgets = (
            self.bar_target_label, self.bar_scale, self.bar_value_label,
        )

        self.retune_var = tk.DoubleVar(value=20.0)
        self._labeled_scale(pc, "Retune speed", self.retune_var, 0, 100, row=5, suffix="%")

        self.strength_var = tk.DoubleVar(value=100.0)
        self._labeled_scale(pc, "Strength", self.strength_var, 0, 100, row=6, suffix="%")

        pc.columnconfigure(1, weight=1)

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
        self.status_label = ttk.Label(frm, textvariable=self.status_var, foreground="gray")
        self.status_label.pack(fill="x", padx=8, pady=(8, 0))

        # Hook params changes
        for v in (self.key_var, self.scale_var, self.retune_var, self.strength_var,
                  self.gain_in_var, self.gain_out_var, self.bypass_var,
                  self.octave_var, self.range_var):
            v.trace_add("write", lambda *_: self._sync_params())

        self.mode_var.trace_add("write", lambda *_: self._on_mode_change())
        self.scale_var.trace_add("write", lambda *_: self._on_scale_change())
        self.key_var.trace_add("write", lambda *_: self._update_bar_label())

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

    # ----- mode / bar slider behaviour ------------------------------------
    def _on_mode_change(self) -> None:
        mode = self.mode_var.get() or "Auto"
        if mode == "Bar":
            for w in self._bar_widgets:
                w.grid()
            self._on_bar_change(None)  # snap + push current value
        else:
            for w in self._bar_widgets:
                w.grid_remove()
        self._sync_params()

    def _on_scale_change(self) -> None:
        # Re-snap the bar slider to a valid degree of the new scale.
        if self.mode_var.get() == "Bar":
            self._on_bar_change(None)
        self._update_bar_label()

    def _on_bar_change(self, _value) -> None:
        if self._bar_snapping:
            return
        intervals = SCALES.get(self.scale_var.get(), SCALES["Chromatic"])
        try:
            raw = int(round(float(self.bar_var.get())))
        except (ValueError, tk.TclError):
            raw = 0
        raw = max(0, min(11, raw))
        nearest = min(intervals, key=lambda i: abs(i - raw))
        if abs(self.bar_var.get() - nearest) > 0.001:
            self._bar_snapping = True
            try:
                self.bar_var.set(nearest)
            finally:
                self._bar_snapping = False
        self.params.bar_target_semitone = int(nearest)
        self._update_bar_label()

    def _update_bar_label(self) -> None:
        try:
            key_idx = NOTE_NAMES.index(self.key_var.get())
        except ValueError:
            key_idx = 0
        semitone = int(self.params.bar_target_semitone) % 12
        name = NOTE_NAMES[(key_idx + semitone) % 12]
        self.bar_value_var.set(name)

    # ----- behavior -------------------------------------------------------
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
        self.params.mode = self.mode_var.get() or "Auto"
        self.params.octave = int(round(self.octave_var.get()))
        self.params.additional_range = int(round(self.range_var.get()))
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

    # ----- song processing -----------------------------------------------
    def _open_song_dialog(self) -> None:
        # Late import so this module's import doesn't pull mido / soundfile
        # / pydub unless the user actually opens the dialog.
        from .song_dialog import SongDialog
        SongDialog(self)

    # ----- settings + theming --------------------------------------------
    def _open_settings(self) -> None:
        dlg = tk.Toplevel(self.root)
        dlg.title("Settings")
        dlg.transient(self.root)
        dlg.resizable(False, False)
        dlg.configure(bg=THEMES[self.theme]["bg"])

        frm = ttk.Frame(dlg, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Interface theme").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        theme_var = tk.StringVar(value=self.theme)
        ttk.Combobox(
            frm, textvariable=theme_var, state="readonly",
            values=list(THEMES.keys()), width=10,
        ).grid(row=0, column=1, sticky="w", padx=4, pady=4)

        btns = ttk.Frame(frm)
        btns.grid(row=1, column=0, columnspan=2, sticky="e", pady=(12, 0))

        def on_ok() -> None:
            new_theme = theme_var.get()
            if new_theme in THEMES and new_theme != self.theme:
                self.theme = new_theme
                self._apply_theme(self.theme)
                self._cfg["theme"] = self.theme
                _save_config(self._cfg)
            dlg.destroy()

        ttk.Button(btns, text="Cancel", command=dlg.destroy).pack(side="right", padx=4)
        ttk.Button(btns, text="OK", command=on_ok).pack(side="right", padx=4)

        dlg.update_idletasks()
        # Centre over the parent window.
        px, py = self.root.winfo_x(), self.root.winfo_y()
        pw, ph = self.root.winfo_width(), self.root.winfo_height()
        dw, dh = dlg.winfo_width(), dlg.winfo_height()
        dlg.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")
        dlg.grab_set()
        dlg.focus_set()

    def _apply_theme(self, theme_name: str) -> None:
        t = THEMES.get(theme_name, THEMES["Light"])

        self.root.configure(bg=t["bg"])

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(".",
                        background=t["bg"], foreground=t["fg"],
                        fieldbackground=t["field_bg"],
                        bordercolor=t["border"], lightcolor=t["bg"], darkcolor=t["bg"])
        style.configure("TFrame", background=t["bg"])
        style.configure("TLabel", background=t["bg"], foreground=t["fg"])
        style.configure("TLabelframe", background=t["bg"], foreground=t["fg"],
                        bordercolor=t["border"])
        style.configure("TLabelframe.Label", background=t["bg"], foreground=t["fg"])

        style.configure("TButton",
                        background=t["field_bg"], foreground=t["fg"],
                        bordercolor=t["border"], lightcolor=t["field_bg"], darkcolor=t["field_bg"])
        style.map("TButton",
                  background=[("active", t["select_bg"]), ("disabled", t["bg"])],
                  foreground=[("active", t["select_fg"]), ("disabled", t["disabled"])])

        style.configure("TCheckbutton", background=t["bg"], foreground=t["fg"],
                        indicatorcolor=t["field_bg"], focuscolor=t["bg"])
        style.map("TCheckbutton",
                  background=[("active", t["bg"])],
                  foreground=[("disabled", t["disabled"])],
                  indicatorcolor=[("selected", t["select_bg"])])

        style.configure("TCombobox",
                        fieldbackground=t["field_bg"], background=t["field_bg"],
                        foreground=t["field_fg"], arrowcolor=t["fg"],
                        bordercolor=t["border"], lightcolor=t["field_bg"], darkcolor=t["field_bg"])
        style.map("TCombobox",
                  fieldbackground=[("readonly", t["field_bg"]), ("disabled", t["bg"])],
                  foreground=[("readonly", t["field_fg"]), ("disabled", t["disabled"])],
                  background=[("readonly", t["field_bg"])],
                  arrowcolor=[("disabled", t["disabled"])])
        # Combobox dropdown listbox is a plain tk.Listbox internally.
        self.root.option_add("*TCombobox*Listbox.background", t["field_bg"])
        self.root.option_add("*TCombobox*Listbox.foreground", t["field_fg"])
        self.root.option_add("*TCombobox*Listbox.selectBackground", t["select_bg"])
        self.root.option_add("*TCombobox*Listbox.selectForeground", t["select_fg"])

        style.configure("Horizontal.TScale",
                        background=t["bg"], troughcolor=t["trough"],
                        bordercolor=t["border"], lightcolor=t["bg"], darkcolor=t["bg"])
        style.configure("Horizontal.TProgressbar",
                        background=t["bar"], troughcolor=t["trough"],
                        bordercolor=t["border"], lightcolor=t["bg"], darkcolor=t["bg"])

        # Status label sits on the main frame; tone it down vs. main fg.
        if hasattr(self, "status_label"):
            self.status_label.configure(foreground=t["muted"])

        # tk.Menu widgets — these don't pick up ttk styles.
        for menu in (self.menubar, self.file_menu):
            menu.configure(
                bg=t["bg"], fg=t["fg"],
                activebackground=t["select_bg"],
                activeforeground=t["select_fg"],
                borderwidth=0,
            )

    def _on_close(self) -> None:
        self.engine.stop()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()
