"""Song-processing dialog. Owned by TuneApp; opened from File → Song processing."""
from __future__ import annotations

import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import numpy as np
import sounddevice as sd

from . import song
from .config import load_config, update_config


# Keep dialog file-pickers using these extensions.
_VOICE_EXTS = (
    ("Audio files", "*.wav *.flac *.ogg *.aiff *.aif *.mp3 *.m4a *.aac"),
    ("All files", "*.*"),
)
_MIDI_EXTS = (
    ("MIDI files", "*.mid *.midi"),
    ("All files", "*.*"),
)
_OUTPUT_EXTS = (
    ("WAV", "*.wav"),
    ("FLAC", "*.flac"),
    ("Ogg Vorbis", "*.ogg"),
    ("MP3", "*.mp3"),
)


class SongDialog:
    def __init__(self, app) -> None:
        self.app = app
        self.root_parent: tk.Tk = app.root

        self.win = tk.Toplevel(self.root_parent)
        self.win.title("Song Processing")
        self.win.transient(self.root_parent)
        self.win.protocol("WM_DELETE_WINDOW", self._on_close)
        self.win.resizable(False, False)
        # Inherit the parent's theme background.
        try:
            from .gui import THEMES
            self.win.configure(bg=THEMES[app.theme]["bg"])
        except Exception:
            pass

        # Preload the last paths used so the user doesn't have to re-pick on
        # each invocation. Only restore paths whose files still exist on disk.
        self._cfg = load_config()
        last_voice = str(self._cfg.get("song_last_voice", "") or "")
        last_midi = str(self._cfg.get("song_last_midi", "") or "")
        last_output = str(self._cfg.get("song_last_output", "") or "")
        last_overlay = bool(self._cfg.get("song_last_overlay", False))

        self.voice_path_var = tk.StringVar(
            value=last_voice if last_voice and Path(last_voice).is_file() else ""
        )
        self.midi_path_var = tk.StringVar(
            value=last_midi if last_midi and Path(last_midi).is_file() else ""
        )
        # Output path is allowed to point at a not-yet-existing file.
        self.output_path_var = tk.StringVar(value=last_output)
        self.overlay_var = tk.BooleanVar(value=last_overlay)
        self.track_label_var = tk.StringVar(value="(no MIDI loaded)")
        self.preview_status_var = tk.StringVar(value="Idle")
        self.run_status_var = tk.StringVar(value="Ready.")
        self.progress_var = tk.DoubleVar(value=0.0)

        self._track_options: list[song.TrackOption] = []
        self._midi_end: float = 0.0
        self._preview_audio: np.ndarray | None = None
        self._preview_sr: int = 44100
        self._preview_started_at: float | None = None
        self._preview_duration: float = 0.0

        self._worker: threading.Thread | None = None
        self._cancel_event = threading.Event()

        self._build_ui()
        # Auto-load MIDI tracks if a preloaded MIDI path is still valid.
        if self.midi_path_var.get():
            self._load_midi(self.midi_path_var.get())
        self._update_ok_state()

        # Centre over parent.
        self.win.update_idletasks()
        px, py = self.root_parent.winfo_x(), self.root_parent.winfo_y()
        pw, ph = self.root_parent.winfo_width(), self.root_parent.winfo_height()
        dw, dh = self.win.winfo_width(), self.win.winfo_height()
        self.win.geometry(f"+{px + (pw - dw) // 2}+{py + (ph - dh) // 2}")
        self.win.grab_set()
        self.win.focus_set()
        # Periodic preview-progress poll.
        self._poll_preview()

    # ---------- UI -----------------------------------------------------
    def _build_ui(self) -> None:
        outer = ttk.Frame(self.win, padding=10)
        outer.pack(fill="both", expand=True)

        files = ttk.LabelFrame(outer, text="Files")
        files.pack(fill="x", pady=(0, 8))

        self._file_row(files, 0, "Voice file", self.voice_path_var, self._browse_voice)
        self._file_row(files, 1, "MIDI file",  self.midi_path_var,  self._browse_midi)
        self._file_row(files, 2, "Output",     self.output_path_var, self._browse_output,
                       placeholder=str(song.default_output_path("", ".wav")))
        files.columnconfigure(1, weight=1)

        # MIDI track selection + preview ------------------------------
        midi = ttk.LabelFrame(outer, text="MIDI track / channel")
        midi.pack(fill="x", pady=(0, 8))

        ttk.Label(midi, text="Track").grid(row=0, column=0, sticky="w", padx=4, pady=4)
        self.track_box = ttk.Combobox(
            midi, textvariable=self.track_label_var, state="disabled", width=60,
        )
        self.track_box.grid(row=0, column=1, columnspan=4, sticky="ew", padx=4, pady=4)
        self.track_box.bind("<<ComboboxSelected>>", lambda _e: self._on_track_changed())

        self.preview_play_btn = ttk.Button(
            midi, text="Play preview", command=self._on_play, state="disabled",
        )
        self.preview_play_btn.grid(row=1, column=0, sticky="w", padx=4, pady=(0, 4))
        self.preview_stop_btn = ttk.Button(
            midi, text="Stop", command=self._on_stop_preview, state="disabled",
        )
        self.preview_stop_btn.grid(row=1, column=1, sticky="w", padx=4, pady=(0, 4))

        self.preview_progress = ttk.Progressbar(
            midi, length=240, mode="determinate", maximum=100,
        )
        self.preview_progress.grid(row=1, column=2, sticky="ew", padx=4, pady=(0, 4))
        ttk.Label(midi, textvariable=self.preview_status_var, width=18)\
            .grid(row=1, column=3, sticky="e", padx=4, pady=(0, 4))
        midi.columnconfigure(2, weight=1)

        # Mix options ------------------------------------------------
        ttk.Checkbutton(
            outer,
            text="Song overlay (mix MIDI into output, looped with the pitch changes)",
            variable=self.overlay_var,
        ).pack(anchor="w", padx=4, pady=(0, 8))

        # Run progress + status --------------------------------------
        run = ttk.LabelFrame(outer, text="Process")
        run.pack(fill="x", pady=(0, 8))
        self.run_progress = ttk.Progressbar(
            run, length=480, mode="determinate", maximum=100,
            variable=self.progress_var,
        )
        self.run_progress.grid(row=0, column=0, columnspan=2, sticky="ew", padx=4, pady=4)
        ttk.Label(run, textvariable=self.run_status_var)\
            .grid(row=1, column=0, columnspan=2, sticky="w", padx=4, pady=2)
        run.columnconfigure(0, weight=1)

        # OK / Cancel ------------------------------------------------
        btns = ttk.Frame(outer)
        btns.pack(fill="x")
        self.ok_btn = ttk.Button(btns, text="OK", command=self._on_ok, state="disabled")
        self.ok_btn.pack(side="right", padx=4)
        self.cancel_btn = ttk.Button(btns, text="Cancel", command=self._on_close)
        self.cancel_btn.pack(side="right", padx=4)

        # React to text changes (e.g. user types a path).
        self.voice_path_var.trace_add("write", lambda *_: self._update_ok_state())
        self.midi_path_var.trace_add("write", lambda *_: self._update_ok_state())

    def _file_row(self, parent, row, label, var, browse_cb, placeholder: str | None = None) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=4, pady=4)
        ent = ttk.Entry(parent, textvariable=var, width=60)
        ent.grid(row=row, column=1, sticky="ew", padx=4, pady=4)
        ttk.Button(parent, text="Browse…", command=browse_cb)\
            .grid(row=row, column=2, sticky="e", padx=4, pady=4)
        if placeholder:
            ent.insert(0, "")  # actual value still bound to var
            # Show greyed placeholder via tooltip-ish label below would be
            # heavier than warranted; keep it simple — empty means default.

    # ---------- File pickers -------------------------------------------
    def _browse_voice(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.win, title="Select voice file", filetypes=_VOICE_EXTS,
        )
        if not path:
            return
        self.voice_path_var.set(path)
        # Suggest a default output path matching the voice stem.
        if not self.output_path_var.get():
            self.output_path_var.set(str(song.default_output_path(path, ".wav")))

    def _browse_midi(self) -> None:
        path = filedialog.askopenfilename(
            parent=self.win, title="Select MIDI file", filetypes=_MIDI_EXTS,
        )
        if not path:
            return
        self.midi_path_var.set(path)
        self._load_midi(path)

    def _browse_output(self) -> None:
        path = filedialog.asksaveasfilename(
            parent=self.win, title="Output file", defaultextension=".wav",
            filetypes=_OUTPUT_EXTS,
            initialfile=Path(self.output_path_var.get()).name
                or (Path(self.voice_path_var.get()).stem + "_tuned.wav"
                    if self.voice_path_var.get() else "tuned.wav"),
        )
        if path:
            self.output_path_var.set(path)

    # ---------- MIDI loading -------------------------------------------
    def _load_midi(self, path: str) -> None:
        try:
            self._midi_end, self._track_options = song.load_midi_options(path)
        except Exception as e:
            messagebox.showerror("Song processing", f"Could not parse MIDI:\n{e}",
                                 parent=self.win)
            self._track_options = []
            self._midi_end = 0.0
            return

        if not self._track_options:
            messagebox.showwarning("Song processing",
                                   "No notes found in this MIDI file.",
                                   parent=self.win)
            return

        labels = [opt.label for opt in self._track_options]
        self.track_box["values"] = labels
        self.track_box.state(["!disabled", "readonly"])
        self.track_box.current(0)
        self.track_label_var.set(labels[0])
        self._on_track_changed()

    def _on_track_changed(self) -> None:
        self._stop_preview_playback()
        # Reset preview render so next Play rebuilds for the new track.
        self._preview_audio = None
        self.preview_progress["value"] = 0
        idx = self.track_box.current()
        if 0 <= idx < len(self._track_options):
            self.preview_play_btn.state(["!disabled"])
            opt = self._track_options[idx]
            self.preview_status_var.set(f"{opt.end_time:5.1f} s")
        else:
            self.preview_play_btn.state(["disabled"])
            self.preview_status_var.set("Idle")

    # ---------- Preview playback ---------------------------------------
    def _on_play(self) -> None:
        idx = self.track_box.current()
        if not (0 <= idx < len(self._track_options)):
            return
        opt = self._track_options[idx]
        if self._preview_audio is None:
            self.preview_status_var.set("Rendering…")
            self.win.update_idletasks()
            self._preview_audio = song.render_midi_preview(opt, sr=self._preview_sr)
            self._preview_duration = (
                len(self._preview_audio) / float(self._preview_sr)
                if self._preview_audio.size else 0.0
            )
        if self._preview_audio is None or self._preview_audio.size == 0:
            self.preview_status_var.set("Empty")
            return
        try:
            sd.stop()
            sd.play(self._preview_audio, samplerate=self._preview_sr)
        except Exception as e:
            messagebox.showerror("Song processing", f"Preview failed:\n{e}",
                                 parent=self.win)
            return
        self._preview_started_at = time.monotonic()
        self.preview_stop_btn.state(["!disabled"])

    def _on_stop_preview(self) -> None:
        self._stop_preview_playback()

    def _stop_preview_playback(self) -> None:
        try:
            sd.stop()
        except Exception:
            pass
        self._preview_started_at = None
        self.preview_progress["value"] = 0
        self.preview_stop_btn.state(["disabled"])
        idx = self.track_box.current()
        if 0 <= idx < len(self._track_options):
            opt = self._track_options[idx]
            self.preview_status_var.set(f"{opt.end_time:5.1f} s")

    def _poll_preview(self) -> None:
        if self._preview_started_at is not None and self._preview_duration > 0:
            elapsed = time.monotonic() - self._preview_started_at
            if elapsed >= self._preview_duration:
                self._stop_preview_playback()
            else:
                pct = 100.0 * elapsed / self._preview_duration
                self.preview_progress["value"] = pct
                self.preview_status_var.set(
                    f"{elapsed:4.1f} / {self._preview_duration:4.1f} s"
                )
        if self.win.winfo_exists():
            self.win.after(100, self._poll_preview)

    # ---------- OK / Cancel + run --------------------------------------
    def _update_ok_state(self) -> None:
        ok = bool(self.voice_path_var.get().strip()) and \
             bool(self.midi_path_var.get().strip())
        if ok:
            self.ok_btn.state(["!disabled"])
        else:
            self.ok_btn.state(["disabled"])

    def _on_ok(self) -> None:
        if self._worker is not None and self._worker.is_alive():
            return
        voice_path = self.voice_path_var.get().strip()
        midi_path = self.midi_path_var.get().strip()
        if not voice_path or not midi_path:
            return
        idx = self.track_box.current()
        if not self._track_options or not (0 <= idx < len(self._track_options)):
            messagebox.showerror("Song processing",
                                 "Select a MIDI track first.", parent=self.win)
            return
        track_option = self._track_options[idx]

        out_path = self.output_path_var.get().strip()
        if not out_path:
            out_path = str(song.default_output_path(voice_path, ".wav"))
            self.output_path_var.set(out_path)

        if Path(out_path).suffix.lower() not in song.OUTPUT_EXTS:
            messagebox.showerror(
                "Song processing",
                f"Output extension must be one of {', '.join(song.OUTPUT_EXTS)}",
                parent=self.win,
            )
            return

        overlay_on = bool(self.overlay_var.get())
        overlay_notes: list[song.MidiNote] | None = None
        if overlay_on:
            overlay_notes = []
            for opt in self._track_options:
                # Skip the synthetic "auto melody" pseudo-option (its notes
                # are duplicates of the per-track ones) and skip drums
                # (sine-wave drums sound nonsensical).
                if opt.is_auto_melody or opt.is_drum_channel:
                    continue
                overlay_notes.extend(opt.notes)
        # Always loop on the full-song duration, not just the chosen track's
        # note range — the user explicitly wants song-length loops.
        full_song_end = self._midi_end if self._midi_end > 0 else None

        # Persist the choices for next time the dialog opens.
        update_config(
            song_last_voice=voice_path,
            song_last_midi=midi_path,
            song_last_output=out_path,
            song_last_overlay=overlay_on,
        )

        self._stop_preview_playback()
        self._cancel_event.clear()
        self.ok_btn.state(["disabled"])
        self.preview_play_btn.state(["disabled"])
        self.track_box.state(["disabled"])
        self.run_status_var.set("Processing…")
        self.progress_var.set(0.0)

        self._worker = threading.Thread(
            target=self._run_worker,
            args=(voice_path, midi_path, track_option, out_path),
            kwargs={"overlay_notes": overlay_notes, "full_song_end": full_song_end},
            daemon=True,
        )
        self._worker.start()

    def _run_worker(self, voice_path, midi_path, track_option, out_path,
                    *, overlay_notes=None, full_song_end=None) -> None:
        def report(f: float) -> None:
            try:
                self.win.after(0, lambda: self.progress_var.set(f * 100.0))
            except Exception:
                pass

        try:
            song.process_song(
                voice_path=voice_path,
                midi_path=midi_path,
                track_option=track_option,
                output_path=out_path,
                overlay_notes=overlay_notes,
                full_song_end=full_song_end,
                progress_callback=report,
                cancel_event=self._cancel_event,
            )
        except Exception as e:
            err = e
            self.win.after(0, lambda: self._worker_done(error=err, out_path=out_path))
            return
        self.win.after(0, lambda: self._worker_done(error=None, out_path=out_path))

    def _worker_done(self, error: Exception | None, out_path: str) -> None:
        self.ok_btn.state(["!disabled"])
        if self._track_options:
            self.track_box.state(["!disabled", "readonly"])
            self.preview_play_btn.state(["!disabled"])
        if self._cancel_event.is_set():
            self.run_status_var.set("Cancelled.")
            return
        if error is not None:
            self.run_status_var.set("Failed.")
            messagebox.showerror("Song processing", str(error), parent=self.win)
            return
        self.run_status_var.set(f"Done → {out_path}")
        self.progress_var.set(100.0)
        messagebox.showinfo("Song processing",
                            f"Wrote:\n{out_path}", parent=self.win)

    def _on_close(self) -> None:
        self._cancel_event.set()
        self._stop_preview_playback()
        try:
            self.win.grab_release()
        except Exception:
            pass
        self.win.destroy()
