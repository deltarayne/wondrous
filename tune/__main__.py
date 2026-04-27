"""Entry point: `python -m tune`."""
from __future__ import annotations

from .gui import TuneApp


def main() -> None:
    TuneApp().run()


if __name__ == "__main__":
    main()
