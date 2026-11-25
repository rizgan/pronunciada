"""Record a real shell session as an asciicast (v2) file.

    python scripts/graphics/record_terminal.py out.cast

A bash shell runs in a pseudo-terminal; the command is typed into it key by key and everything
the shell prints is saved with its timing. Nothing in the recording is edited afterwards.
The speech model must already be in the local cache, or the first run shows its download.
"""
from __future__ import annotations

import json
import os
import pty
import select
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
COLS, ROWS = 96, 12
COMMAND = 'pronunciada check en-learner-1.mp3 "It was a beautiful day for a float" --lang en'


def main(out: str) -> None:
    env = dict(os.environ, PS1="$ ", TERM="xterm-256color", COLUMNS=str(COLS), LINES=str(ROWS),
               PATH=f"{ROOT / '.venv/bin'}:{os.environ['PATH']}", PYTHONWARNINGS="ignore", BASH_SILENCE_DEPRECATION_WARNING="1")
    env.pop("NO_COLOR", None)
    pid, fd = pty.fork()
    if pid == 0:
        os.chdir(ROOT / "site/samples")
        os.execvpe("bash", ["bash", "--norc", "--noprofile", "-i"], env)
    import fcntl, struct, termios
    fcntl.ioctl(fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
    events, t0 = [], time.monotonic()

    def pump(seconds: float) -> None:
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            r, _, _ = select.select([fd], [], [], 0.02)
            if r:
                try:
                    data = os.read(fd, 65536)
                except OSError:
                    return
                if data:
                    events.append([round(time.monotonic() - t0, 4), "o", data.decode("utf-8", "replace")])

    pump(0.5)
    for ch in COMMAND:
        os.write(fd, ch.encode())
        pump(0.014)
    pump(0.5)
    os.write(fd, b"\n")
    pump(4.0)                      # stop recording here; the shell is closed off the record
    os.write(fd, b"exit\n")
    header = {"version": 2, "width": COLS, "height": ROWS, "timestamp": int(time.time()), "env": {"TERM": "xterm-256color"}}
    Path(out).write_text("\n".join([json.dumps(header)] + [json.dumps(e, ensure_ascii=False) for e in events]) + "\n", encoding="utf-8")
    print(f"{len(events)} events, {events[-1][0]:.1f} s -> {out}")


if __name__ == "__main__":
    main(sys.argv[1])
