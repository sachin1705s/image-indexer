"""Terminal folder selection helpers."""

from __future__ import annotations

import os
import sys
import termios
import tty
from dataclasses import dataclass
from pathlib import Path

import typer


@dataclass(frozen=True)
class PickerRow:
    """One selectable row in the terminal folder picker."""

    action: str
    label: str
    path: Path | None = None
    selectable: bool = False


def _shortcut_paths() -> dict[str, Path]:
    home = Path.home()
    return {
        "home": home,
        "desktop": home / "Desktop",
        "documents": home / "Documents",
        "downloads": home / "Downloads",
        "pictures": home / "Pictures",
    }


def _visible_directories(current: Path) -> list[Path]:
    return sorted(
        [
            item
            for item in current.iterdir()
            if item.is_dir() and not item.name.startswith(".")
        ],
        key=lambda item: item.name.lower(),
    )


def _build_rows(current: Path, selected: list[Path]) -> list[PickerRow]:
    rows = [PickerRow("current", f"Current folder: {current}", current, True)]
    if selected:
        rows.append(PickerRow("done", f"Done - index {len(selected)} selected folder(s)"))
    if current.parent != current:
        rows.append(PickerRow("open", "..", current.parent))

    for name, path in _shortcut_paths().items():
        if path.exists() and path.is_dir() and path.resolve() != current:
            rows.append(PickerRow("open", f"{name}: {path}", path.resolve(), True))

    rows.extend(PickerRow("open", item.name, item.resolve(), True) for item in _visible_directories(current))
    return rows


def _toggle_selected(selected: list[Path], path: Path) -> None:
    resolved = path.resolve()
    if resolved in selected:
        selected.remove(resolved)
    else:
        selected.append(resolved)


def _render_picker(current: Path, selected: list[Path], rows: list[PickerRow], cursor: int) -> None:
    sys.stdout.write("\033[2J\033[H")
    typer.echo("Pick folders to index")
    typer.echo("")
    typer.echo(f"Current: {current}")
    typer.echo(f"Selected: {len(selected)} folder(s)")
    if selected:
        for item in selected:
            typer.echo(f"  - {item}")
    typer.echo("")
    typer.echo("Use ↑/↓ to move, Space=check/uncheck, Enter/→=open folder, ←=up, d=done, q=cancel")
    typer.echo("")

    terminal_height = os.get_terminal_size().lines if sys.stdout.isatty() else 30
    max_rows = max(8, terminal_height - 10)
    start = 0
    if cursor >= max_rows:
        start = cursor - max_rows + 1
    visible_rows = rows[start : start + max_rows]

    for offset, row in enumerate(visible_rows, start=start):
        marker = ">" if offset == cursor else " "
        checkbox = ""
        if row.selectable and row.path:
            checkbox = "[✓] " if row.path.resolve() in selected else "[ ] "
        typer.echo(f"{marker} {checkbox}{row.label}")

    if start + max_rows < len(rows):
        typer.echo(f"  ... {len(rows) - (start + max_rows)} more")
    sys.stdout.flush()


def _read_key() -> str:
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        char = sys.stdin.read(1)
        if char == "\x1b":
            char += sys.stdin.read(2)
        return char
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


def _prompt_picker(start_path: Path | None = None) -> list[str]:
    """Fallback command picker for non-interactive shells and tests."""

    current = (start_path or Path.home()).expanduser().resolve()
    selected: list[Path] = []

    while True:
        directories = _visible_directories(current)
        typer.echo("")
        typer.echo(f"Current folder: {current}")
        typer.echo("Selected folders:")
        if selected:
            for item in selected:
                typer.echo(f"  - {item}")
        else:
            typer.echo("  (none)")
        typer.echo("")
        typer.echo("Folders:")
        if not directories:
            typer.echo("  (no visible subfolders)")
        else:
            for index, item in enumerate(directories[:20], start=1):
                typer.echo(f"  {index}. {item.name}")
            if len(directories) > 20:
                typer.echo(f"  ... {len(directories) - 20} more")
        typer.echo("")
        typer.echo("Commands: number=open, a=toggle current, s <n>=toggle listed folder, ..=up, d=done, q=cancel")
        if not directories:
            typer.echo("Tip: press Enter here to select this folder immediately.")

        command = typer.prompt("folder-picker").strip()
        if command == "" and not directories:
            if current not in selected:
                selected.append(current)
            return [str(path) for path in selected]
        if command in {"q", "quit", "exit"}:
            return []
        if command in {"done", "d"}:
            return [str(path) for path in selected]
        if command in {"add", "a", "."}:
            _toggle_selected(selected, current)
            continue
        if command == "..":
            if current.parent != current:
                current = current.parent
            continue
        if command in _shortcut_paths():
            candidate = _shortcut_paths()[command]
            if candidate.exists() and candidate.is_dir():
                current = candidate.resolve()
            else:
                typer.echo("That shortcut folder does not exist on this machine.")
            continue
        if command.startswith("pick ") or command.startswith("s "):
            raw_index = command.split(" ", 1)[1].strip()
            if raw_index.isdigit():
                index = int(raw_index) - 1
                if 0 <= index < len(directories):
                    _toggle_selected(selected, directories[index])
            continue
        if command.isdigit():
            index = int(command) - 1
            if 0 <= index < len(directories):
                current = directories[index].resolve()
            continue

        typer.echo("Unrecognized command.")


def pick_folders(start_path: Path | None = None) -> list[str]:
    """Interactive terminal navigator that supports multi-select."""

    if not sys.stdin.isatty():
        return _prompt_picker(start_path)

    current = (start_path or Path.home()).expanduser().resolve()
    selected: list[Path] = []
    cursor = 0

    while True:
        rows = _build_rows(current, selected)
        if cursor >= len(rows):
            cursor = max(0, len(rows) - 1)
        _render_picker(current, selected, rows, cursor)
        key = _read_key()

        if key in {"q", "Q", "\x03"}:
            sys.stdout.write("\033[2J\033[H")
            return []
        if key in {"d", "D"}:
            sys.stdout.write("\033[2J\033[H")
            return [str(path) for path in selected]
        if key in {"\x1b[A", "k"}:
            cursor = max(0, cursor - 1)
            continue
        if key in {"\x1b[B", "j"}:
            cursor = min(len(rows) - 1, cursor + 1)
            continue
        if key in {"\x1b[D", "h", "\x7f"}:
            if current.parent != current:
                current = current.parent
                cursor = 0
            continue
        if key == " ":
            row = rows[cursor]
            if row.selectable and row.path:
                _toggle_selected(selected, row.path)
            continue
        if key in {"\r", "\n", "\x1b[C"}:
            row = rows[cursor]
            if row.action == "done":
                sys.stdout.write("\033[2J\033[H")
                return [str(path) for path in selected]
            if row.action == "open" and row.path:
                current = row.path.resolve()
                cursor = 0
                continue
