"""Offline WhatsApp chat export parser.

This connector parses WhatsApp chat exports produced by the mobile
application.  The exports are text files that may be provided directly or
inside a ``.zip`` archive.  Only offline parsing is supported – no network
requests are made.

Example chat line used by WhatsApp::

    12/31/20, 12:34 PM - Alice: Happy New Year!

The parser extracts a timestamp, sender and message text for each line
matching this format.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import io
import re
import zipfile
from pathlib import Path
from typing import Iterable, List, Dict


_LINE_RE = re.compile(
    r"^(\d{1,2}/\d{1,2}/\d{2,4}), (\d{1,2}:\d{2}(?:\s?[AP]M)?) - ([^:]+): (.*)$"
)


@dataclass
class Message:
    timestamp: datetime
    sender: str
    text: str


class WhatsAppConnector:
    """Parser for exported WhatsApp chat histories."""

    def __init__(self, export_path: str | None = None) -> None:
        self._messages: List[Message] = []
        if export_path:
            self.load_export(export_path)

    # ------------------------------------------------------------------
    # Loading and parsing
    def load_export(self, export_path: str | Path) -> int:
        """Load a WhatsApp chat export.

        The export can be a plain text file or a ``.zip`` archive produced by
        WhatsApp.  Parsed messages are stored on the instance and the number of
        messages parsed is returned.
        """
        self._messages.clear()
        path = Path(export_path)
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as zf:
                txt_name = next(
                    (n for n in zf.namelist() if n.lower().endswith(".txt")), None
                )
                if txt_name is None:
                    return 0
                data = zf.read(txt_name).decode("utf-8", errors="ignore")
                self._parse_lines(io.StringIO(data))
        else:
            with path.open("r", encoding="utf-8") as fh:
                self._parse_lines(fh)
        return len(self._messages)

    def _parse_lines(self, lines: Iterable[str]) -> None:
        for line in lines:
            line = line.strip()
            m = _LINE_RE.match(line)
            if not m:
                continue
            date_str, time_str, sender, text = m.groups()
            # Determine format; WhatsApp may use 2 or 4 digit years
            year_fmt = "%y" if len(date_str.split("/")[-1]) == 2 else "%Y"
            if "AM" in time_str or "PM" in time_str:
                fmt = f"%m/%d/{year_fmt}, %I:%M %p"
            else:
                fmt = f"%m/%d/{year_fmt}, %H:%M"
            timestamp = datetime.strptime(f"{date_str}, {time_str}", fmt)
            self._messages.append(Message(timestamp, sender, text))

    # ------------------------------------------------------------------
    # Accessors
    def get_messages(self) -> List[Dict[str, str]]:
        """Return parsed messages as dictionaries."""
        return [
            {"timestamp": m.timestamp, "sender": m.sender, "text": m.text}
            for m in self._messages
        ]

    def get_message_count(self) -> int:
        """Number of parsed messages."""
        return len(self._messages)


# Convenience functions retained for backwards compatibility -----------------

def load_export(path: str | Path) -> WhatsAppConnector:
    conn = WhatsAppConnector()
    conn.load_export(path)
    return conn


def get_messages(path: str | Path) -> List[Dict[str, str]]:
    return load_export(path).get_messages()


def get_message_count(path: str | Path) -> int:
    return load_export(path).get_message_count()
