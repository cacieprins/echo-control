# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Plain human-readable daily play log written when tracks pass the listen-time commit gate."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from .config import STATE_DIR


def _local_now() -> datetime:
    return datetime.now(UTC).astimezone()


@dataclass
class DailyPlayLog:
    """Appends one timestamped line per committed track to `<log_dir>/YYYY-MM-DD.log`."""

    log_dir: Path = STATE_DIR / "logs"

    def path_for(self, dt: datetime | None = None) -> Path:
        now = dt or _local_now()
        return self.log_dir / f"{now:%Y-%m-%d}.log"

    def append(
        self,
        track_path: str,
        display_name: str,
        *,
        score: float = 0.0,
        intro_sim: float = 0.0,
        heat: float = 0.0,
        dt: datetime | None = None,
    ) -> Path:
        now = dt or _local_now()
        target = self.path_for(now)
        target.parent.mkdir(parents=True, exist_ok=True)
        clean_name = " ".join(display_name.splitlines()).strip() or Path(track_path).stem
        line = (
            f"{now:%H:%M:%S}  {clean_name}  "
            f"[score={score:.2f} intro={intro_sim:.2f} heat={heat:.2f}]  {track_path}\n"
        )
        with target.open("a", encoding="utf-8") as fh:
            fh.write(line)
        return target

    def count_today(self, dt: datetime | None = None) -> int:
        target = self.path_for(dt)
        if not target.exists():
            return 0
        try:
            return sum(1 for line in target.read_text(encoding="utf-8").splitlines() if line.strip())
        except OSError:
            return 0
