# SPDX-FileCopyrightText: 2026 Cacie Prins
# SPDX-License-Identifier: AGPL-3.0-or-later

from __future__ import annotations


class EchoControlError(Exception):
    """Base error for user-facing failures with a CLI exit code."""

    exit_code: int = 1

    def __init__(self, message: str, *, exit_code: int | None = None) -> None:
        super().__init__(message)
        if exit_code is not None:
            self.exit_code = exit_code


class UsageError(EchoControlError):
    """Invalid CLI flags or arguments."""

    exit_code: int = 2

    def __init__(self, message: str = "", *, show_usage: bool = False) -> None:
        super().__init__(message, exit_code=2)
        self.show_usage: bool = show_usage


class ConfigError(EchoControlError):
    """Invalid configuration file or setting."""
