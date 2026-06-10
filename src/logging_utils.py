"""Vendored minimal logger — avoids external lib dependency."""
import logging
from typing import Optional


class _Logger:
    """Thin wrapper around stdlib logging.Logger with callable interface."""

    def __init__(self) -> None:
        self._log = logging.getLogger("azul_rl")
        if not self._log.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
            self._log.addHandler(handler)
        self._log.setLevel(logging.DEBUG)

    def __call__(self, message: str, level: Optional[str] = None, indent: int = 0) -> None:
        fn = getattr(self._log, level or "info")
        fn(message)


logger: _Logger = _Logger()
