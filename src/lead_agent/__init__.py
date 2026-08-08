"""JJ Miller & Co. Facebook Lead Agent."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("jjmiller-facebook-lead-agent")
except PackageNotFoundError:  # pragma: no cover - source tree without an editable install
    __version__ = "0.0.0"

__all__ = ["__version__"]
