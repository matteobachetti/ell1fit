import sys
import logging
from colorama import Fore, Back, Style
from typing import Optional, Dict


class ColoredFormatter(logging.Formatter):
    """Colored log formatter."""

    def __init__(self, *args, colors: Optional[Dict[str, str]] = None, **kwargs) -> None:
        """Initialize the formatter with specified format strings."""

        super().__init__(*args, **kwargs)

        self.colors = colors if colors else {}

    def format(self, record) -> str:
        """Format the specified record as text."""

        record.color = self.colors.get(record.levelname, "")
        record.reset = Style.RESET_ALL

        return super().format(record)


formatter = ColoredFormatter(
    "{color} [{levelname:.1s}] {asctime} {name}:{reset} {message}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
    colors={
        "DEBUG": Fore.CYAN,
        "INFO": Fore.GREEN,
        "WARNING": Fore.YELLOW,
        "ERROR": Fore.RED,
        "CRITICAL": Fore.RED + Back.WHITE + Style.BRIGHT,
    },
)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(formatter)

logger = logging.getLogger("ell1fit")
logger.addHandler(logging.NullHandler())


def configure_logging(level: int = logging.INFO) -> None:
    """Configure root logging for CLI execution.

    This leaves library imports side-effect free and only attaches a console
    handler when explicitly requested by command-line entry points.
    """
    root_logger = logging.getLogger()

    if not any(getattr(h, "_ell1fit_handler", False) for h in root_logger.handlers):
        cli_handler = logging.StreamHandler(sys.stdout)
        cli_handler.setFormatter(formatter)
        cli_handler._ell1fit_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(cli_handler)

    root_logger.setLevel(level)
