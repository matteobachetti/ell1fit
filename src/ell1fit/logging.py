"""Console logging for the command-line tools.

Deliberately inert on import. A library that reconfigures logging when imported
takes that decision away from whatever application imported it, so the module
level only registers a ``NullHandler`` on the package's own namespaced logger.
:func:`configure_logging` attaches the coloured console handler, and only the
CLI entry points call it.

The handler it adds is tagged so that repeated calls do not stack up duplicate
handlers, which would print every message several times.

Because that handler sits on the root logger, chatty third-party libraries would
otherwise print through it too. :data:`NOISY_LIBRARIES` lists the ones that
report routine internal work at ``INFO`` -- ``fontTools`` emits a line per font
table every time matplotlib subsets fonts for a PDF -- and they are held at
``WARNING`` so that only real problems from them reach the console.
"""

import sys
import logging
from colorama import Fore, Back, Style
from typing import Optional, Dict


__all__ = [
    "ColoredFormatter",
    "NOISY_LIBRARIES",
    "configure_logging",
]


#: Third-party loggers whose ``INFO`` output is internal bookkeeping, not news.
NOISY_LIBRARIES = (
    "fontTools",
    "matplotlib",
    "PIL",
)


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

    The libraries in :data:`NOISY_LIBRARIES` are held at ``WARNING`` whatever
    ``level`` is asked for: more detail means more detail about *this* package,
    not about matplotlib's font machinery.
    """
    root_logger = logging.getLogger()

    if not any(getattr(h, "_ell1fit_handler", False) for h in root_logger.handlers):
        cli_handler = logging.StreamHandler(sys.stdout)
        cli_handler.setFormatter(formatter)
        cli_handler._ell1fit_handler = True  # type: ignore[attr-defined]
        root_logger.addHandler(cli_handler)

    root_logger.setLevel(level)

    for name in NOISY_LIBRARIES:
        logging.getLogger(name).setLevel(logging.WARNING)
