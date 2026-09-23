"""Tests for the console logging setup of the command-line tools."""

import logging

import pytest

from ell1fit.logging import NOISY_LIBRARIES, configure_logging


@pytest.fixture
def clean_logging():
    """Restore the global logger state, since ``configure_logging`` changes it."""
    root_logger = logging.getLogger()
    old_handlers = list(root_logger.handlers)
    old_level = root_logger.level
    old_levels = {name: logging.getLogger(name).level for name in NOISY_LIBRARIES}

    yield

    root_logger.handlers = old_handlers
    root_logger.setLevel(old_level)
    for name, level in old_levels.items():
        logging.getLogger(name).setLevel(level)


@pytest.mark.parametrize("name", ["fontTools.subset", "matplotlib.font_manager", "PIL.Image"])
def test_noisy_libraries_are_silenced(clean_logging, name):
    """Chatty dependencies, such as the font subsetter used when saving PDFs, stay quiet."""
    logging.getLogger(name).setLevel(logging.NOTSET)
    configure_logging()

    assert not logging.getLogger(name).isEnabledFor(logging.INFO)
    assert logging.getLogger(name).isEnabledFor(logging.WARNING)


def test_noisy_libraries_stay_silenced_in_debug_mode(clean_logging):
    """Asking for debug output means debug output from us, not from our dependencies."""
    configure_logging(logging.DEBUG)

    assert not logging.getLogger("fontTools.subset").isEnabledFor(logging.INFO)


def test_our_own_messages_still_get_through(clean_logging, caplog):
    """Silencing the dependencies must not silence the package's own INFO messages."""
    configure_logging()

    with caplog.at_level(logging.INFO):
        logging.info("hello")

    assert "hello" in caplog.text
