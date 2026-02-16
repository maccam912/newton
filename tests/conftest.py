"""Shared fixtures for newton tests."""

from __future__ import annotations

import pytest

from newton.config import Config


@pytest.fixture
def cfg() -> Config:
    """A default Config with no external dependencies."""
    return Config()
