"""Pytest configuration for the Persistent Memory Assistant test suite."""

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--no-camera",
        action="store_true",
        default=False,
        help="Skip tests that require a live IP Webcam stream.",
    )

