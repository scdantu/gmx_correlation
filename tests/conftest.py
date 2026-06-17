"""pytest configuration for gmx_correlation test suite."""
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: tests that run KNN on large N")
