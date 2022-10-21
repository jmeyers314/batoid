import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--skip_gha",
        action="store_true",
        default=False,
        help="skip certain tests for github actions"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "skip_gha: mark test as skippable in github actions"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip_gha"):
        # --skip_gha given in cli: skip problematic tests
        skip_gha = pytest.mark.skip(reason="omit --skip_gha option to run")
        for item in items:
            if "skip_gha" in item.keywords:
                item.add_marker(skip_gha)
