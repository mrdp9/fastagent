import pytest

# pytest-asyncio configuration: treat every async test as asyncio mode auto.
def pytest_collection_modifyitems(config, items):
    for item in items:
        if "asyncio" in item.keywords:
            continue
