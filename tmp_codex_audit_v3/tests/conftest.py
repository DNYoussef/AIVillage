def pytest_ignore_collect(path, config):
    return "experimental" in str(path)
