"""`tomllib` is 3.11+; this package supports 3.9."""
try:
    import tomllib
except ModuleNotFoundError:                                  # pragma: no cover
    import tomli as tomllib


def load(path):
    return tomllib.loads(path.read_text())
