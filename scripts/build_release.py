"""Build a local wheel and source archive with the current environment.

This does not install packages or publish anything. Build dependencies must
already be available, or use `python -m pip wheel .` to install them in an isolated
build environment.
"""


def main():
    import os
    from pathlib import Path
    from tempfile import TemporaryDirectory

    from setuptools.build_meta import build_sdist, build_wheel

    root = Path(__file__).resolve().parents[1]
    os.chdir(root)
    dist = root / "dist"
    dist.mkdir(exist_ok=True)
    with TemporaryDirectory(prefix=".release-", dir=dist) as scratch:
        wheel = build_wheel(scratch)
        sdist = build_sdist(scratch)
        for name in (wheel, sdist):
            (Path(scratch) / name).replace(dist / name)


if __name__ == "__main__":
    main()
