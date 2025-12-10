import os
import subprocess
import tempfile
import zipfile
from pathlib import Path

# ---- Required API -----------------------------------------------------------

def build_wheel(wheel_directory, config_settings=None, metadata_directory=None):
    # 1. Run your Makefile
    subprocess.check_call(["make", "lib"])

    # 2. Create a wheel file
    wheel_name = "gateau-0.1-py3-none-any.whl"
    wheel_path = Path(wheel_directory) / wheel_name

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)

        # Copy your package into the wheel
        pkg_src = Path("gateau")
        pkg_dst = td / "gateau"
        _copytree(pkg_src, pkg_dst)

        # Minimal dist-info metadata
        dist_info = td / "gateau-0.1.dist-info"
        dist_info.mkdir()
        (dist_info / "WHEEL").write_text(
            "Wheel-Version: 1.0\n"
            "Generator: minimal\n"
            "Root-Is-Purelib: true\n"
            "Tag: py3-none-any\n"
        )
        (dist_info / "METADATA").write_text(
            "Metadata-Version: 2.1\n"
            "Name: gateau\n"
            "Version: 0.1\n"
        )

        # Zip it into a wheel
        with zipfile.ZipFile(wheel_path, "w") as whl:
            for path in td.rglob("*"):
                whl.write(path, path.relative_to(td))

    return wheel_name


def build_sdist(sdist_directory, config_settings=None):
    # You can implement this if you want; pip does not require it
    raise NotImplementedError("sdist build not implemented")


# ---- Helpers ----------------------------------------------------------------

def _copytree(src: Path, dst: Path):
    for root, dirs, files in os.walk(src):
        root = Path(root)
        rel = root.relative_to(src)
        (dst / rel).mkdir(parents=True, exist_ok=True)
        for f in files:
            (dst / rel / f).write_bytes((root / f).read_bytes())

