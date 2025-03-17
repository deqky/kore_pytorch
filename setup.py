from setuptools import setup, find_packages
import sys


if sys.platform == "win32" and sys.maxsize.bit_length() == 31:
    print(
        "32-bit Windows Python runtime is not supported. Please switch to 64-bit Python."
    )
    sys.exit(-1)

import platform


python_min_version = (3, 10, 0)
python_min_version_str = ".".join(map(str, python_min_version))
if sys.version_info < python_min_version:
    print(
        f"You are using Python {platform.python_version()}. Python >={python_min_version_str} is required."
    )
    sys.exit(-1)

setup(
    name="kore-pytorch",
    version="0.1",
    author_email="e.a.koryukin@gmail.com",
    description= "Custom layers for PyTorch library.",
    python_requires=">=3.10",
    packages=find_packages(),
)