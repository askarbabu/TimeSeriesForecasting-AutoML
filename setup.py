from setuptools import setup, find_packages
from os import path

from pkg_resources import parse_requirements

HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, "README.md"), "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open(path.join(HERE, "requirements.txt")) as fp:
    install_requires = fp.read()


version_info = (0, 0, 1)
__version__ = ".".join(map(str, version_info))


setup(
    name='tsLQC',

    version=__version__,
    packages=find_packages(),
    url='liquidity-capital.com',
    license='',
    author='askar',
    author_email='askar@liquidity-capital.com',
    description="time series",
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ]
)