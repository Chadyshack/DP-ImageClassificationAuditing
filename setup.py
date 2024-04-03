import os
import re

import setuptools

# for simplicity we actually store the version in the __version__ attribute in the source
here = os.path.realpath(os.path.dirname(__file__))
print(here)
with open(os.path.join(here, 'fastDP', '__init__.py')) as f:
    meta_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
    if meta_match:
        version = meta_match.group(1)
    else:
        raise RuntimeError("Unable to find __version__ string.")

with open(os.path.join(here, 'README.md')) as f:
    readme = f.read()

setuptools.setup(
    name="fastDP",
    version=version,
    author="Zhiqi Bu",
    author_email="woodyx218@gmail.com",
    description="Optimally efficient implementation of differentially private optimization (with per-sample gradient clipping.",
    long_description=readme,
    url="",
    packages=setuptools.find_packages(exclude=['examples', 'tests']),
    install_requires=[
        "torch~=1.11.0",
        "prv-accountant",
        "transformers>=4.20.1",
        "numpy",
        "scipy",
        "jupyterlab",
        "jupyter",
        "opacus>=1.0",
        "ml-swissknife",
        "opt_einsum",
        "pytest",
        "pydantic==1.10",
        "tqdm>=4.62.1",
        "deepspeed~=0.8.0",
        "fairscale==0.4"
    ],
    python_requires='~=3.8',
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: Apache Software License",
    ],
)
