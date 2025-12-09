"""
Setup configuration for VertiPaq Optimizer
"""

from setuptools import setup, find_packages
import os

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read version from module
version = "1.0.0"

setup(
    name="vertipaq-optimizer",
    version=version,
    author="Igor Cotruta",
    description="Row ordering optimizer for columnar compression (VertiPaq-style)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Hugoberry/vertipaq-optimizer",
    py_modules=["vertipaq_optimizer"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    keywords="vertipaq powerbi compression columnar rle optimization parquet",
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.0",
        "pyarrow>=14.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-benchmark>=3.4",
            "black>=21.0",
            "flake8>=3.9",
        ],
    },
    project_urls={
        "Bug Reports": "https://github.com/Hugoberry/vertipaq-optimizer/issues",
        "Source": "https://github.com/Hugoberry/vertipaq-optimizer",
        "Documentation": "https://github.com/Hugoberry/vertipaq-optimizer#readme",
    },
)
