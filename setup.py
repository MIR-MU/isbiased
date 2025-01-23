#!/usr/bin/env python
# -*- coding:utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    readme = fh.read()

setup(
    name="isbiased",
    version='0.0.1',
    description="IsBiased: A library to measure known biases of Question answering models.",
    long_description_content_type="text/markdown",
    long_description=readme,
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Programming Language :: Python :: 3.8"
    ],
    author="Anonymous",
    author_email="tobe@filled.com",
    python_requires=">=3.7",
    license="MIT",
    packages=find_packages(include=["isbiased", "isbiased.*"]),
    include_package_data=True,
    zip_safe=True,
    install_requires=[
        "datasets==2.2.1",
        "ipykernel",
        "ipython",
        "jupyter-client",
        "jupyter-core",
        "matplotlib-inline",
        "nltk",
        "numpy==1.26.4",
        "pandas==1.3.5",
        "scikit-learn",
        "scipy==1.15.1",
        "spacy",
        "torch>=1.11.0",
        "transformers==4.28.1",
        "adaptor",
        "accelerate>=0.26.0",
        "dill<0.3.5",
        "wandb"
    ],
    test_require=[
        "pytest"
    ],
)
