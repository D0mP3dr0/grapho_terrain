from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = fh.read().splitlines()

setup(
    name="grapho_terrain",
    version="0.1.0",
    author="UFABC",
    author_email="author@ufabc.edu.br",
    description="A Python package for geospatial graph analysis of terrain and urban features",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ufabc/grapho_terrain",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: GIS",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
) 