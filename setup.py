import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="spatial_scene_grammars",
    version="0.0.0",
    author="Greg Izatt",
    author_email="gizatt@mit.edu",
    description="Research package for doing inference with scene grammars.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gizatt/spatial_scene_grammars",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: Ubuntu Bionic",
    ],
    python_requires='>=3.6',
)