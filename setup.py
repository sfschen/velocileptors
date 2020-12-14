import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="velocileptors",
    version="1.0",
    author="Stephen Chen",
    author_email="shifan_chen@berkeley.edu",
    description="Code for cosmological perturbation theory",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/sfschen/velocileptors",
    packages=['velocileptors'],  #setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    install_requires=['numpy','scipy'],
)
