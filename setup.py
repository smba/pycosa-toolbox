import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name="pycosa",
    version="0.1.4",
    author="Stefan Mühlbauer",
    author_email="s.muehlbauer@mars.ucc.ie",
    description="Configuration sampling toolbox in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smba/pycosa-toolbox",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)

install_requires = [
    "numpy",
    "pandas",
    "z3-solver",
    "xmltodict",
    "pyeda",
    "networkx",
    "statsmodels",
    "matplotlib",
    "pyDOE2",
    "deprecated",
]

if __name__ == "__main__":
    setuptools.setup(**setup_args, install_requires=install_requires)
