import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setup_args = dict(
    name="pycosa",
    version="0.1.6",
    author="Stefan Mühlbauer",
    author_email="s.muehlbauer@mars.ucc.ie",
    description="Configuration sampling toolbox in Python.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/smba/pycosa-toolbox",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: AGPL License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    # ehem
    packages=setuptools.find_packages(where="src"),
    package_dir={"": "src"},
    include_package_data=True,
)

with open("requirements.txt") as f:
    required = f.read().splitlines()

if __name__ == "__main__":
    setuptools.setup(**setup_args, install_requires=required)
