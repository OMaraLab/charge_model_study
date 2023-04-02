"""
Charge model study
Study looking at different charge models
"""
import versioneer
import os
import sys
from setuptools import setup, find_packages

sys.path.append(os.path.dirname(__file__))


short_description = "Study looking at different charge models".strip().split("\n")[
    0]

# from https://github.com/pytest-dev/pytest-runner#conditional-requirement
needs_pytest = {'pytest', 'test', 'ptr'}.intersection(sys.argv)
pytest_runner = ['pytest-runner'] if needs_pytest else []

try:
    with open("README.md", "r") as handle:
        long_description = handle.read()
except:
    long_description = "\n".join(short_description[2:])


setup(
    # Self-descriptive entries which should always be present
    name='charge_model_study',
    author='Lily Wang',
    author_email='lily.wang@anu.edu.au',
    description=short_description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    license='GNU Public License v2+',

    # Which Python importable modules should be included when your package is installed
    # Handled automatically by setuptools. Use 'exclude' to prevent some specific
    # subpackage(s) from being added, if needed
    packages=find_packages(),

    # Optional include package data to ship with your package
    # Customize MANIFEST.in if the general case does not suit your needs
    # Comment out this line to prevent the files from being packaged with your software
    include_package_data=True,
    python_requires=">=3.8",          # Python version restrictions
    # Allows `setup.py test` to work correctly with pytest
    setup_requires=[] + pytest_runner,
    # Required packages, pulls from pip if needed
    # do not use for Conda deployment
    install_requires=[
        "pyyaml==6.0",
        "scipy==1.9",
    ],
    # Additional entries you may want simply uncomment the lines you want and fill in the data
    # url='charge_model_study.readthedocs.io/en/latest/',  # Website
    # platforms=['Linux',
    #            'Mac OS-X',
    #            'Unix',
    #            'Windows'],            # Valid platforms your code works on, adjust to your flavor

    # Manual control if final package is compressible or not, set False to prevent the .egg from being made
    # zip_safe=False,

)
