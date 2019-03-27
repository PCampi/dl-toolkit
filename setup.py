"""Setup module for installation via pip or conda."""

import setuptools

with open('./README.md', 'r') as fh:
    long_desc = fh.read()

setuptools.setup(
    name="optirisk-finance-dl-toolkit",
    version='0.1.0',
    author='Pietro Mattia Campi',
    author_email='pietro.campi.21@gmail.com',
    description=
    'Stock market time series preprocessing and forecasting using pandas, numpy and keras',
    long_description=long_desc,
    long_description_content_type='text/markdown',
    url='https://github.com/PieCampi/dl-toolkit',
    packages=setuptools.find_packages('src'),
    package_dir={'': 'src'},
    install_requires=['pandas>=0.23', 'scikit-learn>=0.20'],
    setup_requires=['pytest-runner'],
    tests_require=['pytest'],
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Science/Research",
    ],
    keywords=['machine learning', 'finance', 'data science'])
