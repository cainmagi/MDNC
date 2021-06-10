#!python
# -*- coding: UTF-8 -*-
'''
################################################################
# Installation
# @ Modern Deep Network Toolkits for pyTorch
# Yuchen Jin @ cainmagi@gmail.com
# Requirements: (Pay attention to version)
#   python 3.5+
#   pyTorch 1.0.0+
# This module provides the fundamental `sequence` engine for
# pyTorch.
################################################################
'''
import setuptools

with open('README.md', 'r') as fh:
    LONG_DESCRIPTION = fh.read()

INSTALL_REQUIRES = [
    'numpy>=1.13.0',
    'scipy>=1.0.0',
    'h5py>=3.1.0',
    'matplotlib>=3.1.1',
    'urllib3>=1.26.2',
    'tqdm>=4.50.2'
]

setuptools.setup(
    name='MDNC',
    version='0.1.6',
    author='Yuchen Jin',
    author_email='cainmagi@gmail.com',
    description='Modern Deep Network Toolkits for PyTorch. This is a extension for PyTorch 1.x.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/cainmagi/MDNC',
    project_urls={
        'Bug Tracker': 'https://github.com/cainmagi/MDNC/issues',
    },
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    license='MIT',
    keywords=[
        'pytorch', 'pytorh-1', 'pytorch-framework', 'pytorch-extension', 'deep-learning', 'machine-learning',
        'python', 'python3', 'python-library'
    ],
    packages=setuptools.find_packages(),
    install_requires=INSTALL_REQUIRES,
    python_requires='>=3.5',
)
