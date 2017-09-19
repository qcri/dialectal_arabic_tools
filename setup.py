#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'Click>=6.0',
    # TODO: put package requirements here
]

setup_requirements = [
    # TODO(disooqi): put setup requirements (distutils extensions, etc.) here
]

test_requirements = [
    # TODO: put package test requirements here
]

setup(
    name='dialectal_arabic_tools',
    version='0.1.0',
    description="Dialectal Arabic Tools comprises the different modules developed in Qatar Computing Research Institute (QCRI) developed by the ALT team to handle Dialectal Arabic Segmentation, POS tagging, daicrtization and more",
    long_description=readme + '\n\n' + history,
    author="Mohamed Eldesouki",
    author_email='disooqi@gmail.com',
    url='https://github.com/disooqi/dialectal_arabic_tools',
    packages=find_packages(include=['dialectal_arabic_tools']),
    entry_points={
        'console_scripts': [
            'dialectal_arabic_tools=dialectal_arabic_tools.cli:main'
        ]
    },
    include_package_data=True,
    install_requires=requirements,
    zip_safe=False,
    keywords='dialectal_arabic_tools',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,
)
