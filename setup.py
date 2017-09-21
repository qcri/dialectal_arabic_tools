#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = [
    'numpy',
    'Click>=6.0',
    'keras',
    'tensorflow',
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
    version='0.1.2',
    description="Dialectal Arabic Tools comprises the different modules developed in Qatar Computing Research Institute (QCRI) developed by the ALT team to handle Dialectal Arabic Segmentation, POS tagging, daicrtization and more",
    long_description=readme + '\n\n' + history,
    author="Mohamed Eldesouki",
    author_email='disooqi@gmail.com',
    url='https://github.com/qcri/dialectal_arabic_tools',
    packages=find_packages(include=['dialectal_arabic_tools']),
    entry_points={
        'console_scripts': [
            'dialectal_arabic_tools=dialectal_arabic_tools.cli:main'
        ]
    },
    include_package_data=True,

    zip_safe=False,
    keywords='dialectal_arabic_tools',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    test_suite='tests',
    tests_require=test_requirements,
    setup_requires=setup_requirements,

    dependency_links=['git+git://github.com/phipleg/keras@crf#egg=keras'],
# dependency_links = ['git+https://github.com/liamzebedee/scandir.git#egg=scandir-0.1'],
#     install_requires = ['scandir'],
    install_requires=requirements,
)
