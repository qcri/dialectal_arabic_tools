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
    'tensorflow',
    'h5py',
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
    version='0.0.1a9',
    description="Dialectal Arabic Tools comprises the different modules developed in Qatar Computing Research Institute (QCRI) developed by the ALT team to handle Dialectal Arabic Segmentation, POS tagging, daicrtization and more",
    long_description=readme + '\n\n' + history,
    author="Mohamed Eldesouki",
    author_email='disooqi@gmail.com',
    url='https://github.com/qcri/dialectal_arabic_tools',
    packages=find_packages(include=['dialectal_arabic_tools']),
    #package_dir={'mypkg': 'src/mypkg'},
    # https://setuptools.readthedocs.io/en/latest/setuptools.html#including-data-files
    # If using the setuptools-specific include_package_data argument, files specified by package_data will not be
    # automatically added to the manifest unless they are listed in the MANIFEST.in file.)
    # package_data={'dialectal_arabic_tools': ['data/*']},
    entry_points={
        'console_scripts': [
            'dialectal_arabic_tools=dialectal_arabic_tools.cli:main'
        ]
    },
    include_package_data=True,

    # data_files=[('bitmaps', ['bm/b1.gif', 'bm/b2.gif']),
    #               ('config', ['cfg/data.cfg']),
    #               ('/etc/init.d', ['init-script'])],

    zip_safe=True,
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

    #http://peak.telecommunity.com/DevCenter/setuptools#dependencies-that-aren-t-in-pypi

    dependency_links=[
        # git+git://github.com/phipleg/keras@crf#egg=keras,
        # 'https://github.com/phipleg/keras/tree/crf#egg=keras-crf-2.0.6',
        # 'https://github.com/phipleg/keras/archive/crf.zip#egg=keras-99.99',
                      ],
    # dependency_links = ['git+https://github.com/liamzebedee/scandir.git#egg=scandir-0.1'],
    #     install_requires = ['scandir'],
    install_requires=requirements,
)
