# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='tusha',
    version='0.1.0',
    description='tusha library',
    long_description=readme,
    author='Guy Yosiphon',
    author_email='guyyos@gmail.com',
    url='https://github.com/guyyos/tusha',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

