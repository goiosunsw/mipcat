import os
from setuptools import setup

import unittest
def my_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

def package_files(directory):
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

resources = package_files('mipcat/resources')
print(resources)

setup(name='mipcat',
      version='0.1',
      description='Scripts for the analysis of emotion in music recordings',
      url='http://github.com/goiosunw/mipcat',
      author='Andre Almeida',
      author_email='a.almeida@unsw.edu.au',
      license='GPL v3',
      packages=['mipcat', 'mipcat.signal', 'mipcat.video', 'mipcat.align'],
      package_dir={'mipcat': 'mipcat'},
      package_data = {'mipcat': resources},
      test_suite = 'setup.my_tests',
      zip_safe=False)

