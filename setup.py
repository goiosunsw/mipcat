from setuptools import setup

import unittest
def my_tests():
    test_loader = unittest.TestLoader()
    test_suite = test_loader.discover('tests', pattern='test_*.py')
    return test_suite

setup(name='mipcat',
      version='0.1',
      description='Scripts for the analysis of emotion in music recordings',
      url='http://github.com/goiosunw/mipcat',
      author='Andre Almeida',
      author_email='a.almeida@unsw.edu.au',
      license='GPL v3',
      packages=['mipcat', 'mipcat.signal', 'mipcat.video', 'mipcat.align'],
      package_dir={'mipcat': 'mipcat'},
      package_data = {'mipcat': ['resources/melodies.yaml',
                                 'resources/allruns.yaml',
                                 'resources/file_list.yaml',
                                 'resources/channel_desc.yaml']},
      test_suite = 'setup.my_tests',
      zip_safe=False)

