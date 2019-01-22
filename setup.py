from os.path import exists
from setuptools import setup

extras_require = {
  'mdlab': ['scipy >= 1.8.0'],
}
extras_require['complete'] = sorted(set(sum(extras_require.values(), [])))

setup(name='mdlab',
      version='0.1',
      description='Tools, functions, and scripts for common tasks, especially those relating to McDermott lab projects.',
      long_description=open('README.md').read() if exists('README.md') else '',
      # url='http://github.com/dask/dask/',
      license='MIT',
      maintainer='Ray Gonzalez',
      maintainer_email='raygon@mit.edu',
      packages=['mdlab'],
      extras_require=extras_require,
      zip_safe=False)
