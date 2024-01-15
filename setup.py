"""ECLIPSR
"""

from setuptools import setup, find_packages


# package version
MAJOR = 1
MINOR = 0
ATTR = '4'

setup(name="eclipsr",
      version=f'{MAJOR}.{MINOR}.{ATTR}',
      author='Luc IJspeert',
      url='https://github.com/LucIJspeert/eclipsr',
      license='GNU General Public License v3.0',
      description='Eclipse Candidates in Light curves and Inference of Period at a Speedy Rate',
      long_description=open('README.md').read(),
      packages=['eclipsr'],
      package_dir={'eclipsr': 'eclipsr'},
      package_data={'': ['eclipsr/data/tess_sectors.dat', 'eclipsr/data/random_forrest.dump']},
      include_package_data=True,
      install_requires=['numpy', 'scipy', 'numba', 'scikit-learn', 'matplotlib', 'h5py'],
      extras_require={'fits functionality': ['astropy']})
