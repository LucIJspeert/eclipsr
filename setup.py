"""ECLIPSR

Code written by: Luc IJspeert
"""

from setuptools import setup, find_packages


# package version
MAJOR = 1
MINOR = 1
ATTR = '2'

setup(name="eclipsr",
      version=f'{MAJOR}.{MINOR}.{ATTR}',
      author='Luc IJspeert',
      description='Eclipse Candidates in Light curves and Inference of Period at a Speedy Rate',
      long_description=open('README.md').read(),
      url='https://github.com/LucIJspeert/eclipsr',
      license='GNU General Public License v3.0',
      packages=find_packages(),
      package_dir={'eclipsr': 'eclipsr'},
      package_data={'eclipsr': ['data/tess_sectors.dat',
                                'data/random_forrest.dump',
                                'data/sim_000_lc.dat']},
      include_package_data=True,
      install_requires=['numpy',
                        'scipy',
                        'numba',
                        'scikit-learn',
                        'matplotlib',
                        'h5py',
                        'astropy'],
      python_requires=">=3.6")
