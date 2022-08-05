from distutils.core import setup

setup(
    name='eclipsr',
    version='0.1',
    author='Thaddaeus Kiker',
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    install_requires = ['numpy', 'scipy', 'numba', 'astropy', 'matplotlib', 'h5py']
)