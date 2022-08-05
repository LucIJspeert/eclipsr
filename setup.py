from setuptools import setup, find_packages

setup(name="eclipsr",
    version='0.1',
    author='Luc IJspeert',
    license='GNU General Public License v3.0',
    long_description=open('README.md').read(),
    packages=find_packages(),
    package_data={'': ['tess_sectors.dat']},
    include_package_data=True,
    install_requires=['numpy', 'scipy', 'numba', 'astropy', 'matplotlib', 'h5py'])
