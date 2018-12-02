from setuptools import setup
from setuptools import find_packages

setup(
    name='ConvIsing',
    version='0.1.0',
    description='Learning RG flow using the classical ising model',
    url='https://github.com/jeffminlin/convising',
    author='Jeffmin Lin',
    author_email='jeffminlin@berkeley.edu',
    packages=find_packages(),
    install_requires=['numpy>=1.9.1',
                      'keras>=2.2.1',
                      'tensorflow',
                      'h5py'],
    extras_require={
        'test': ['pytest']
    },
)
