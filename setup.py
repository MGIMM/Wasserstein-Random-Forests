#from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension

sourcefiles = ['./WassersteinRandomForests/wrf.pyx']

extensions = [Extension("WassersteinRandomForests", sourcefiles)]

setup(
    ext_modules=cythonize(extensions),
    name='WassersteinRandomForests',
    version='0.0.2',
    description='A Random Forests variant that estimates the conditional distribution.',
    url='https://github.com/MGIMM/Wasserstein-Random-Forests',
    author='Qiming Du',
    author_email='qiming.du@upmc.fr',
    license='MIT',
    packages=['WassersteinRandomForests'],
    install_requires=[
           'tqdm',
           #'joblib',
           #'seaborn'
       ],
    zip_safe=False
)
