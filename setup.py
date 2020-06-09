#from distutils.core import setup
from setuptools import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy

sourcefiles = ['./WassersteinRandomForests/wrf.pyx']

extensions = [Extension("WassersteinRandomForests", sourcefiles,include_dirs=[numpy.get_include()])]

setup(
    ext_modules=cythonize(extensions, compiler_directives={'language_level' : "3"}),
    name='WassersteinRandomForests',
    version='0.1.3',
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
