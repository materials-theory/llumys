import io
from setuptools import find_packages, setup, Extension
from llumys.gnn import __version__

# Read in the README for the long description on PyPI
def long_description():
    with io.open('README.md', 'r', encoding='utf-8') as f:
        readme = f.read()
    return readme

setup(name = 'llumys',
      version = __version__,
      description='',
      long_description=long_description(),
      url='https://github.com/materials-theory/llumys',
      author='Giyeok Lee',
      author_email='giyeok.lee@sydney.edu.au',
      license='MIT',
      packages=find_packages(),
    #   package_data={'':["elements_vesta.ini"]},
      include_package_data = True,
      zip_safe = False,
      keywords='MLIP Learning-loss GNN DFT ASE',
      classifiers=[
          'Programming Language :: Python :: 3',
          'Programming Language :: Python :: 3.10',
          'Programming Language :: Python :: 3.11',
          'Programming Language :: Python :: 3.12'
          ],
      install_requires=['ase', 'numpy', 'torch', 'e3nn'],
    #   entry_points = {'console_scripts':['llumys = llumys.main:main']}
      )