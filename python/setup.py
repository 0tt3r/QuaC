from distutils.core import setup, Extension
import os

module1 = Extension('quac',
                    include_dirs = [ os.environ['PETSC_DIR'] + '/include',
                                     os.environ['PETSC_DIR'] + '/' + os.environ['PETSC_ARCH'] + '/include',
                                     os.getcwd() + '/../src' ],
                    library_dirs = [ os.environ['PETSC_DIR'] + '/' + os.environ['PETSC_ARCH'] + '/lib',
                                     os.getcwd() + '/..' ],
                    libraries = [ 'quac', 'petsc' ],
                    sources = ['toolkit.c'],
                    extra_compile_args=['-std=gnu99'])

setup (name = 'QuaC',
       version = '1.0',
       description = 'QuaC: Time-Dependent Open-Quantum-Systems Solver',
       author = 'Matthew Otten, et al.',
       author_email = 'otten@anl.gov',
       url = 'https://github.com/0tt3r/QuaC',
       ext_modules = [module1])

