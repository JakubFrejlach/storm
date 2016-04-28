#!/usr/bin/env python
from setuptools import setup
from distutils.core import Extension
from distutils.command.build_ext import build_ext
import os.path
import platform
from glob import glob

PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))

# Glob source files for modules
core_sources = glob(os.path.join('src', 'core', '*.cpp'))
info_sources = glob(os.path.join('src', 'info', '*.cpp'))

# Configuration shared between external modules follows

# To help along, if storm and/or pybind is not system installed, retrieve from storm distribution
include_dirs = ['.', 'src', 'resources/pybind11/include']
local_storm_path = os.path.join(PROJECT_DIR, '..')
if os.path.exists(local_storm_path):
    include_dirs.append(local_storm_path)

# Like includes, also add local path for library, assuming made in 'build'
library_dirs = []
local_storm_lib_path = os.path.join(PROJECT_DIR, '..', 'build/src')
if os.path.exists(local_storm_lib_path):
    library_dirs.append(local_storm_lib_path)

libraries = ['storm']
extra_compile_args = ['-std=c++11']
define_macros = []

extra_link_args = []
if platform.system() == 'Darwin':
    extra_link_args.append('-Wl,-rpath,'+library_dirs[0])

ext_core = Extension(
    name='core',
    sources=['src/mod_core.cpp'] + core_sources,
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
    extra_link_args=extra_link_args
)

ext_info = Extension(
    name='info.info',
    sources=['src/mod_info.cpp'] + info_sources,
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    extra_compile_args=extra_compile_args,
    define_macros=define_macros,
    extra_link_args=extra_link_args
)

class stormpy_build_ext(build_ext):
    """Extend build_ext to provide CLN toggle option
    """
    user_options = build_ext.user_options + [
        ('use-cln', None,
         "use cln numbers instead of gmpxx")
        ]

    def __init__(self, *args, **kwargs):
        build_ext.__init__(self, *args, **kwargs)

    def initialize_options (self):
        build_ext.initialize_options(self)
        self.use_cln = None

    def finalize_options(self):
        build_ext.finalize_options(self)

        if self.use_cln:
            self.libraries += ['cln']
            if not self.define:
                self.define = []
            else:
                self.define = list(self.define)
            self.define += [('STORMPY_USE_CLN', 1)]
        else:
            self.libraries += ['gmp', 'gmpxx']
            if not self.undef:
                self.undef = []
            self.undef += ['STORMPY_USE_CLN']

        if library_dirs:
            # Makes local storm library lookup that much easier
            self.rpath += library_dirs

setup(name="stormpy",
      version="0.9",
      author="M. Volk",
      author_email="matthias.volk@cs.rwth-aachen.de",
      maintainer="S. Junges",
      maintainer_email="sebastian.junges@cs.rwth-aachen.de",
      url="http://moves.rwth-aachen.de",
      description="stormpy - Python Bindings for Storm",
      packages=['stormpy', 'stormpy.info'],
      package_dir={'':'lib'},
      ext_package='stormpy',
      ext_modules=[ext_core, ext_info
                   ],
      cmdclass={
        'build_ext': stormpy_build_ext,
      }
)
