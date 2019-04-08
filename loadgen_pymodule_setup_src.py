"""MLPerf Inference LoadGen python bindings.

Creates a module that python can import.
All source files are compiled by python's C++ toolchain  without depending
on a loadgen lib.
"""

from setuptools import setup, Extension

sources = [
  "bindings/c_api.cc",
  "bindings/python_api.cc",
  "parse_command_line.cc",
  "loadgen.cc",
]

mlpi_loadgen_module = Extension('mlpi_loadgen',
                    define_macros = [('MAJOR_VERSION', '0'),
                                     ('MINOR_VERSION', '5')],
                    include_dirs = [ 'gen' ],
                    sources = [ "gen/loadgen/" + s for s in sources ])

setup (name = 'mlpi_loadgen',
       version = '0.5a0',
       description = 'MLPerf Inference LoadGen python bindings',
       url = 'https://mlperf.org',
       ext_modules = [mlpi_loadgen_module])