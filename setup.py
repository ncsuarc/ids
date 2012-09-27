#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension("ids",
                   extra_compile_args = ['-std=gnu99', '-g3'],
                   libraries = ['ueye_api'],
                   sources = ['ids.c', 'ids_methods.c', 'ids_constants.c', 'ids_Camera.c', 'ids_Camera_methods.c', 'ids_Camera_attributes.c'])

setup (name = 'ids',
       version = '0.1',
       description = 'Wrapper for IDS ueye library',
       author = 'Michael Pratt',
       ext_modules = [module])
