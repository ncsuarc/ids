#!/usr/bin/env python

from distutils.core import setup, Extension

module = Extension("ids",
                   libraries = ['ueye_api'],
                   sources = ['ids.c'])

setup (name = 'ids',
       version = '0.1',
       description = 'Wrapper for IDS ueye library',
       author = 'Michael Pratt',
       ext_modules = [module])
