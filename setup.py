#!/usr/bin/env python

# Copyright (c) 2012, 2013, North Carolina State University Aerial Robotics Club
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the North Carolina State University Aerial Robotics Club
#       nor the names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from distutils.core import setup, Extension

ids_core = Extension("ids_core",
                     extra_compile_args = ['-std=gnu99', '-g3'],
                     library_dirs = ['/usr/local/lib/'],
                     libraries = ['ueye_api', 'm', 'z', 'tiff'],
                     sources = [
                            'ids_core/ids_core.c',
                            'ids_core/ids_core_methods.c',
                            'ids_core/ids_core_constants.c',
                            'ids_core/ids_core_Camera.c',
                            'ids_core/ids_core_Camera_methods.c',
                            'ids_core/ids_core_Camera_attributes.c',
                            'ids_core/ids_core_color.c',
                     ])

setup(name = 'ids',
      version = '1.0',
      description = 'Interface for IDS machine vision cameras',
      author = 'NC State Aerial Robotics Club',
      license = 'BSD',
      py_modules = ['ids'],
      ext_modules = [ids_core])
