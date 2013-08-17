/*
 * Copyright (c) 2012, 2013, North Carolina State University Aerial Robotics Club
 * All rights reserved.
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the North Carolina State University Aerial Robotics Club
 *       nor the names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Python.h>
#include <ueye.h>

#include "ids.h"

void add_constants(PyObject *m) {
    /* Color */
    PyModule_AddIntConstant(m, "COLOR_BAYER_16", IS_CM_SENSOR_RAW16);
    PyModule_AddIntConstant(m, "COLOR_BAYER_12", IS_CM_SENSOR_RAW12);
    PyModule_AddIntConstant(m, "COLOR_BAYER_8", IS_CM_SENSOR_RAW8);
    PyModule_AddIntConstant(m, "COLOR_MONO_16", IS_CM_MONO16);
    PyModule_AddIntConstant(m, "COLOR_MONO_12", IS_CM_MONO12);
    PyModule_AddIntConstant(m, "COLOR_MONO_8", IS_CM_MONO8);
    PyModule_AddIntConstant(m, "COLOR_RGB10", IS_CM_RGB10_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGBA8", IS_CM_RGBA8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGBY8", IS_CM_RGBY8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_RGB8", IS_CM_RGB8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR10", IS_CM_BGR10_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGRA8", IS_CM_BGRA8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGRY8", IS_CM_BGRY8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR8", IS_CM_BGR8_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR565", IS_CM_BGR565_PACKED);
    PyModule_AddIntConstant(m, "COLOR_BGR5", IS_CM_BGR5_PACKED);
    PyModule_AddIntConstant(m, "COLOR_UYVY", IS_CM_UYVY_PACKED);
    PyModule_AddIntConstant(m, "COLOR_CBYCRY", IS_CM_CBYCRY_PACKED);

    PyModule_AddIntConstant(m, "FILETYPE_JPG", IS_IMG_JPG);
    PyModule_AddIntConstant(m, "FILETYPE_BMP", IS_IMG_BMP);
    PyModule_AddIntConstant(m, "FILETYPE_PNG", IS_IMG_PNG);
}
