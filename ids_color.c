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

int color_to_bitdepth(int color) {
    switch (color) {
    case IS_CM_SENSOR_RAW8:
    case IS_CM_MONO8:
        return 8;
    case IS_CM_SENSOR_RAW12:
    case IS_CM_SENSOR_RAW16:
    case IS_CM_MONO12:
    case IS_CM_MONO16:
        return 16;
    case IS_CM_BGR8_PACKED:
    case IS_CM_RGB8_PACKED:
        return 24;
    default:
        return 32;
    }
}

PyObject *set_color_mode(ids_Camera *self, int color) {
    int ret = is_SetColorMode(self->handle, color);
    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_INVALID_COLOR_FORMAT:
        PyErr_SetString(PyExc_ValueError, "Unsupported color format.");
        return NULL;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to set color mode.");
        return NULL;
    }

    /* Set object bit depth */
    self->bitdepth = color_to_bitdepth(color);

    Py_INCREF(Py_True);
    return Py_True;
}
