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
#include <structmember.h>
#include <ueye.h>

#include "ids_core.h"

static void ids_core_Camera_dealloc(ids_core_Camera *self);
static int ids_core_Camera_init(ids_core_Camera *self, PyObject *args, PyObject *kwds);

PyMemberDef ids_core_Camera_members[] = {
    {NULL}
};

PyTypeObject ids_core_CameraType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ids_core.Camera",              /* tp_name */
    sizeof(ids_core_Camera),        /* tp_basicsize */
    0,                         /* tp_itemsize */
    (destructor) ids_core_Camera_dealloc,        /* tp_dealloc */
    0,                         /* tp_print */
    0,                         /* tp_getattr */
    0,                         /* tp_setattr */
    0,                         /* tp_reserved */
    0,                         /* tp_repr */
    0,                         /* tp_as_number */
    0,                         /* tp_as_sequence */
    0,                         /* tp_as_mapping */
    0,                         /* tp_hash  */
    0,                         /* tp_call */
    0,                         /* tp_str */
    0,                         /* tp_getattro */
    0,                         /* tp_setattro */
    0,                         /* tp_as_buffer */
    Py_TPFLAGS_DEFAULT,        /* tp_flags */
    "Camera([handle=0, nummem=3, color=ids_core.COLOR_BGRA8]) -> Camera object\n\n"
    "Handle allows selection of camera to connect to.\n"
    "nummem is the number of image memory buffers to create for image storage.\n"
    "Color is the color space in which to store images.\n"
    "This can only be changed after initialization with Camera.color if it won't\n"
    "change the bitdepth, as that would require the size of the memory buffers change.",       /* tp_doc */
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    ids_core_Camera_methods,        /* tp_methods */
    ids_core_Camera_members,        /* tp_members */
    ids_core_Camera_getseters,      /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)ids_core_Camera_init, /* tp_init */
    0,                         /* tp_alloc */
    0,                         /* tp_new */
};

static void ids_core_Camera_dealloc(ids_core_Camera *self) {
    /* Use ready flag to determine state of readiness to deallocate */
    switch (self->ready) {
    case READY:
        is_ExitImageQueue(self->handle);
    case ALLOCATED_MEM:
        free_all_ids_core_mem(self);
    case CONNECTED:
        /* Attempt to close camera */
        is_ExitCamera(self->handle);
    }

    Py_TYPE(self)->tp_free((PyObject*)self);
}

static int ids_core_Camera_init(ids_core_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"width", "height", "handle", "nummem", "color", NULL};
    uint32_t nummem = 3;
    self->handle = 0;
    self->mem = NULL;
    self->bitdepth = 0;
    self->color = IS_CM_BGRA8_PACKED;
    self->autofeatures = 0;
    self->ready = NOT_READY;

    /*
     * This means the definition is:
     * def __init__(self, width, height, handle=0, nummem=3,
     *              color=ids_core.COLOR_BGA8):
     */
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "II|iIi", kwlist,
            &self->width, &self->height, &self->handle, &nummem, &self->color)) {
        return -1;
    }

    int ret = is_InitCamera(&self->handle, NULL);
    switch (ret) {
    case IS_SUCCESS:
        break;
    case IS_CANT_OPEN_DEVICE:
        PyErr_SetString(PyExc_IOError, "Unable to open camera. Camera not connected.");
        return -1;
    case IS_INVALID_HANDLE:
        PyErr_SetString(PyExc_IOError, "Unable to open camera. Invalid camera handle.");
        return -1;
    default:
        PyErr_Format(PyExc_IOError, "Unable to open camera (Error %d).", ret);
        return -1;
    }

    self->ready = CONNECTED;

    if (!set_color_mode(self, self->color)) {
        return -1;
    }

    if (!alloc_ids_core_mem(self, self->width, self->height, nummem)) {
        return -1;
    }

    self->ready = ALLOCATED_MEM;

    /* Initialize image queue so we can WaitForNextImage */
    if (is_InitImageQueue(self->handle, 0) != IS_SUCCESS) {
        PyErr_SetString(PyExc_IOError, "Unable to start image queue.");
        return -1;
    }

    self->ready = READY;

    return 0;
}
