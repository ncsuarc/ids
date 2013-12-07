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
#include <datetime.h>
#include <ueye.h>

#define PY_ARRAY_UNIQUE_SYMBOL  ids_core_ARRAY_API
#include <numpy/arrayobject.h>

#include "ids_core.h"

/* IDS Exceptions */
PyObject *IDSError;
PyObject *IDSTimeoutError;
PyObject *IDSCaptureStatus;

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef ids_coremodule = {
    PyModuleDef_HEAD_INIT,
    "ids_core",    /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    ids_coreMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_ids_core(void) {
#else
PyMODINIT_FUNC initids_core(void) {
#endif
    PyObject* m;

    ids_core_CameraType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ids_core_CameraType) < 0) {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    import_array();
    PyDateTime_IMPORT;

#if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&ids_coremodule);
#else
    m = Py_InitModule("ids_core", ids_coreMethods);
#endif

    if (m == NULL) {
#if PY_MAJOR_VERSION >= 3
        return NULL;
#else
        return;
#endif
    }

    Py_INCREF(&ids_core_CameraType);
    PyModule_AddObject(m, "Camera", (PyObject *) &ids_core_CameraType);

    add_constants(m);

    /* IDS Exceptions */
    IDSError = PyErr_NewExceptionWithDoc("ids_core.IDSError",
            "Base class for exceptions caused by an error with the IDS camera or libraries.",
            NULL, NULL);
    Py_INCREF(IDSError);
    PyModule_AddObject(m, "IDSError", IDSError);

    IDSTimeoutError = PyErr_NewExceptionWithDoc("ids_core.IDSTimeoutError",
            "Raised when a camera operation times out.", IDSError, NULL);
    Py_INCREF(IDSTimeoutError);
    PyModule_AddObject(m, "IDSTimeoutError", IDSTimeoutError);

    IDSCaptureStatus = PyErr_NewExceptionWithDoc("ids_core.IDSCaptureStatus",
            "Raised to indicate that a transfer error occured, and that the "
            "capture status can be queried for further information",
            IDSError, NULL);
    Py_INCREF(IDSCaptureStatus);
    PyModule_AddObject(m, "IDSCaptureStatus", IDSCaptureStatus);

#if PY_MAJOR_VERSION >= 3
    return m;
#endif
}

int main(int argc, char *argv[]) {
#if PY_MAJOR_VERSION >= 3
    wchar_t name[128];
    mbstowcs(name, argv[0], 128);
#else
    char name[128];
    strncpy(name, argv[0], 128);
#endif

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(name);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
#if PY_MAJOR_VERSION >= 3
    PyInit_ids_core();
#else
    initids_core();
#endif

    return 0;
}

/* Stupid hack, needs DateTime, which gets clobbered in other files */
PyObject *image_info(ids_core_Camera *self, int image_id) {
    UEYEIMAGEINFO image_info;

    int ret = is_GetImageInfo(self->handle, image_id, &image_info, sizeof(image_info));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }

    PyObject *info = PyDict_New();

    PyDict_SetItemString(info, "device_timestamp", Py_BuildValue("K", image_info.u64TimestampDevice));
    PyDict_SetItemString(info, "timestamp", PyDateTime_FromDateAndTime(image_info.TimestampSystem.wYear, image_info.TimestampSystem.wMonth, image_info.TimestampSystem.wDay, image_info.TimestampSystem.wHour,  image_info.TimestampSystem.wMinute, image_info.TimestampSystem.wSecond, 1000*image_info.TimestampSystem.wMilliseconds));
    PyDict_SetItemString(info, "digital_input", Py_BuildValue("I", image_info.dwIoStatus&4));
    PyDict_SetItemString(info, "gpio1", Py_BuildValue("I", image_info.dwIoStatus&2));
    PyDict_SetItemString(info, "gpio2", Py_BuildValue("I", image_info.dwIoStatus&1));
    PyDict_SetItemString(info, "frame_number", Py_BuildValue("K", image_info.u64FrameNumber));
    PyDict_SetItemString(info, "camera_buffers", Py_BuildValue("I", image_info.dwImageBuffers));
    PyDict_SetItemString(info, "used_camera_buffers", Py_BuildValue("I", image_info.dwImageBuffersInUse));
    PyDict_SetItemString(info, "height", Py_BuildValue("I", image_info.dwImageHeight));
    PyDict_SetItemString(info, "width", Py_BuildValue("I", image_info.dwImageWidth));

    return info;
}

