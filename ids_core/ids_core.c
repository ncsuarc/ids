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

/*
 * Convert UEYETIME to UTC struct tm
 *
 * UEYETIME provides a timestamp in local time, convert it to UTC, and return
 * the time as a struct tm.
 *
 * @param timestamp Pointer to UEYETIME timestamp
 * @param dest  Destination struct tm for UTC time
 */
static void timestamp_to_utc(UEYETIME *timestamp, struct tm *dest) {
    struct tm local;
    struct tm *utc;
    time_t utctime;

    local.tm_year = timestamp->wYear - 1900;
    local.tm_mon = timestamp->wMonth - 1;
    local.tm_mday = timestamp->wDay;
    local.tm_hour = timestamp->wHour;
    local.tm_min = timestamp->wMinute;
    local.tm_sec = timestamp->wSecond;
    local.tm_isdst = -1;  /* Automatically determine */

    /* mktime ignores tm_wday and tm_wyear */
    utctime = mktime(&local);
    utc = gmtime(&utctime);

    memcpy(dest, utc, sizeof(*dest));
}

/* Stupid hack, needs DateTime, which gets clobbered in other files */
PyObject *image_info(ids_core_Camera *self, int image_id) {
    UEYEIMAGEINFO image_info;
    struct tm utc_timestamp;

    int ret = is_GetImageInfo(self->handle, image_id, &image_info, sizeof(image_info));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        raise_general_error(self, ret);
        return NULL;
    }

    timestamp_to_utc(&image_info.TimestampSystem, &utc_timestamp);

    PyObject *info = PyDict_New();

    PyObject *device_timestamp =
        Py_BuildValue("K", image_info.u64TimestampDevice);
    PyObject *timestamp = /* Assume milliseconds don't change across timezone */
        PyDateTime_FromDateAndTime(utc_timestamp.tm_year + 1900,
                                   utc_timestamp.tm_mon + 1,
                                   utc_timestamp.tm_mday,
                                   utc_timestamp.tm_hour,
                                   utc_timestamp.tm_min,
                                   utc_timestamp.tm_sec,
                                   1000*image_info.TimestampSystem.wMilliseconds);
    PyObject *digital_input = Py_BuildValue("I", image_info.dwIoStatus&4);
    PyObject *gpio1 = Py_BuildValue("I", image_info.dwIoStatus&2);
    PyObject *gpio2 = Py_BuildValue("I", image_info.dwIoStatus&1);
    PyObject *frame_number = Py_BuildValue("K", image_info.u64FrameNumber);
    PyObject *camera_buffers = Py_BuildValue("I", image_info.dwImageBuffers);
    PyObject *used_camera_buffers =
        Py_BuildValue("I", image_info.dwImageBuffersInUse);
    PyObject *height = Py_BuildValue("I", image_info.dwImageHeight);
    PyObject *width = Py_BuildValue("I", image_info.dwImageWidth);

    PyDict_SetItemString(info, "device_timestamp", device_timestamp);
    PyDict_SetItemString(info, "timestamp", timestamp);
    PyDict_SetItemString(info, "digital_input", digital_input);
    PyDict_SetItemString(info, "gpio1", gpio1);
    PyDict_SetItemString(info, "gpio2", gpio2);
    PyDict_SetItemString(info, "frame_number", frame_number);
    PyDict_SetItemString(info, "camera_buffers", camera_buffers);
    PyDict_SetItemString(info, "used_camera_buffers", used_camera_buffers);
    PyDict_SetItemString(info, "height", height);
    PyDict_SetItemString(info, "width", width);

    Py_DECREF(device_timestamp);
    Py_DECREF(timestamp);
    Py_DECREF(digital_input);
    Py_DECREF(gpio1);
    Py_DECREF(gpio2);
    Py_DECREF(frame_number);
    Py_DECREF(camera_buffers);
    Py_DECREF(used_camera_buffers);
    Py_DECREF(height);
    Py_DECREF(width);

    return info;
}
