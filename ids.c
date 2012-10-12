#include <Python.h>
#include <structmember.h>
#include <datetime.h>
#include <ueye.h>

#define PY_ARRAY_UNIQUE_SYMBOL  ids_ARRAY_API
#include <numpy/arrayobject.h>

#include "ids.h"

#if PY_MAJOR_VERSION >= 3
static struct PyModuleDef idsmodule = {
    PyModuleDef_HEAD_INIT,
    "ids",    /* name of module */
    NULL, /* module documentation, may be NULL */
    -1,       /* size of per-interpreter state of the module,
                or -1 if the module keeps state in global variables. */
    idsMethods
};
#endif

#if PY_MAJOR_VERSION >= 3
PyMODINIT_FUNC PyInit_ids(void) {
#else
PyMODINIT_FUNC initids(void) {
#endif
    PyObject* m;

    ids_CameraType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ids_CameraType) < 0) {
        #if PY_MAJOR_VERSION >= 3
        return NULL;
        #else
        return;
        #endif
    }

    import_array();
    PyDateTime_IMPORT;

    #if PY_MAJOR_VERSION >= 3
    m = PyModule_Create(&idsmodule);
    #else
    m = Py_InitModule("ids", idsMethods);
    #endif

    if (m == NULL) {
        #if PY_MAJOR_VERSION >= 3
        return NULL;
        #else
        return;
        #endif
    }

    Py_INCREF(&ids_CameraType);
    PyModule_AddObject(m, "Camera", (PyObject *) &ids_CameraType);

    add_constants(m);

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
    PyInit_ids();
    #else
    initids();
    #endif

    return 0;
}

/* Stupid hack, needs DateTime, which gets clobbered in other files */
PyObject *image_info(ids_Camera *self, int image_id) {
    UEYEIMAGEINFO image_info;

    int ret = is_GetImageInfo(self->handle, image_id, &image_info, sizeof(image_info));
    switch (ret) {
    case IS_SUCCESS:
        break;
    default:
        PyErr_SetString(PyExc_IOError, "Unable to retrieve image info.");
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

