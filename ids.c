#include <Python.h>
#include <structmember.h>
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
