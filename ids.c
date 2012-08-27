#include <Python.h>
#include <structmember.h>
#include <ueye.h>

#include "ids.h"

PyMODINIT_FUNC initids(void) {
    PyObject* m;

    ids_CameraType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ids_CameraType) < 0) {
        return;
    }

    m = Py_InitModule("ids", idsMethods);

    if (m == NULL) {
        return NULL;
    }

    Py_INCREF(&ids_CameraType);
    PyModule_AddObject(m, "Camera", (PyObject *) &ids_CameraType);
}

int main(int argc, char *argv[]) {
    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(argv[0]);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Add a static module */
    initids();

    return 0;
}
