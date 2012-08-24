#include <Python.h>
#include <ueye.h>

static PyObject *ids_numcams(PyObject *self, PyObject *args);

static PyMethodDef  idsMethods[] = {
    {"numcams", ids_numcams, METH_VARARGS, "Return the number of cameras connected."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initids(void) {
    (void) Py_InitModule("ids", idsMethods);
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

static PyObject *ids_numcams(PyObject *self, PyObject *args) {
    UEYE_CAMERA_LIST    cameras;

    is_GetNumberOfCameras((int*) &cameras.dwCount);

    return Py_BuildValue("i", cameras.dwCount);
}
