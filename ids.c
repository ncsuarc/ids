#include <Python.h>
#include <ueye.h>

static PyObject *ids_numcams(PyObject *self, PyObject *args);

typedef struct {
    PyObject_HEAD;
    /* Type fields here */
} ids_CameraObject;

static PyTypeObject ids_CameraType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ids.Camera",              /* tp_name */
    sizeof(ids_CameraObject),  /* tp_basicsize */
    0,                         /* tp_itemsize */
    0,                         /* tp_dealloc */
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
    "IDS Camera Object",       /* tp_doc */
};

static PyMethodDef  idsMethods[] = {
    {"number_cameras", ids_number_cameras, METH_VARARGS, "Return the number of cameras connected."},
    {NULL, NULL, 0, NULL}
};

PyMODINIT_FUNC initids(void) {
    PyObject* m;

    ids_CameraType.tp_new = PyType_GenericNew;
    if (PyType_Ready(&ids_CameraType) < 0) {
        return;
    }

    m = Py_InitModule("ids", idsMethods);

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

static PyObject *ids_number_cameras(PyObject *self, PyObject *args) {
    UEYE_CAMERA_LIST    cameras;

    is_GetNumberOfCameras((int*) &cameras.dwCount);

    return Py_BuildValue("i", cameras.dwCount);
}

//static PyObject *ids_camera_list(PyObject *self, PyObject *args) {
//    UEYE_CAMERA_LIST    cameras;
//
//    is_GetCameraList(&cameras);
//}
