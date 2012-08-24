#include <Python.h>
#include <ueye.h>

static PyObject *ids_number_cameras(PyObject *self, PyObject *args);
static PyObject *ids_camera_list(PyObject *self, PyObject *args);

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
    {"camera_list", ids_camera_list, METH_VARARGS, "Information on each detected camera."},
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

static PyObject *ids_camera_list(PyObject *self, PyObject *args) {
    UEYE_CAMERA_LIST    cameras;

    is_GetNumberOfCameras(&cameras.dwCount);
    is_GetCameraList(&cameras);

    PyObject *dict = PyDict_New();
    PyObject *list = PyList_New(0);

    PyDict_SetItemString(dict, "dwCount", PyInt_FromLong(cameras.dwCount));
    PyDict_SetItemString(dict, "uci", list);

    for (int i = 0; i < cameras.dwCount; i++) {
        PyObject *camera_info = PyDict_New();

        PyDict_SetItemString(camera_info, "dwCameraId", Py_BuildValue("I", cameras.uci[i].dwCameraID));
        PyDict_SetItemString(camera_info, "dwDeviceId", Py_BuildValue("I", cameras.uci[i].dwDeviceID));
        PyDict_SetItemString(camera_info, "dwSensorId", Py_BuildValue("I", cameras.uci[i].dwSensorID));
        PyDict_SetItemString(camera_info, "dwInUse", Py_BuildValue("I", cameras.uci[i].dwInUse));
        PyDict_SetItemString(camera_info, "SerNo", Py_BuildValue("s", cameras.uci[i].SerNo));
        PyDict_SetItemString(camera_info, "Model", Py_BuildValue("s", cameras.uci[i].Model));
        PyDict_SetItemString(camera_info, "dwStatus", Py_BuildValue("I", cameras.uci[i].dwStatus));

        PyList_Append(list, camera_info);
    }

    return dict;
}
