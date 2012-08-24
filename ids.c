#include <Python.h>
#include "structmember.h"
#include <ueye.h>

typedef struct {
    PyObject_HEAD;
    HIDS handle;
    /* Type fields here */
} ids_Camera;

static PyMemberDef ids_Camera_members[] = {
    {"handle", T_UINT, offsetof(ids_Camera, handle), 0, "camera handle"},
    {NULL}
};

static PyObject *ids_number_cameras(PyObject *self, PyObject *args) {
    UEYE_CAMERA_LIST    cameras;

    is_GetNumberOfCameras((int*) &cameras.dwCount);

    return Py_BuildValue("i", cameras.dwCount);
}

static PyObject *ids_camera_list(PyObject *self, PyObject *args) {
    UEYE_CAMERA_LIST    cameras;

    is_GetNumberOfCameras((int *) &cameras.dwCount);
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

static PyObject *ids_Camera_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    ids_Camera *self;

    self = (ids_Camera *) type->tp_alloc(type, 0);

    if (self != NULL) {
        self->handle = -1;
    }

    return (PyObject *) self;
}

static int ids_Camera_init(ids_Camera *self, PyObject *args, PyObject *kwds) {
    static char *kwlist[] = {"handle", NULL};

    self->handle = 0;
    if (!PyArg_ParseTupleAndKeywords(args, kwds, "|i", kwlist, &self->handle)) {
        return -1;
    }

    /* TODO: Add (more) error checking */
    if (is_InitCamera(&self->handle, NULL) != IS_SUCCESS) {
        return -1;
    }

    return 0;
}

static PyObject *ids_Camera_close(ids_Camera *self, PyObject *args, PyObject *kwds) {
    if (is_ExitCamera(self->handle) != IS_SUCCESS) {
        Py_INCREF(Py_False);
        return Py_False;
    }

    Py_INCREF(Py_True);
    return Py_True;
}

static PyMethodDef ids_Camera_methods[] = {
    {"close", (PyCFunction) ids_Camera_close, METH_VARARGS, "Closes open camera"},
    {NULL}
};

static PyTypeObject ids_CameraType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    "ids.Camera",              /* tp_name */
    sizeof(ids_Camera),        /* tp_basicsize */
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
    0,                         /* tp_traverse */
    0,                         /* tp_clear */
    0,                         /* tp_richcompare */
    0,                         /* tp_weaklistoffset */
    0,                         /* tp_iter */
    0,                         /* tp_iternext */
    ids_Camera_methods,        /* tp_methods */
    ids_Camera_members,        /* tp_members */
    0,                         /* tp_getset */
    0,                         /* tp_base */
    0,                         /* tp_dict */
    0,                         /* tp_descr_get */
    0,                         /* tp_descr_set */
    0,                         /* tp_dictoffset */
    (initproc)ids_Camera_init, /* tp_init */
    0,                         /* tp_alloc */
    ids_Camera_new,            /* tp_new */
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
