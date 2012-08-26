#include <Python.h>
#include "structmember.h"
#include <ueye.h>

#include "ids.h"

static PyObject *ids_number_cameras(PyObject *self, PyObject *args);
static PyObject *ids_camera_list(PyObject *self, PyObject *args);

PyMethodDef idsMethods[] = {
    {"number_cameras", ids_number_cameras, METH_VARARGS, "Return the number of cameras connected."},
    {"camera_list", ids_camera_list, METH_VARARGS, "Information on each detected camera."},
    {NULL, NULL, 0, NULL}
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

